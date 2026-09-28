use bytemuck::{Pod, Zeroable};
use wgpu::BufferBindingType::{Storage, Uniform};

use crate::gpu::{GpuContext, dispatch};
use crate::simulation::Simulation;
use crate::simulation::particle_mesh::fft_kernel;
use crate::star::Star;

const WORKGROUP_SIZE: usize = 256;

/// Must match MAX_P in the shader: one FFT line has to fit in shared memory.
const MAX_P: usize = 1024;

/// Node masses are summed with integer atomics (WGSL has no float ones), in
/// fixed point. The scale puts the whole galaxy's mass at 2^30, so no node
/// can overflow an i32.
const FIXED_POINT_TOTAL: f32 = (1u32 << 30) as f32;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    dt: f32,
    h: f32,
    origin: [f32; 2],
    size: u32,
    p: u32,
    log_p: u32,
    mass_scale: f32,
}

/// Particle mesh with the FFT node solver, all on the GPU. Same algorithm as
/// `ParticleMesh::new_fft`.
pub struct GpuParticleMesh {
    params: Params,
    params_buffer: wgpu::Buffer,
    mass_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    star_workgroups: u32,
    field_workgroups: u32,
    drift_deposit: wgpu::ComputePipeline,
    load: wgpu::ComputePipeline,
    fft_rows_forward: wgpu::ComputePipeline,
    fft_cols_forward: wgpu::ComputePipeline,
    multiply: wgpu::ComputePipeline,
    fft_rows_inverse: wgpu::ComputePipeline,
    fft_cols_inverse: wgpu::ComputePipeline,
    kick_drift: wgpu::ComputePipeline,
}

impl GpuParticleMesh {
    pub fn new(
        gpu: &GpuContext,
        buffer: &wgpu::Buffer,
        stars: &[Star],
        size: i32,
        h: f32,
        x0: f32,
        y0: f32,
    ) -> Self {
        let (p, kernel) = fft_kernel(size, h);
        assert!(p <= MAX_P, "grid {size} needs FFT side {p}, max is {MAX_P}");

        let n = (size + 1) as usize;
        let total_mass: f32 = stars.iter().map(|s| s.mass).sum();
        let params = Params {
            dt: 0.0,
            h,
            origin: [x0, y0],
            size: size as u32,
            p: p as u32,
            log_p: p.trailing_zeros(),
            mass_scale: FIXED_POINT_TOTAL / total_mass,
        };

        let kernel: Vec<[f32; 2]> = kernel.iter().map(|k| [k.re, k.im]).collect();

        let params_buffer = gpu.uniform_buffer(&[params]);
        let mass_buffer = gpu.storage_buffer(&vec![0i32; n * n]);
        let field_buffer = gpu.storage_buffer(&vec![[0.0f32; 2]; p * p]);
        let kernel_buffer = gpu.storage_buffer(&kernel);

        let shader = gpu
            .device
            .create_shader_module(wgpu::include_wgsl!("gpu_particle_mesh.wgsl"));
        let layout = gpu.compute_bind_group_layout(&[
            Storage { read_only: false },
            Storage { read_only: false },
            Storage { read_only: false },
            Storage { read_only: true },
            Uniform,
        ]);
        let pipeline = |entry| gpu.compute_pipeline(&shader, &layout, entry);

        Self {
            bind_group: gpu.bind_group(
                &layout,
                &[buffer, &mass_buffer, &field_buffer, &kernel_buffer, &params_buffer],
            ),
            params,
            params_buffer,
            mass_buffer,
            star_workgroups: stars.len().div_ceil(WORKGROUP_SIZE) as u32,
            field_workgroups: (p * p).div_ceil(WORKGROUP_SIZE) as u32,
            drift_deposit: pipeline("drift_deposit"),
            load: pipeline("load"),
            fft_rows_forward: pipeline("fft_rows_forward"),
            fft_cols_forward: pipeline("fft_cols_forward"),
            multiply: pipeline("multiply"),
            fft_rows_inverse: pipeline("fft_rows_inverse"),
            fft_cols_inverse: pipeline("fft_cols_inverse"),
            kick_drift: pipeline("kick_drift"),
        }
    }
}

impl Simulation for GpuParticleMesh {
    fn step(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        _buffer: &wgpu::Buffer,
        dt: f32,
    ) {
        self.params.dt = dt;
        gpu.write(&self.params_buffer, &[self.params]);
        encoder.clear_buffer(&self.mass_buffer, 0, None);

        // one workgroup per FFT line
        let lines = self.params.p;

        for (pipeline, workgroups) in [
            (&self.drift_deposit, self.star_workgroups),
            (&self.load, self.field_workgroups),
            (&self.fft_rows_forward, lines),
            (&self.fft_cols_forward, lines),
            (&self.multiply, self.field_workgroups),
            (&self.fft_rows_inverse, lines),
            (&self.fft_cols_inverse, lines),
            (&self.kick_drift, self.star_workgroups),
        ] {
            dispatch(encoder, pipeline, &self.bind_group, workgroups);
        }
    }
}

#[cfg(test)]
mod tests {
    use wgpu::util::DeviceExt as _;

    use super::*;
    use crate::galaxy::{GalaxyConfig, generate_galaxy};
    use crate::simulation::ParticleMesh;

    #[test]
    fn gpu_matches_cpu() {
        let Ok(gpu) = pollster::block_on(GpuContext::headless()) else {
            eprintln!("no GPU adapter, skipping");
            return;
        };
        let stars = generate_galaxy(&GalaxyConfig {
            center: [0.0, 0.0],
            radius: 200.0,
            star_count: 5000,
            core_mass: 500000.0,
            core_radius: 2.0,
            star_mass: 100.0,
            star_radius: 1.5,
            gap: 20.0,
            arms: 2,
        });

        let buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&stars),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut on_gpu = GpuParticleMesh::new(&gpu, &buffer, &stars, 63, 8.0, -252.0, -252.0);
        let mut on_cpu = ParticleMesh::new_fft(stars.clone(), 63, 8.0, -252.0, -252.0);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        for _ in 0..10 {
            on_gpu.step(&gpu, &mut encoder, &buffer, 0.004);
            on_cpu.update(0.004);
        }
        encoder.copy_buffer_to_buffer(&buffer, 0, &staging, 0, buffer.size());
        gpu.queue.submit([encoder.finish()]);

        staging.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let mapped = staging.slice(..).get_mapped_range();
        let result: &[Star] = bytemuck::cast_slice(&mapped);

        for (g, c) in result.iter().zip(on_cpu.stars()) {
            assert!((g.pos[0] - c.pos[0]).hypot(g.pos[1] - c.pos[1]) < 0.01);
        }
    }
}
