use bytemuck::{Pod, Zeroable};
use wgpu::BufferBindingType::{Storage, Uniform};

use crate::gpu::{GpuContext, dispatch};
use crate::simulation::Simulation;
use crate::simulation::particle_mesh::fft_kernel;
use crate::star::Star;

const WORKGROUP_SIZE: usize = 256;
const MAX_P: usize = 1024;
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

pub struct GpuParticleMesh {
    params: Params,
    params_buffer: wgpu::Buffer,
    mass_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    passes: Vec<(wgpu::ComputePipeline, u32)>,
}

impl GpuParticleMesh {
    pub fn new(gpu: &GpuContext, buffer: &wgpu::Buffer, stars: &[Star], size: i32, h: f32) -> Self {
        let (p, kernel) = fft_kernel(size, h);
        assert!(p <= MAX_P, "grid {size} needs FFT side {p}, max is {MAX_P}");

        let n = (size + 1) as usize;
        let half = size as f32 * h / 2.0;
        let total_mass: f32 = stars.iter().map(|s| s.mass).sum();
        let params = Params {
            dt: 0.0,
            h,
            origin: [-half, -half],
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

        let star_groups = stars.len().div_ceil(WORKGROUP_SIZE) as u32;
        let field_groups = (p * p).div_ceil(WORKGROUP_SIZE) as u32;
        let lines = p as u32;
        let passes = [
            ("drift_deposit", star_groups),
            ("load", field_groups),
            ("fft_rows_forward", lines),
            ("fft_cols_forward", lines),
            ("multiply", field_groups),
            ("fft_rows_inverse", lines),
            ("fft_cols_inverse", lines),
            ("kick_drift", star_groups),
        ]
        .map(|(entry, groups)| (gpu.compute_pipeline(&shader, &layout, entry), groups))
        .to_vec();

        Self {
            bind_group: gpu.bind_group(
                &layout,
                &[buffer, &mass_buffer, &field_buffer, &kernel_buffer, &params_buffer],
            ),
            params,
            params_buffer,
            mass_buffer,
            passes,
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

        for (pipeline, groups) in &self.passes {
            dispatch(encoder, pipeline, &self.bind_group, *groups);
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

        let mut on_gpu = GpuParticleMesh::new(&gpu, &buffer, &stars, 63, 8.0);
        let mut on_cpu = ParticleMesh::new_fft(stars.clone(), 63, 8.0);
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
