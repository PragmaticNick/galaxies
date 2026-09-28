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
    use super::*;
    use crate::galaxy::{GalaxyConfig, generate_galaxy};
    use crate::simulation::ParticleMesh;

    fn read_stars(gpu: &GpuContext, buffer: &wgpu::Buffer, count: usize) -> Vec<Star> {
        let size = (count * size_of::<Star>()) as u64;
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        gpu.queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let stars = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        stars
    }

    /// Step time of the GPU PM for growing star counts and grids. Run with:
    /// cargo test --release gpu_pm_speed -- --ignored --nocapture
    #[test]
    #[ignore]
    fn gpu_pm_speed() {
        let gpu = pollster::block_on(GpuContext::headless()).unwrap();
        let steps = 100;

        for star_count in [50_000, 500_000, 2_000_000] {
            let galaxy = GalaxyConfig {
                center: [0.0, 0.0],
                radius: 400.0,
                star_count,
                core_mass: 500000.0,
                core_radius: 2.0,
                star_mass: 500000.0 / star_count as f32,
                star_radius: 1.5,
                gap: 20.0,
                arms: 2,
            };
            let stars = generate_galaxy(&galaxy);
            let buffer = gpu.storage_buffer(&stars);

            for (size, h) in [(127, 10.0), (255, 5.0), (511, 2.5)] {
                let half = size as f32 * h / 2.0;
                let mut pm = GpuParticleMesh::new(&gpu, &buffer, &stars, size, h, -half, -half);

                let mut run = |steps| {
                    let mut encoder = gpu.device.create_command_encoder(&Default::default());
                    for _ in 0..steps {
                        pm.step(&gpu, &mut encoder, &buffer, 0.001);
                    }
                    gpu.queue.submit([encoder.finish()]);
                    gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                };
                run(1);
                let start = std::time::Instant::now();
                run(steps);
                let ms = start.elapsed().as_secs_f64() * 1000.0 / steps as f64;
                println!("{star_count:>9} stars, grid {size:3}x{size:3}: {ms:6.3} ms/step");
            }
        }
    }

    /// Same galaxy, same grid, several steps on the GPU and with the CPU FFT
    /// PM: positions must agree up to float and fixed-point rounding.
    #[test]
    fn gpu_pm_matches_cpu() {
        let Ok(gpu) = pollster::block_on(GpuContext::headless()) else {
            eprintln!("no GPU adapter, skipping");
            return;
        };

        let galaxy = GalaxyConfig {
            center: [0.0, 0.0],
            radius: 200.0,
            star_count: 5000,
            core_mass: 500000.0,
            core_radius: 2.0,
            star_mass: 10.0,
            star_radius: 1.5,
            gap: 20.0,
            arms: 2,
        };
        let stars = generate_galaxy(&galaxy);
        let (size, h) = (63, 8.0);
        let half = size as f32 * h / 2.0;
        let dt = 0.004;
        let steps = 10;

        let buffer = wgpu::util::DeviceExt::create_buffer_init(
            &gpu.device,
            &wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&stars),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            },
        );
        let mut gpu_pm = GpuParticleMesh::new(&gpu, &buffer, &stars, size, h, -half, -half);
        for _ in 0..steps {
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            gpu_pm.step(&gpu, &mut encoder, &buffer, dt);
            gpu.queue.submit([encoder.finish()]);
        }
        let on_gpu = read_stars(&gpu, &buffer, stars.len());

        let mut cpu_pm = ParticleMesh::new_fft(stars.clone(), size, h, -half, -half);
        for _ in 0..steps {
            cpu_pm.update(dt);
        }

        let mut max_error: f32 = 0.0;
        let mut max_travel: f32 = 0.0;
        for ((g, c), s) in on_gpu.iter().zip(cpu_pm.stars()).zip(&stars) {
            let error = ((g.pos[0] - c.pos[0]).powi(2) + (g.pos[1] - c.pos[1]).powi(2)).sqrt();
            let travel = ((c.pos[0] - s.pos[0]).powi(2) + (c.pos[1] - s.pos[1]).powi(2)).sqrt();
            max_error = max_error.max(error);
            max_travel = max_travel.max(travel);
        }
        println!("max error {max_error:.2e}, max travel {max_travel:.2}");
        assert!(max_error < max_travel * 1.0e-3, "max error {max_error}, max travel {max_travel}");
    }
}
