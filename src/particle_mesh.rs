use bytemuck::{Pod, Zeroable};
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use wgpu::BufferBindingType::{Storage, Uniform};

use crate::config::{EPS, G};
use crate::gpu::{GpuContext, dispatch};
use crate::star::Star;

const WORKGROUP_SIZE: usize = 256;
const MAX_P: usize = 2048;
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
    gm: f32,
    eps2: f32,
}

pub struct ParticleMesh {
    params: Params,
    params_buffer: wgpu::Buffer,
    mass_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    passes: Vec<(wgpu::ComputePipeline, u32)>,
}

impl ParticleMesh {
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
            gm: G * total_mass,
            eps2: EPS * EPS,
        };

        let kernel: Vec<[f32; 2]> = kernel.iter().map(|k| [k.re, k.im]).collect();
        let params_buffer = gpu.uniform_buffer(&[params]);
        let mass_buffer = gpu.storage_buffer(&vec![0i32; n * n]);
        let field_buffer = gpu.storage_buffer(&vec![[0.0f32; 2]; p * p]);
        let kernel_buffer = gpu.storage_buffer(&kernel);

        let shader = gpu
            .device
            .create_shader_module(wgpu::include_wgsl!("particle_mesh.wgsl"));
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

    pub fn step(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, dt: f32) {
        self.params.dt = dt;
        gpu.write(&self.params_buffer, &[self.params]);
        encoder.clear_buffer(&self.mass_buffer, 0, None);

        for (pipeline, groups) in &self.passes {
            dispatch(encoder, pipeline, &self.bind_group, *groups);
        }
    }
}

// The softened pull on a unit mass at every grid offset, zero-padded to p x p so the convolution
// isn't periodic, and transformed once up front: on the GPU it is only multiplied in.
fn fft_kernel(size: i32, h: f32) -> (usize, Vec<Complex32>) {
    let n = (size + 1) as usize;
    let p = (2 * n - 1).next_power_of_two();

    let mut kernel = vec![Complex32::ZERO; p * p];
    let reach = n as i32 - 1;
    for ui in -reach..=reach {
        for uj in -reach..=reach {
            let dx = ui as f32 * h;
            let dy = uj as f32 * h;
            let r2 = dx * dx + dy * dy + EPS * EPS;
            let k = -G / (r2 * r2.sqrt());

            let wi = ui.rem_euclid(p as i32) as usize;
            let wj = uj.rem_euclid(p as i32) as usize;
            kernel[wi * p + wj] = Complex32::new(k * dx, k * dy);
        }
    }

    let forward = FftPlanner::new().plan_fft_forward(p);
    let mut transposed = vec![Complex32::ZERO; p * p];
    fft_2d(forward.as_ref(), &mut kernel, &mut transposed, p);

    (p, kernel)
}

fn fft_2d(fft: &dyn Fft<f32>, data: &mut [Complex32], transposed: &mut [Complex32], p: usize) {
    let transpose = |from: &[Complex32], to: &mut [Complex32]| {
        for (r, row) in to.chunks_mut(p).enumerate() {
            for (c, value) in row.iter_mut().enumerate() {
                *value = from[c * p + r];
            }
        }
    };

    // process() runs one FFT per p-long row of the buffer
    fft.process(data);
    transpose(data, transposed);
    fft.process(transposed);
    transpose(transposed, data);
}
