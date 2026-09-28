use rayon::prelude::*;

use crate::gpu::GpuContext;
use crate::star::Star;

mod cpu_direct;
mod gpu_direct;
mod gpu_particle_mesh;
mod particle_mesh;

pub use cpu_direct::CpuDirect;
pub use gpu_direct::GpuDirect;
pub use gpu_particle_mesh::GpuParticleMesh;
pub use particle_mesh::ParticleMesh;

pub use crate::config::{EPS, G};

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub enum Strategy {
    GpuDirect,
    CpuDirect,
    ParticleMesh {
        size: i32,
        h: f32,
    },
    ParticleMeshFft {
        size: i32,
        h: f32,
    },
    GpuParticleMeshFft {
        size: i32,
        h: f32,
    },
}

impl Strategy {
    pub fn name(self) -> &'static str {
        match self {
            Strategy::GpuDirect => "gpu direct",
            Strategy::CpuDirect => "cpu direct",
            Strategy::ParticleMesh { .. } => "particle mesh",
            Strategy::ParticleMeshFft { .. } => "particle mesh fft",
            Strategy::GpuParticleMeshFft { .. } => "gpu particle mesh fft",
        }
    }

    pub fn grid(self) -> Option<(i32, f32)> {
        match self {
            Strategy::ParticleMesh { size, h }
            | Strategy::ParticleMeshFft { size, h }
            | Strategy::GpuParticleMeshFft { size, h } => Some((size, h)),
            _ => None,
        }
    }
}

pub fn drift_half(stars: &mut [Star], dt: f32) {
    stars.par_iter_mut().for_each(|s| {
        s.pos[0] += 0.5 * s.vel[0] * dt;
        s.pos[1] += 0.5 * s.vel[1] * dt;
    });
}

pub trait Simulation {
    fn step(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        buffer: &wgpu::Buffer,
        dt: f32,
    );
}
