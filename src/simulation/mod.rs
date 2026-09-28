use crate::gpu::GpuContext;

mod cpu_direct;
mod gpu_direct;
mod gpu_particle_mesh;
mod particle_mesh;

pub use cpu_direct::CpuDirect;
pub use gpu_direct::GpuDirect;
pub use gpu_particle_mesh::GpuParticleMesh;
pub use particle_mesh::ParticleMesh;

pub const G: f32 = 100.0;
pub const EPS: f32 = 8.0;

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
