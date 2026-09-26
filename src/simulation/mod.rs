use crate::gpu::GpuContext;
use crate::star::Star;

mod cpu_direct;
mod gpu_direct;

pub use cpu_direct::CpuDirect;
pub use gpu_direct::GpuDirect;

pub const G: f32 = 100.0;
pub const EPS: f32 = 30.0;

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub enum Strategy {
    GpuDirect,
    CpuDirect,
}

impl Strategy {
    pub fn name(self) -> &'static str {
        match self {
            Strategy::GpuDirect => "gpu direct",
            Strategy::CpuDirect => "cpu direct",
        }
    }
}

pub trait Simulation {
    fn init(gpu: &GpuContext, buffer: &wgpu::Buffer, stars: Vec<Star>) -> Self
    where
        Self: Sized;

    fn step(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        buffer: &wgpu::Buffer,
        dt: f32,
    );
}
