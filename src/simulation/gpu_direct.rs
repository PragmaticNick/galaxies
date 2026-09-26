use wgpu::BufferBindingType::{Storage, Uniform};

use crate::gpu::{GpuContext, dispatch};
use crate::simulation::{EPS, G, Simulation};
use crate::star::Star;

const WORKGROUP_SIZE: usize = 256;

pub struct GpuDirect {
    workgroup_count: u32,
    params_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    drift_pipeline: wgpu::ComputePipeline,
    kick_pipeline: wgpu::ComputePipeline,
    commit_pipeline: wgpu::ComputePipeline,
}

impl Simulation for GpuDirect {
    fn init(gpu: &GpuContext, buffer: &wgpu::Buffer, stars: Vec<Star>) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::include_wgsl!("gpu_direct.wgsl"));
        let params_buffer = gpu.uniform_buffer(&[0.0f32; 4]);
        let layout = gpu.compute_bind_group_layout(&[Storage { read_only: false }, Uniform]);

        Self {
            workgroup_count: stars.len().div_ceil(WORKGROUP_SIZE) as u32,
            bind_group: gpu.bind_group(&layout, &[buffer, &params_buffer]),
            params_buffer,
            drift_pipeline: gpu.compute_pipeline(&shader, &layout, "drift_half"),
            kick_pipeline: gpu.compute_pipeline(&shader, &layout, "kick"),
            commit_pipeline: gpu.compute_pipeline(&shader, &layout, "commit"),
        }
    }

    fn step(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        _buffer: &wgpu::Buffer,
        dt: f32,
    ) {
        gpu.write(&self.params_buffer, &[dt, EPS, G, 0.0]);

        for pipeline in [
            &self.drift_pipeline,
            &self.kick_pipeline,
            &self.commit_pipeline,
        ] {
            dispatch(encoder, pipeline, &self.bind_group, self.workgroup_count);
        }
    }
}
