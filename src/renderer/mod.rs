use crate::gpu::{GpuContext, clear_pass};
use crate::renderer::overlay::TextOverlay;

mod overlay;

pub struct Renderer {
    pipeline: wgpu::RenderPipeline,
    screen_buffer: wgpu::Buffer,
    screen_bind_group: wgpu::BindGroup,
    star_bind_group: wgpu::BindGroup,
    star_count: u32,
    overlay: TextOverlay,
}

impl Renderer {
    pub fn new(
        gpu: &GpuContext,
        stars: &wgpu::Buffer,
        star_count: usize,
        strategy_name: &str,
    ) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let pipeline = gpu.render_pipeline(&shader, additive_blend());

        let screen_buffer = gpu.uniform_buffer(&screen_size(gpu));
        let layout = |group| pipeline.get_bind_group_layout(group);

        Self {
            screen_bind_group: gpu.bind_group(&layout(0), &[&screen_buffer]),
            star_bind_group: gpu.bind_group(&layout(1), &[stars]),
            pipeline,
            screen_buffer,
            star_count: star_count as u32,
            overlay: TextOverlay::new(gpu, star_count, strategy_name),
        }
    }

    /// Call after `GpuContext::resize`.
    pub fn resize(&self, gpu: &GpuContext) {
        gpu.write(&self.screen_buffer, &screen_size(gpu));
    }

    pub fn draw(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        self.overlay.prepare(gpu);

        let mut pass = clear_pass(encoder, view, wgpu::Color::BLACK);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.screen_bind_group, &[]);
        pass.set_bind_group(1, &self.star_bind_group, &[]);
        pass.draw(0..self.star_count * 6, 0..1);

        self.overlay.draw(&mut pass);
    }
}

fn screen_size(gpu: &GpuContext) -> [f32; 2] {
    [gpu.config.width as f32, gpu.config.height as f32]
}

fn additive_blend() -> wgpu::BlendState {
    let additive = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::SrcAlpha,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    };
    wgpu::BlendState {
        color: additive,
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            ..additive
        },
    }
}
