use crate::gpu::{GpuContext, clear_pass};
use crate::renderer::overlay::TextOverlay;

mod overlay;

pub struct Renderer {
    pipeline: wgpu::RenderPipeline,
    grid: Option<(wgpu::RenderPipeline, wgpu::BindGroup)>,
    view_radius: f32,
    grid_half: f32,
    view_buffer: wgpu::Buffer,
    view_bind_group: wgpu::BindGroup,
    star_bind_group: wgpu::BindGroup,
    star_count: u32,
    overlay: TextOverlay,
}

impl Renderer {
    pub fn new(
        gpu: &GpuContext,
        stars: &wgpu::Buffer,
        star_count: usize,
        view_radius: f32,
        grid_half: Option<f32>,
        strategy_name: &str,
    ) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let pipeline =
            gpu.render_pipeline(&shader, additive_blend(), wgpu::PrimitiveTopology::TriangleList);

        let view_buffer = gpu.uniform_buffer(&view_params(gpu, view_radius, grid_half.unwrap_or(0.0)));
        let layout = |group| pipeline.get_bind_group_layout(group);

        let grid = grid_half.map(|_| {
            let shader = gpu
                .device
                .create_shader_module(wgpu::include_wgsl!("grid.wgsl"));
            let pipeline =
                gpu.render_pipeline(&shader, additive_blend(), wgpu::PrimitiveTopology::LineList);
            let bind_group = gpu.bind_group(&pipeline.get_bind_group_layout(0), &[&view_buffer]);
            (pipeline, bind_group)
        });

        Self {
            view_bind_group: gpu.bind_group(&layout(0), &[&view_buffer]),
            star_bind_group: gpu.bind_group(&layout(1), &[stars]),
            pipeline,
            grid,
            view_radius,
            grid_half: grid_half.unwrap_or(0.0),
            view_buffer,
            star_count: star_count as u32,
            overlay: TextOverlay::new(gpu, star_count, strategy_name),
        }
    }

    pub fn resize(&self, gpu: &GpuContext) {
        gpu.write(&self.view_buffer, &view_params(gpu, self.view_radius, self.grid_half));
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
        pass.set_bind_group(0, &self.view_bind_group, &[]);
        pass.set_bind_group(1, &self.star_bind_group, &[]);
        pass.draw(0..self.star_count * 6, 0..1);

        if let Some((pipeline, bind_group)) = &self.grid {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.draw(0..8, 0..1);
        }

        self.overlay.draw(&mut pass);
    }
}

fn view_params(gpu: &GpuContext, view_radius: f32, grid_half: f32) -> [f32; 4] {
    let aspect = gpu.config.width as f32 / gpu.config.height as f32;
    [1.0 / (view_radius * aspect), 1.0 / view_radius, grid_half, 0.0]
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
