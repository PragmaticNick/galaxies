use crate::config::{
    BLOOM_STRENGTH, EXPOSURE, GLOW_SIZE, GLOW_STRENGTH, MIN_RADIUS_PX, SHOW_OVERLAY, VIEW_RADIUS,
};
use crate::gpu::{ADDITIVE, GpuContext, render_pass};
use crate::renderer::background::Background;
use crate::renderer::bloom::Bloom;
use crate::renderer::overlay::TextOverlay;

mod background;
mod bloom;
mod overlay;

const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

pub struct Renderer {
    pipeline: wgpu::RenderPipeline,
    tonemap: wgpu::RenderPipeline,
    background: Background,
    hdr_view: wgpu::TextureView,
    bloom: Bloom,
    tonemap_bind_group: wgpu::BindGroup,
    view_radius: f32,
    center: [f32; 2],
    view_buffer: wgpu::Buffer,
    view_bind_group: wgpu::BindGroup,
    star_bind_group: wgpu::BindGroup,
    star_count: u32,
    overlay: Option<TextOverlay>,
}

impl Renderer {
    pub fn new(gpu: &GpuContext, stars: &wgpu::Buffer, star_count: usize) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let pipeline = gpu.render_pipeline(
            &shader,
            HDR_FORMAT,
            ADDITIVE,
            &[
                ("MIN_RADIUS_PX", MIN_RADIUS_PX as f64),
                ("GLOW_SIZE", GLOW_SIZE as f64),
                ("GLOW_STRENGTH", GLOW_STRENGTH as f64),
            ],
        );

        let shader = gpu
            .device
            .create_shader_module(wgpu::include_wgsl!("tonemap.wgsl"));
        let tonemap = gpu.render_pipeline(
            &shader,
            gpu.config.format,
            // tonemapped galaxy light adds on top of the background
            ADDITIVE,
            &[("EXPOSURE", EXPOSURE as f64), ("BLOOM_STRENGTH", BLOOM_STRENGTH as f64)],
        );
        let hdr_view = hdr_target(gpu);
        let bloom = Bloom::new(gpu, HDR_FORMAT, &hdr_view);
        let tonemap_bind_group = tonemap_bind_group(gpu, &tonemap, &hdr_view, &bloom);

        let view_buffer = gpu.uniform_buffer(&[0.0f32; 6]);
        let layout = |group| pipeline.get_bind_group_layout(group);

        let renderer = Self {
            view_bind_group: gpu.bind_group(&layout(0), &[&view_buffer]),
            star_bind_group: gpu.bind_group(&layout(1), &[stars]),
            pipeline,
            tonemap,
            background: Background::new(gpu),
            hdr_view,
            bloom,
            tonemap_bind_group,
            view_radius: VIEW_RADIUS,
            center: [0.0, 0.0],
            view_buffer,
            star_count: star_count as u32,
            overlay: SHOW_OVERLAY.then(|| TextOverlay::new(gpu, star_count)),
        };
        renderer.update_view(gpu);
        renderer
    }

    pub fn resize(&mut self, gpu: &GpuContext) {
        self.hdr_view = hdr_target(gpu);
        self.bloom.resize(gpu, HDR_FORMAT, &self.hdr_view);
        self.tonemap_bind_group = tonemap_bind_group(gpu, &self.tonemap, &self.hdr_view, &self.bloom);
        self.update_view(gpu);
    }

    fn update_view(&self, gpu: &GpuContext) {
        let aspect = gpu.config.width as f32 / gpu.config.height as f32;
        let r = self.view_radius;
        let [cx, cy] = self.center;
        gpu.write(&self.view_buffer, &[1.0 / (r * aspect), 1.0 / r, cx, cy, self.world_per_pixel(gpu), 0.0]);
    }

    pub fn pan(&mut self, gpu: &GpuContext, dx: f32, dy: f32) {
        let k = self.world_per_pixel(gpu);
        self.center[0] -= dx * k;
        self.center[1] += dy * k;
        self.update_view(gpu);
    }

    pub fn zoom(&mut self, gpu: &GpuContext, factor: f32, cursor: [f32; 2]) {
        let before = self.to_world(gpu, cursor);
        self.view_radius *= factor;
        let after = self.to_world(gpu, cursor);
        self.center[0] += before[0] - after[0];
        self.center[1] += before[1] - after[1];
        self.update_view(gpu);
    }

    fn world_per_pixel(&self, gpu: &GpuContext) -> f32 {
        2.0 * self.view_radius / gpu.config.height as f32
    }

    fn to_world(&self, gpu: &GpuContext, pixel: [f32; 2]) -> [f32; 2] {
        let k = self.world_per_pixel(gpu);
        [
            self.center[0] + (pixel[0] - gpu.config.width as f32 / 2.0) * k,
            self.center[1] - (pixel[1] - gpu.config.height as f32 / 2.0) * k,
        ]
    }

    pub fn draw(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        time: f32,
    ) {
        if let Some(overlay) = &mut self.overlay {
            overlay.prepare(gpu);
        }

        let mut pass = render_pass(encoder, &self.hdr_view, wgpu::LoadOp::Clear(wgpu::Color::BLACK));
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.view_bind_group, &[]);
        pass.set_bind_group(1, &self.star_bind_group, &[]);
        pass.draw(0..self.star_count * 6, 0..1);
        drop(pass);

        self.bloom.draw(encoder);

        let mut pass = render_pass(encoder, view, wgpu::LoadOp::Clear(self.background.clear_color()));
        self.background.draw(gpu, &mut pass, time);
        pass.set_pipeline(&self.tonemap);
        pass.set_bind_group(0, &self.tonemap_bind_group, &[]);
        pass.draw(0..3, 0..1);

        if let Some(overlay) = &self.overlay {
            overlay.draw(&mut pass);
        }
    }
}

fn hdr_target(gpu: &GpuContext) -> wgpu::TextureView {
    gpu.render_target(gpu.config.width, gpu.config.height, HDR_FORMAT)
}

fn tonemap_bind_group(
    gpu: &GpuContext,
    tonemap: &wgpu::RenderPipeline,
    hdr: &wgpu::TextureView,
    bloom: &Bloom,
) -> wgpu::BindGroup {
    gpu.texture_bind_group(&tonemap.get_bind_group_layout(0), &[hdr, bloom.output()], &bloom.sampler)
}
