use crate::config::{
    BLOOM_STRENGTH, EXPOSURE, GLOW_SIZE, GLOW_STRENGTH, GRID_BRIGHTNESS, GRID_CELL_BRIGHTNESS, MIN_RADIUS_PX,
    SHOW_OVERLAY,
};
use crate::gpu::{GpuContext, clear_pass};
use crate::renderer::bloom::Bloom;
use crate::renderer::overlay::TextOverlay;

mod bloom;
mod overlay;

const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

pub struct Renderer {
    pipeline: wgpu::RenderPipeline,
    tonemap: wgpu::RenderPipeline,
    hdr_view: wgpu::TextureView,
    bloom: Bloom,
    tonemap_bind_group: wgpu::BindGroup,
    grid: Option<(wgpu::RenderPipeline, wgpu::BindGroup)>,
    view_radius: f32,
    center: [f32; 2],
    grid_half: f32,
    grid_cells: u32,
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
        grid: Option<(i32, f32)>,
        strategy_name: &str,
    ) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let pipeline = gpu.render_pipeline(
            &shader,
            HDR_FORMAT,
            additive_blend(),
            wgpu::PrimitiveTopology::TriangleList,
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
            wgpu::BlendState::REPLACE,
            wgpu::PrimitiveTopology::TriangleList,
            &[("EXPOSURE", EXPOSURE as f64), ("BLOOM_STRENGTH", BLOOM_STRENGTH as f64)],
        );
        let hdr_view = hdr_texture(gpu);
        let bloom = Bloom::new(gpu, HDR_FORMAT, &hdr_view);
        let tonemap_bind_group = tonemap_bind_group(gpu, &tonemap, &hdr_view, &bloom);

        let view_buffer = gpu.uniform_buffer(&[0.0f32; 8]);
        let layout = |group| pipeline.get_bind_group_layout(group);

        let (grid_cells, grid_half) = grid.map_or((0, 0.0), |(size, h)| (size as u32, size as f32 * h / 2.0));
        let grid = grid.map(|_| {
            let shader = gpu
                .device
                .create_shader_module(wgpu::include_wgsl!("grid.wgsl"));
            let pipeline = gpu.render_pipeline(
                &shader,
                HDR_FORMAT,
                additive_blend(),
                wgpu::PrimitiveTopology::LineList,
                &[
                    ("GRID_BRIGHTNESS", GRID_BRIGHTNESS as f64),
                    ("GRID_CELL_BRIGHTNESS", GRID_CELL_BRIGHTNESS as f64),
                ],
            );
            let bind_group = gpu.bind_group(&pipeline.get_bind_group_layout(0), &[&view_buffer]);
            (pipeline, bind_group)
        });

        let renderer = Self {
            view_bind_group: gpu.bind_group(&layout(0), &[&view_buffer]),
            star_bind_group: gpu.bind_group(&layout(1), &[stars]),
            pipeline,
            tonemap,
            hdr_view,
            bloom,
            tonemap_bind_group,
            grid,
            view_radius,
            center: [0.0, 0.0],
            grid_half,
            grid_cells,
            view_buffer,
            star_count: star_count as u32,
            overlay: TextOverlay::new(gpu, star_count, strategy_name),
        };
        renderer.update_view(gpu);
        renderer
    }

    pub fn resize(&mut self, gpu: &GpuContext) {
        self.hdr_view = hdr_texture(gpu);
        self.bloom.resize(gpu, HDR_FORMAT, &self.hdr_view);
        self.tonemap_bind_group = tonemap_bind_group(gpu, &self.tonemap, &self.hdr_view, &self.bloom);
        self.update_view(gpu);
    }

    fn update_view(&self, gpu: &GpuContext) {
        let aspect = gpu.config.width as f32 / gpu.config.height as f32;
        let r = self.view_radius;
        let [cx, cy] = self.center;
        gpu.write(&self.view_buffer, &[1.0 / (r * aspect), 1.0 / r, cx, cy, self.grid_half, self.world_per_pixel(gpu), self.grid_cells as f32, 0.0]);
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
    ) {
        if SHOW_OVERLAY {
            self.overlay.prepare(gpu);
        }

        let mut pass = clear_pass(encoder, &self.hdr_view, wgpu::Color::BLACK);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.view_bind_group, &[]);
        pass.set_bind_group(1, &self.star_bind_group, &[]);
        pass.draw(0..self.star_count * 6, 0..1);

        if let Some((pipeline, bind_group)) = &self.grid {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.draw(0..4 * (self.grid_cells + 1), 0..1);
        }
        drop(pass);

        self.bloom.draw(encoder);

        let mut pass = clear_pass(encoder, view, wgpu::Color::BLACK);
        pass.set_pipeline(&self.tonemap);
        pass.set_bind_group(0, &self.tonemap_bind_group, &[]);
        pass.draw(0..3, 0..1);

        if SHOW_OVERLAY {
            self.overlay.draw(&mut pass);
        }
    }
}

fn hdr_texture(gpu: &GpuContext) -> wgpu::TextureView {
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: gpu.config.width,
            height: gpu.config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HDR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    texture.create_view(&Default::default())
}

fn tonemap_bind_group(
    gpu: &GpuContext,
    tonemap: &wgpu::RenderPipeline,
    hdr: &wgpu::TextureView,
    bloom: &Bloom,
) -> wgpu::BindGroup {
    gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &tonemap.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(hdr),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(bloom.output()),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&bloom.sampler),
            },
        ],
    })
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
