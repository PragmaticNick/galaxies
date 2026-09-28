use crate::config::BLOOM_LEVELS;
use crate::gpu::GpuContext;

pub struct Bloom {
    down: wgpu::RenderPipeline,
    up: wgpu::RenderPipeline,
    pub sampler: wgpu::Sampler,
    levels: Vec<Level>,
}

struct Level {
    view: wgpu::TextureView,
    down_source: wgpu::BindGroup,
    up_source: Option<wgpu::BindGroup>,
}

impl Bloom {
    pub fn new(gpu: &GpuContext, format: wgpu::TextureFormat, hdr: &wgpu::TextureView) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::include_wgsl!("bloom.wgsl"));
        let pipeline = |upsample: bool, blend| {
            let constants = [("UPSAMPLE", if upsample { 1.0 } else { 0.0 })];
            gpu.render_pipeline(&shader, format, blend, wgpu::PrimitiveTopology::TriangleList, &constants)
        };
        let additive = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let mut bloom = Self {
            down: pipeline(false, wgpu::BlendState::REPLACE),
            up: pipeline(true, wgpu::BlendState { color: additive, alpha: additive }),
            sampler,
            levels: Vec::new(),
        };
        bloom.resize(gpu, format, hdr);
        bloom
    }

    pub fn resize(&mut self, gpu: &GpuContext, format: wgpu::TextureFormat, hdr: &wgpu::TextureView) {
        let views: Vec<wgpu::TextureView> = (1..=BLOOM_LEVELS)
            .map(|k| {
                let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d {
                        width: (gpu.config.width >> k).max(1),
                        height: (gpu.config.height >> k).max(1),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                texture.create_view(&Default::default())
            })
            .collect();

        let source = |pipeline: &wgpu::RenderPipeline, view: &wgpu::TextureView| {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            })
        };

        self.levels = (0..views.len())
            .map(|k| Level {
                down_source: source(&self.down, if k == 0 { hdr } else { &views[k - 1] }),
                up_source: views.get(k + 1).map(|next| source(&self.up, next)),
                view: views[k].clone(),
            })
            .collect();
    }

    pub fn output(&self) -> &wgpu::TextureView {
        &self.levels[0].view
    }

    pub fn draw(&self, encoder: &mut wgpu::CommandEncoder) {
        for level in &self.levels {
            self.run(encoder, &self.down, &level.down_source, &level.view, wgpu::LoadOp::Clear(wgpu::Color::BLACK));
        }
        for level in self.levels.iter().rev() {
            if let Some(source) = &level.up_source {
                self.run(encoder, &self.up, source, &level.view, wgpu::LoadOp::Load);
            }
        }
    }

    fn run(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::RenderPipeline,
        source: &wgpu::BindGroup,
        target: &wgpu::TextureView,
        load: wgpu::LoadOp<wgpu::Color>,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, source, &[]);
        pass.draw(0..3, 0..1);
    }
}
