use crate::config::BLOOM_LEVELS;
use crate::gpu::{ADDITIVE, GpuContext, render_pass};

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
            gpu.render_pipeline(&shader, format, blend, &constants)
        };
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let mut bloom = Self {
            down: pipeline(false, wgpu::BlendState::REPLACE),
            up: pipeline(true, ADDITIVE),
            sampler,
            levels: Vec::new(),
        };
        bloom.resize(gpu, format, hdr);
        bloom
    }

    pub fn resize(&mut self, gpu: &GpuContext, format: wgpu::TextureFormat, hdr: &wgpu::TextureView) {
        // each level is half the size of the one before, starting from half the screen
        let views: Vec<wgpu::TextureView> = (1..=BLOOM_LEVELS)
            .map(|k| gpu.render_target((gpu.config.width >> k).max(1), (gpu.config.height >> k).max(1), format))
            .collect();

        let source = |pipeline: &wgpu::RenderPipeline, view: &wgpu::TextureView| {
            gpu.texture_bind_group(&pipeline.get_bind_group_layout(0), &[view], &self.sampler)
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
        // downsample the screen level by level, then add each level back into the one above it
        for level in &self.levels {
            run(encoder, &self.down, &level.down_source, &level.view, wgpu::LoadOp::Clear(wgpu::Color::BLACK));
        }
        for level in self.levels.iter().rev() {
            if let Some(source) = &level.up_source {
                run(encoder, &self.up, source, &level.view, wgpu::LoadOp::Load);
            }
        }
    }
}

fn run(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::RenderPipeline,
    source: &wgpu::BindGroup,
    target: &wgpu::TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
) {
    let mut pass = render_pass(encoder, target, load);
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, source, &[]);
    pass.draw(0..3, 0..1);
}
