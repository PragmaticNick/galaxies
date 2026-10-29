// Twinkling starfield behind the galaxy: flat dark-blue fill plus tiny
// screen-space stars pulsing in brightness and size.

use std::f32::consts::PI;

use bytemuck::{Pod, Zeroable};
use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::config::{
    BG_COLOR, BG_SEED, BG_STAR_COLORS, BG_STAR_COUNT, BG_STAR_OPACITY_RANGE, BG_STAR_PERIOD_RANGE,
    BG_STAR_RADIUS_RANGE, BG_TWINKLE_MIN_OPACITY, BG_TWINKLE_MIN_RADIUS,
};
use crate::gpu::GpuContext;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BgStar {
    pos: [f32; 2],
    radius: f32,
    opacity: f32,
    color: [f32; 3],
    period: f32,
    phase: f32,
    _pad: [f32; 3],
}

pub struct Background {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,
    clear_color: wgpu::Color,
}

impl Background {
    pub fn new(gpu: &GpuContext) -> Self {
        let srgb = gpu.config.format.is_srgb();
        let stars = gpu.storage_buffer(&generate_stars(srgb));
        let params_buffer = gpu.uniform_buffer(&[0.0f32; 8]);

        let shader = gpu
            .device
            .create_shader_module(wgpu::include_wgsl!("background.wgsl"));
        let pipeline = gpu.render_pipeline(
            &shader,
            gpu.config.format,
            wgpu::BlendState::ALPHA_BLENDING,
            &[],
        );
        let bind_group = gpu.bind_group(&pipeline.get_bind_group_layout(0), &[&stars, &params_buffer]);

        let [r, g, b] = hex_to_rgb(BG_COLOR, srgb);
        Self {
            pipeline,
            bind_group,
            params_buffer,
            clear_color: wgpu::Color {
                r: r as f64,
                g: g as f64,
                b: b as f64,
                a: 1.0,
            },
        }
    }

    pub fn clear_color(&self) -> wgpu::Color {
        self.clear_color
    }

    // `time` in seconds drives the twinkle
    pub fn draw(&self, gpu: &GpuContext, pass: &mut wgpu::RenderPass, time: f32) {
        gpu.write(
            &self.params_buffer,
            &[
                time,
                BG_TWINKLE_MIN_OPACITY,
                gpu.config.width as f32,
                gpu.config.height as f32,
                BG_TWINKLE_MIN_RADIUS,
                0.0,
                0.0,
                0.0,
            ],
        );
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..BG_STAR_COUNT * 6, 0..1);
    }
}

fn generate_stars(srgb: bool) -> Vec<BgStar> {
    let mut rng = StdRng::seed_from_u64(BG_SEED);
    (0..BG_STAR_COUNT)
        .map(|_| {
            let (hot, cool) = BG_STAR_COLORS[rng.random_range(0..BG_STAR_COLORS.len())];
            let hot = hex_to_rgb(hot, false);
            let cool = hex_to_rgb(cool, false);
            let t: f32 = rng.random();
            let color = [0, 1, 2].map(|c| {
                let v = hot[c] + (cool[c] - hot[c]) * t;
                if srgb { srgb_to_linear(v) } else { v }
            });

            BgStar {
                pos: [rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)],
                radius: rng.random_range(BG_STAR_RADIUS_RANGE.0..BG_STAR_RADIUS_RANGE.1),
                opacity: rng.random_range(BG_STAR_OPACITY_RANGE.0..BG_STAR_OPACITY_RANGE.1),
                color,
                period: rng.random_range(BG_STAR_PERIOD_RANGE.0..BG_STAR_PERIOD_RANGE.1),
                phase: rng.random_range(0.0..2.0 * PI),
                _pad: [0.0; 3],
            }
        })
        .collect()
}

fn hex_to_rgb(hex: u32, linear: bool) -> [f32; 3] {
    [16, 8, 0].map(|shift| {
        let v = ((hex >> shift) & 0xFF) as f32 / 255.0;
        if linear { srgb_to_linear(v) } else { v }
    })
}

// sRGB surface blends in linear space, so authored hex colors need decoding
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}
