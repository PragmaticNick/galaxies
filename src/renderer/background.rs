// Twinkling starfield behind the galaxy, ported from the video's manim
// background (media/background/background.py): flat dark-blue fill plus
// tiny screen-space stars pulsing in brightness and size.

use std::f32::consts::PI;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use wgpu::util::DeviceExt as _;

use crate::settings::{
    BG_COLOR, BG_SEED, BG_STAR_COLORS, BG_STAR_COUNT, BG_STAR_OPACITY_RANGE, BG_STAR_PERIOD_RANGE,
    BG_STAR_RADIUS_RANGE, BG_TWINKLE_MIN_OPACITY, BG_TWINKLE_MIN_RADIUS,
};

// manim frame is 14.22 units wide; radius is stored as a fraction of screen width
const MANIM_FRAME_WIDTH: f32 = 8.0 * 16.0 / 9.0;

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
    start: Instant,
    clear_color: wgpu::Color,
}

impl Background {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let srgb = format.is_srgb();
        let stars = generate_stars(srgb);

        let star_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Background Star Buffer"),
            contents: bytemuck::cast_slice(&stars),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Background Params Buffer"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = init_pipeline(device, format);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Background Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: star_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let [r, g, b] = hex_to_rgb(BG_COLOR, srgb);
        Self {
            pipeline,
            bind_group,
            params_buffer,
            start: Instant::now(),
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

    pub fn prepare(&self, queue: &wgpu::Queue, width: u32, height: u32) {
        let time = self.start.elapsed().as_secs_f32();
        queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::cast_slice(&[
                time,
                BG_TWINKLE_MIN_OPACITY,
                width as f32,
                height as f32,
                BG_TWINKLE_MIN_RADIUS,
                0.0,
                0.0,
                0.0,
            ]),
        );
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass) {
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
                radius: rng.random_range(BG_STAR_RADIUS_RANGE.0..BG_STAR_RADIUS_RANGE.1)
                    / MANIM_FRAME_WIDTH,
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

fn init_pipeline(device: &wgpu::Device, format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("background.wgsl"));
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Background Pipeline"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}
