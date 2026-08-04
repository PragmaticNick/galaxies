use std::io::Write as _;
use std::path::Path;
use std::process::{Command, Stdio};

use glyphon::{
    Attrs, Buffer, Color, Family, FontSystem, Resolution, Shaping, SwashCache, TextArea,
    TextAtlas, TextBounds, TextRenderer, Viewport,
};

use crate::physics::{PhysicsStrategy, update_physics};
use crate::renderer::geometry::Vertex;
use crate::renderer::{init_camera, init_pipeline, init_text, stars_to_vertices};
use crate::star::Star;

pub struct ExportConfig {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub seconds: f32,
    pub output: String,
    pub strategy: PhysicsStrategy,
}

const TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

// Displayed FPS is faked to look like a live capture: it hovers around
// FAKE_FPS_BASE and occasionally dips/peaks toward the min/max instead of
// staying pinned at the export's fixed frame rate.
const FAKE_FPS_BASE: f32 = 44.0;
const FAKE_FPS_MIN: f32 = 40.0;
const FAKE_FPS_MAX: f32 = 45.0;

fn gaussian(rng: &mut impl rand::RngExt) -> f32 {
    let u1: f32 = rng.random_range(1e-6f32..1.0);
    let u2: f32 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

pub fn export_video(cfg: &ExportConfig, mut stars: Vec<Star>) -> anyhow::Result<()> {
    pollster::block_on(run(cfg, &mut stars))
}

async fn run(cfg: &ExportConfig, stars: &mut [Star]) -> anyhow::Result<()> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        flags: Default::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: Default::default(),
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            required_limits: wgpu::Limits::default(),
            memory_hints: Default::default(),
            trace: wgpu::Trace::Off,
        })
        .await?;

    let (_, _camera_buffer, camera_layout, camera_bind_group) =
        init_camera(&device, cfg.width, cfg.height);
    let render_pipeline = init_pipeline(&device, TEXTURE_FORMAT, &camera_layout);

    let (mut font_system, mut swash_cache, mut text_viewport, mut text_atlas, mut text_renderer, mut fps_buffer) =
        init_text(&device, &queue, TEXTURE_FORMAT);
    text_viewport.update(
        &queue,
        Resolution {
            width: cfg.width,
            height: cfg.height,
        },
    );

    let mut rng = rand::rng();
    let star_count = stars.len();
    let strategy_name = cfg.strategy.name();
    update_hud_text(
        &device,
        &queue,
        &mut font_system,
        &mut swash_cache,
        &mut text_atlas,
        &text_viewport,
        &mut text_renderer,
        &mut fps_buffer,
        &format!(
            "FPS: {:.0}\nStars: {}\nStrategy: {}",
            FAKE_FPS_BASE, star_count, strategy_name
        ),
    );

    let vertex_count = stars.len() * 6;
    let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Export Vertex Buffer"),
        size: (vertex_count * std::mem::size_of::<Vertex>()) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let color_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Export Color Texture"),
        size: wgpu::Extent3d {
            width: cfg.width,
            height: cfg.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TEXTURE_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let unpadded_bytes_per_row = cfg.width * 4;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Export Readback Buffer"),
        size: (padded_bytes_per_row * cfg.height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut ffmpeg = spawn_ffmpeg(&cfg.output, cfg.width, cfg.height, cfg.fps)?;
    let mut writer = ffmpeg
        .stdin
        .take()
        .expect("ffmpeg stdin was piped at spawn");

    let frame_count = (cfg.seconds * cfg.fps as f32).round() as u32;
    let dt = 1.0 / cfg.fps as f32;
    let mut frame_bytes = vec![0u8; (unpadded_bytes_per_row * cfg.height) as usize];

    for frame in 0..frame_count {
        update_physics(stars, dt, cfg.strategy);

        if frame > 0 && frame % cfg.fps.max(1) == 0 {
            let fake_fps =
                (FAKE_FPS_BASE + gaussian(&mut rng) * 1.5).clamp(FAKE_FPS_MIN, FAKE_FPS_MAX);
            update_hud_text(
                &device,
                &queue,
                &mut font_system,
                &mut swash_cache,
                &mut text_atlas,
                &text_viewport,
                &mut text_renderer,
                &mut fps_buffer,
                &format!(
                    "FPS: {:.0}\nStars: {}\nStrategy: {}",
                    fake_fps, star_count, strategy_name
                ),
            );
        }

        let vertices = stars_to_vertices(stars);
        queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&vertices));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Export Frame Encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Export Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&render_pipeline);
            pass.set_bind_group(0, &camera_bind_group, &[]);
            pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            pass.draw(0..vertices.len() as u32, 0..1);

            text_renderer
                .render(&text_atlas, &text_viewport, &mut pass)
                .unwrap();
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &color_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(cfg.height),
                },
            },
            wgpu::Extent3d {
                width: cfg.width,
                height: cfg.height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));

        let slice = readback_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv()??;

        {
            let padded = slice.get_mapped_range();
            for row in 0..cfg.height as usize {
                let src_start = row * padded_bytes_per_row as usize;
                let src = &padded[src_start..src_start + unpadded_bytes_per_row as usize];
                let dst_start = row * unpadded_bytes_per_row as usize;
                frame_bytes[dst_start..dst_start + unpadded_bytes_per_row as usize]
                    .copy_from_slice(src);
            }
        }
        readback_buffer.unmap();

        writer.write_all(&frame_bytes)?;

        if frame % cfg.fps.max(1) == 0 {
            log::info!("exported frame {frame}/{frame_count}");
        }
    }

    drop(writer);
    let status = ffmpeg.wait()?;
    anyhow::ensure!(status.success(), "ffmpeg exited with {status}");
    log::info!("wrote {}", cfg.output);

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn update_hud_text(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    font_system: &mut FontSystem,
    swash_cache: &mut SwashCache,
    text_atlas: &mut TextAtlas,
    text_viewport: &Viewport,
    text_renderer: &mut TextRenderer,
    fps_buffer: &mut Buffer,
    text: &str,
) {
    fps_buffer.set_text(
        font_system,
        text,
        &Attrs::new().family(Family::Monospace),
        Shaping::Basic,
        None,
    );
    fps_buffer.shape_until_scroll(font_system, false);
    text_renderer
        .prepare(
            device,
            queue,
            font_system,
            text_atlas,
            text_viewport,
            [TextArea {
                buffer: fps_buffer,
                left: 10.0,
                top: 10.0,
                scale: 1.0,
                bounds: TextBounds {
                    left: 0,
                    top: 0,
                    right: 600,
                    bottom: 144,
                },
                default_color: Color::rgb(255, 255, 255),
                custom_glyphs: &[],
            }],
            swash_cache,
        )
        .unwrap();
}

fn spawn_ffmpeg(
    output: &str,
    width: u32,
    height: u32,
    fps: u32,
) -> anyhow::Result<std::process::Child> {
    if Path::new(output).exists() {
        std::fs::remove_file(output)?;
    }
    Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            &format!("{width}x{height}"),
            "-r",
            &fps.to_string(),
            "-i",
            "-",
            "-c:v",
            "prores_ks",
            "-profile:v",
            "4444",
            "-pix_fmt",
            "yuva444p10le",
            output,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| anyhow::anyhow!("failed to spawn ffmpeg (is it installed and on PATH?): {e}"))
}
