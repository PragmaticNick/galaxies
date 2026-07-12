use std::mem;
use std::sync::Arc;

use glyphon::{
    Attrs, Buffer, Cache, Color, Family, FontSystem, Metrics, Resolution, Shaping, SwashCache,
    TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};
use wgpu::util::DeviceExt as _;
use winit::window::Window;

use crate::renderer::camera::CameraUniform;
use crate::renderer::fps::FpsCounter;
use crate::star::Star;

mod camera;
mod fps;

const COMPUTE_PASSES: [&str; 3] = ["drift", "kick", "commit"];

const WORKGROUP_SIZE: usize = 256;

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    window: Arc<Window>,

    render_pipeline: wgpu::RenderPipeline,
    drift_pipeline: wgpu::ComputePipeline,
    kick_pipeline: wgpu::ComputePipeline,
    commit_pipeline: wgpu::ComputePipeline,

    star_render_bind_group: wgpu::BindGroup,
    star_compute_bind_group: wgpu::BindGroup,
    compute_bind_group_layout: wgpu::BindGroupLayout,
    sim_buffer: wgpu::Buffer,
    eps: f32,
    g: f32,

    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    font_system: FontSystem,
    swash_cache: SwashCache,
    text_viewport: Viewport,
    text_atlas: TextAtlas,
    text_renderer: TextRenderer,
    fps_buffer: Buffer,
    fps_counter: FpsCounter,
    last_fps: f32,
    preset_name: String,
    force_text_update: bool,

    timestamps: wgpu::QuerySet,
    timestamp_resolve: wgpu::Buffer,
    timestamp_read: wgpu::Buffer,
    timestamp_period: f32,
    pass_ms: [f32; 3],
}

impl Renderer {
    pub async fn new(
        window: Arc<Window>,
        stars: &[Star],
        eps: f32,
        g: f32,
        preset_name: &str,
    ) -> anyhow::Result<Self> {
        let (surface, device, queue, config) = init_wgpu(window.clone()).await?;
        let (camera_uniform, camera_buffer) = init_camera(&device, config.width, config.height);
        let star_buffer = init_star_buffer(&device, &queue, stars);

        let sim_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sim Params Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let render_pipeline = init_pipeline(&device, config.format);
        let (compute_bind_group_layout, drift_pipeline, kick_pipeline, commit_pipeline) =
            init_compute_pipelines(&device);

        let camera_bind_group = buffer_bind_group(
            &device,
            "Camera Bind Group",
            &render_pipeline.get_bind_group_layout(0),
            &camera_buffer,
        );
        let star_render_bind_group = buffer_bind_group(
            &device,
            "Star Render Bind Group",
            &render_pipeline.get_bind_group_layout(1),
            &star_buffer,
        );
        let star_compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Star Compute Bind Group"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: star_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sim_buffer.as_entire_binding(),
                },
            ],
        });

        let (font_system, swash_cache, text_viewport, text_atlas, text_renderer, fps_buffer) =
            init_text(&device, &queue, config.format);
        let timestamps = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Compute Timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 6,
        });
        let timestamp_resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Timestamp Resolve"),
            size: 48,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let timestamp_read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Timestamp Read"),
            size: 48,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let timestamp_period = queue.get_timestamp_period();

        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
            is_surface_configured: false,
            drift_pipeline,
            kick_pipeline,
            commit_pipeline,
            render_pipeline,
            star_render_bind_group,
            star_compute_bind_group,
            compute_bind_group_layout,
            sim_buffer,
            eps,
            g,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            font_system,
            swash_cache,
            text_viewport,
            text_atlas,
            text_renderer,
            fps_buffer,
            fps_counter: FpsCounter::new(),
            last_fps: 0.0,
            preset_name: preset_name.to_string(),
            force_text_update: true,
            timestamp_period,
            timestamps,
            timestamp_resolve,
            timestamp_read,
            pass_ms: [0.0; 3],
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;
            self.camera_uniform.resize(width, height);
            self.queue.write_buffer(
                &self.camera_buffer,
                0,
                bytemuck::cast_slice(&[self.camera_uniform]),
            );
        }
    }

    pub fn request_redraw(&self) {
        self.window.request_redraw();
    }

    pub fn set_stars(&mut self, stars: &[Star]) {
        let star_buffer = init_star_buffer(&self.device, &self.queue, stars);
        self.star_render_bind_group = buffer_bind_group(
            &self.device,
            "Star Render Bind Group",
            &self.render_pipeline.get_bind_group_layout(1),
            &star_buffer,
        );
        self.star_compute_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Star Compute Bind Group"),
            layout: &self.compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: star_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.sim_buffer.as_entire_binding(),
                },
            ],
        });
    }

    pub fn set_physics(&mut self, eps: f32, g: f32) {
        self.eps = eps;
        self.g = g;
    }

    pub fn set_preset_name(&mut self, name: &str) {
        self.preset_name = name.to_string();
        self.force_text_update = true;
    }

    pub fn render(&mut self, star_count: usize, dt: f32) -> anyhow::Result<()> {
        if !self.is_surface_configured {
            return Ok(());
        }

        self.tick_fps(star_count);
        self.prepare_text();

        self.queue.write_buffer(
            &self.sim_buffer,
            0,
            bytemuck::cast_slice(&[dt, self.eps, self.g, 0.0]),
        );

        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => return Ok(()),
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Frame Encoder"),
            });
        {
            let workgroup_count = star_count.div_ceil(WORKGROUP_SIZE) as u32;
            let dispatches = [
                &self.drift_pipeline,
                &self.kick_pipeline,
                &self.commit_pipeline,
            ];
            for (idx, pipeline) in dispatches.into_iter().enumerate() {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Pass"),
                    timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                        query_set: &self.timestamps,
                        beginning_of_pass_write_index: Some((idx * 2) as u32),
                        end_of_pass_write_index: Some((idx * 2 + 1) as u32),
                    }),
                });
                pass.set_bind_group(0, &self.star_compute_bind_group, &[]);
                pass.set_pipeline(pipeline);
                pass.dispatch_workgroups(workgroup_count, 1, 1);
            }
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });

            pass.set_pipeline(&self.render_pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_bind_group(1, &self.star_render_bind_group, &[]);
            pass.draw(0..(star_count * 6) as u32, 0..1);

            self.text_renderer
                .render(&self.text_atlas, &self.text_viewport, &mut pass)
                .unwrap();
        }

        encoder.resolve_query_set(&self.timestamps, 0..6, &self.timestamp_resolve, 0);
        encoder.copy_buffer_to_buffer(&self.timestamp_resolve, 0, &self.timestamp_read, 0, 48);

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        self.timestamp_read
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        {
            let view = self.timestamp_read.slice(..).get_mapped_range();
            let ticks: &[u64] = bytemuck::cast_slice(&view);
            for k in 0..3 {
                let dt = ticks[k * 2 + 1].saturating_sub(ticks[k * 2]);
                self.pass_ms[k] = dt as f32 * self.timestamp_period / 1_000_000.0;
            }
        }
        self.timestamp_read.unmap();

        Ok(())
    }

    fn tick_fps(&mut self, star_count: usize) {
        let fps_changed = match self.fps_counter.tick() {
            Some(fps) => {
                self.last_fps = fps;
                true
            }
            None => false,
        };

        if fps_changed || self.force_text_update {
            self.force_text_update = false;
            let mut text = format!(
                "FPS: {:.0}\nStars: {}\nStrategy: gpu\nPreset: {}",
                self.last_fps, star_count, self.preset_name
            );
            for (label, dt) in COMPUTE_PASSES.iter().zip(self.pass_ms) {
                text.push_str(&format!("\n{label}: {dt:.3} ms"));
            }
            self.fps_buffer.set_text(
                &mut self.font_system,
                &text,
                &Attrs::new().family(Family::Monospace),
                Shaping::Basic,
                None,
            );
            self.fps_buffer
                .shape_until_scroll(&mut self.font_system, false);
        }
    }
    fn prepare_text(&mut self) {
        self.text_viewport.update(
            &self.queue,
            Resolution {
                width: self.config.width,
                height: self.config.height,
            },
        );
        self.text_renderer
            .prepare(
                &self.device,
                &self.queue,
                &mut self.font_system,
                &mut self.text_atlas,
                &self.text_viewport,
                [TextArea {
                    buffer: &self.fps_buffer,
                    left: 10.0,
                    top: 10.0,
                    scale: 1.0,
                    bounds: TextBounds {
                        left: 0,
                        top: 0,
                        right: 600,
                        bottom: 400,
                    },
                    default_color: Color::rgb(255, 255, 255),
                    custom_glyphs: &[],
                }],
                &mut self.swash_cache,
            )
            .unwrap();
    }
}

async fn init_wgpu(
    window: Arc<Window>,
) -> anyhow::Result<(
    wgpu::Surface<'static>,
    wgpu::Device,
    wgpu::Queue,
    wgpu::SurfaceConfiguration,
)> {
    let size = window.inner_size();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        flags: Default::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: Default::default(),
    });
    let surface = instance.create_surface(window).unwrap();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await?;

    let required_features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            required_limits: wgpu::Limits::default(),
            memory_hints: Default::default(),
            trace: wgpu::Trace::Off,
        })
        .await?;
    let surface_caps = surface.get_capabilities(&adapter);
    let surface_format = surface_caps
        .formats
        .iter()
        .find(|f| f.is_srgb())
        .copied()
        .unwrap_or(surface_caps.formats[0]);
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: size.width,
        height: size.height,
        present_mode: surface_caps.present_modes[0],
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    Ok((surface, device, queue, config))
}

fn init_camera(device: &wgpu::Device, width: u32, height: u32) -> (CameraUniform, wgpu::Buffer) {
    let camera_uniform = CameraUniform::new(width, height);
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Camera Buffer"),
        contents: bytemuck::cast_slice(&[camera_uniform]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    (camera_uniform, buffer)
}

fn init_star_buffer(device: &wgpu::Device, queue: &wgpu::Queue, stars: &[Star]) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Star Buffer"),
        size: (stars.len() * mem::size_of::<Star>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    queue.write_buffer(&buffer, 0, bytemuck::cast_slice(stars));

    buffer
}
fn buffer_bind_group(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    })
}

fn init_pipeline(device: &wgpu::Device, format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
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
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn init_compute_pipelines(
    device: &wgpu::Device,
) -> (
    wgpu::BindGroupLayout,
    wgpu::ComputePipeline,
    wgpu::ComputePipeline,
    wgpu::ComputePipeline,
) {
    let shader = device.create_shader_module(wgpu::include_wgsl!("compute.wgsl"));

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Compute Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Compute Pipeline Layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let make = |entry: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: Default::default(),
        })
    };

    let drift_pipeline = make("drift_half");
    let kick_pipeline = make("kick");
    let commit_pipeline = make("commit");
    (
        bind_group_layout,
        drift_pipeline,
        kick_pipeline,
        commit_pipeline,
    )
}

fn init_text(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) -> (
    FontSystem,
    SwashCache,
    Viewport,
    TextAtlas,
    TextRenderer,
    Buffer,
) {
    let mut font_system = FontSystem::new();
    let swash_cache = SwashCache::new();
    let cache = Cache::new(device);
    let viewport = Viewport::new(device, &cache);
    let mut atlas = TextAtlas::new(device, queue, &cache, format);
    let renderer = TextRenderer::new(&mut atlas, device, wgpu::MultisampleState::default(), None);
    let mut fps_buffer = Buffer::new(&mut font_system, Metrics::new(32.0, 40.0));
    fps_buffer.set_size(&mut font_system, Some(600.0), Some(400.0));
    fps_buffer.set_text(
        &mut font_system,
        "FPS: --",
        &Attrs::new().family(Family::Monospace),
        Shaping::Basic,
        None,
    );
    fps_buffer.shape_until_scroll(&mut font_system, false);
    (
        font_system,
        swash_cache,
        viewport,
        atlas,
        renderer,
        fps_buffer,
    )
}
