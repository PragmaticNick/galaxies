use std::sync::Arc;
use std::time::Instant;

use winit::{
    application::ApplicationHandler,
    event::{KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Fullscreen, Window, WindowId},
};

use crate::config::{GALAXY, STRATEGY, TIME_SCALE, VIEW_RADIUS, ZOOM_STEP};
use crate::galaxy::generate_galaxy;
use crate::gpu::GpuContext;
use crate::renderer::Renderer;
use crate::simulation::{
    CpuDirect, GpuDirect, GpuParticleMesh, ParticleMesh, Simulation, Strategy,
};

mod config;
mod galaxy;
mod gpu;
mod renderer;
mod simulation;
mod star;

struct Engine {
    gpu: GpuContext,
    star_buffer: wgpu::Buffer,
    renderer: Renderer,
    simulation: Box<dyn Simulation>,
}

impl Engine {
    async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let gpu = GpuContext::new(window).await?;
        let stars = generate_galaxy(&GALAXY);

        let star_buffer = gpu.storage_buffer(&stars);

        let renderer = Renderer::new(
            &gpu,
            &star_buffer,
            stars.len(),
            VIEW_RADIUS,
            STRATEGY.grid_half(),
            STRATEGY.name(),
        );
        let simulation: Box<dyn Simulation> = match STRATEGY {
            Strategy::GpuDirect => Box::new(GpuDirect::new(&gpu, &star_buffer, stars.len())),
            Strategy::CpuDirect => Box::new(CpuDirect::new(stars)),
            Strategy::ParticleMesh { size, h } => Box::new(ParticleMesh::new(stars, size, h)),
            Strategy::ParticleMeshFft { size, h } => Box::new(ParticleMesh::new_fft(stars, size, h)),
            Strategy::GpuParticleMeshFft { size, h } => {
                Box::new(GpuParticleMesh::new(&gpu, &star_buffer, &stars, size, h))
            }
        };

        Ok(Self {
            gpu,
            star_buffer,
            renderer,
            simulation,
        })
    }

    fn frame(&mut self, dt: f32) {
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        self.simulation
            .step(&self.gpu, &mut encoder, &self.star_buffer, dt);

        let output = self.gpu.acquire();
        if let Some(output) = &output {
            let view = output.texture.create_view(&Default::default());
            self.renderer.draw(&self.gpu, &mut encoder, &view);
        }

        self.gpu.queue.submit([encoder.finish()]);
        if let Some(output) = output {
            output.present();
        }
    }
}

#[derive(Default)]
struct App {
    engine: Option<Engine>,
    last_frame: Option<Instant>,
    cursor: [f32; 2],
    dragging: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::Poll);
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );
        window.set_fullscreen(Some(Fullscreen::Borderless(None)));
        self.engine = Some(pollster::block_on(Engine::new(window)).unwrap());
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(engine) = &self.engine {
            engine.gpu.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(engine) = &mut self.engine else {
            return;
        };

        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                ..
            } => event_loop.exit(),
            WindowEvent::Resized(size) => {
                engine.gpu.resize(size.width, size.height);
                engine.renderer.resize(&engine.gpu);
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                self.dragging = state.is_pressed();
            }
            WindowEvent::CursorMoved { position, .. } => {
                let cursor = [position.x as f32, position.y as f32];
                if self.dragging {
                    let (dx, dy) = (cursor[0] - self.cursor[0], cursor[1] - self.cursor[1]);
                    engine.renderer.pan(&engine.gpu, dx, dy);
                }
                self.cursor = cursor;
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 100.0,
                };
                engine.renderer.zoom(&engine.gpu, ZOOM_STEP.powf(lines), self.cursor);
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = self
                    .last_frame
                    .map_or(0.0, |t| (now - t).as_secs_f32().min(0.016))
                    * TIME_SCALE;
                self.last_frame = Some(now);
                engine.frame(dt);
            }
            _ => {}
        }
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    EventLoop::new()?.run_app(&mut App::default())?;

    Ok(())
}

