use std::sync::Arc;
use std::time::Instant;

use winit::{
    application::ApplicationHandler,
    event::{KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Fullscreen, Window, WindowId},
};

use crate::galaxy::{GalaxyConfig, generate_galaxy};
use crate::gpu::GpuContext;
use crate::renderer::Renderer;
use crate::simulation::{CpuDirect, GpuDirect, Simulation, Strategy};

mod galaxy;
mod gpu;
mod renderer;
mod simulation;
mod star;

const STRATEGY: Strategy = Strategy::GpuDirect;

const GALAXY: GalaxyConfig = GalaxyConfig {
    center: [0.0, 0.0],
    radius: 200.0,
    star_count: 50000,
    core_radius: 2.0,
    core_mass: 500000.0,
    star_mass: 10.0,
    star_radius: 1.5,
    gap: 20.0,
};

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
            GALAXY.radius,
            GALAXY.center,
            STRATEGY.name(),
        );
        let simulation: Box<dyn Simulation> = match STRATEGY {
            Strategy::GpuDirect => Box::new(GpuDirect::init(&gpu, &star_buffer, stars)),
            Strategy::CpuDirect => Box::new(CpuDirect::init(&gpu, &star_buffer, stars)),
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
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = self
                    .last_frame
                    .map_or(0.0, |t| (now - t).as_secs_f32().min(0.016));
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
