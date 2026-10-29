use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use winit::{
    application::ApplicationHandler,
    event::{KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Fullscreen, Window, WindowId},
};

use crate::config::{
    GALAXIES, GALAXY_SEED, PAN_SPEED, PM_CELL_SIZE, PM_GRID_SIZE, TIME_SCALE, ZOOM_SPEED, ZOOM_STEP,
};
use crate::galaxy::generate_galaxy;
use crate::gpu::GpuContext;
use crate::particle_mesh::ParticleMesh;
use crate::renderer::Renderer;

mod config;
mod galaxy;
mod gpu;
mod particle_mesh;
mod renderer;
mod star;

struct Engine {
    gpu: GpuContext,
    renderer: Renderer,
    simulation: ParticleMesh,
    start: Instant,
}

impl Engine {
    fn new(gpu: GpuContext) -> Self {
        let stars: Vec<_> = GALAXIES
            .iter()
            .enumerate()
            .flat_map(|(i, galaxy)| generate_galaxy(galaxy, GALAXY_SEED + i as u64))
            .collect();

        let star_buffer = gpu.storage_buffer(&stars);
        let renderer = Renderer::new(&gpu, &star_buffer, stars.len());
        let simulation = ParticleMesh::new(&gpu, &star_buffer, &stars, PM_GRID_SIZE, PM_CELL_SIZE);

        Self {
            gpu,
            renderer,
            simulation,
            start: Instant::now(),
        }
    }

    fn frame(&mut self, dt: f32) {
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        self.simulation.step(&self.gpu, &mut encoder, dt);

        let output = self.gpu.acquire();
        if let Some(output) = &output {
            let view = output.texture.create_view(&Default::default());
            let time = self.start.elapsed().as_secs_f32();
            self.renderer.draw(&self.gpu, &mut encoder, &view, time);
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
    held: HashSet<KeyCode>,
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
        let gpu = pollster::block_on(GpuContext::new(window)).unwrap();
        self.engine = Some(Engine::new(gpu));
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
            WindowEvent::KeyboardInput {
                event: KeyEvent { physical_key: PhysicalKey::Code(code), state, .. },
                ..
            } => {
                if state.is_pressed() {
                    self.held.insert(code);
                } else {
                    self.held.remove(&code);
                }
            }
            WindowEvent::Focused(false) => self.held.clear(),
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
                let real_dt = self.last_frame.map_or(0.0, |t| (now - t).as_secs_f32().min(0.1));
                self.last_frame = Some(now);
                move_camera(engine, &self.held, real_dt);
                engine.frame(real_dt.min(0.016) * TIME_SCALE);
            }
            _ => {}
        }
    }
}

fn move_camera(engine: &mut Engine, held: &HashSet<KeyCode>, dt: f32) {
    let axis = |pos, neg| held.contains(&pos) as i32 as f32 - held.contains(&neg) as i32 as f32;
    let (x, y) = (axis(KeyCode::KeyD, KeyCode::KeyA), axis(KeyCode::KeyW, KeyCode::KeyS));
    let zoom = axis(KeyCode::ArrowUp, KeyCode::ArrowDown);

    let gpu = &engine.gpu;
    if x != 0.0 || y != 0.0 {
        // pan() follows drag semantics (content moves with the cursor), so invert x.
        let px = PAN_SPEED * gpu.config.height as f32 * dt;
        engine.renderer.pan(gpu, -x * px, y * px);
    }
    if zoom != 0.0 {
        let center = [gpu.config.width as f32 / 2.0, gpu.config.height as f32 / 2.0];
        engine.renderer.zoom(gpu, ZOOM_SPEED.powf(-zoom * dt), center);
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    EventLoop::new()?.run_app(&mut App::default())?;

    Ok(())
}
