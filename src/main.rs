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
use crate::simulation::{CpuDirect, GpuDirect, ParticleMesh, Simulation, Strategy};

mod galaxy;
mod gpu;
mod renderer;
mod simulation;
mod star;

const STRATEGY: Strategy = Strategy::ParticleMesh { size: 64, h: 10.0 };

const GALAXY: GalaxyConfig = GalaxyConfig {
    center: [0.0, 0.0],
    radius: 200.0,
    star_count: 100000,
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
            Strategy::GpuDirect => Box::new(GpuDirect::new(&gpu, &star_buffer, stars.len())),
            Strategy::CpuDirect => Box::new(CpuDirect::new(stars)),
            Strategy::ParticleMesh { size, h } => {
                // grid centered on the galaxy
                let half = size as f32 * h / 2.0;
                let x0 = GALAXY.center[0] - half;
                let y0 = GALAXY.center[1] - half;
                Box::new(ParticleMesh::new(stars, size, h, x0, y0))
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

#[allow(dead_code)]
fn compare_pm_with_direct() {
    let galaxy = GalaxyConfig {
        center: [0.0, 0.0],
        radius: 100.0,
        star_count: 200,
        core_radius: 2.0,
        core_mass: 50000.0,
        star_mass: 10.0,
        star_radius: 1.5,
        gap: 10.0,
    };
    let dt = 0.016;
    let steps = 20;

    let stars = generate_galaxy(&galaxy);

    let mut direct = CpuDirect::new(stars.clone());
    for _ in 0..steps {
        direct.update(dt);
    }

    let half = galaxy.radius * 1.2;
    let x0 = galaxy.center[0] - half;
    let y0 = galaxy.center[1] - half;

    for h in [20.0, 10.0, 5.0, 2.5, 1.0] {
        let size = (2.0 * half / h).ceil() as i32;

        let mut pm = ParticleMesh::new(stars.clone(), size, h, x0, y0);
        for _ in 0..steps {
            pm.update(dt);
        }

        let mut total = 0.0;
        let mut max: f32 = 0.0;
        for (a, b) in pm.stars().iter().zip(direct.stars()) {
            let dx = a.pos[0] - b.pos[0];
            let dy = a.pos[1] - b.pos[1];
            let d = (dx * dx + dy * dy).sqrt();
            total += d;
            max = max.max(d);
        }
        let mean = total / stars.len() as f32;

        println!("h = {h:5.2}, grid = {size:3}x{size:3}: total = {total:10.4}, mean = {mean:8.4}, max = {max:8.4}");
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    EventLoop::new()?.run_app(&mut App::default())?;

    //compare_pm_with_direct();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs the app's galaxy with the app's PM grid against CpuDirect.
    /// Slow (direct is O(N²) on 50k stars), run with:
    /// cargo test --release pm_error_vs_direct -- --ignored --nocapture
    #[test]
    #[ignore]
    fn pm_error_vs_direct() {
        let Strategy::ParticleMesh { size, h } = STRATEGY else {
            panic!("STRATEGY is not ParticleMesh");
        };
        let dt = 0.016;
        let steps = 10;

        let stars = generate_galaxy(&GALAXY);

        let half = size as f32 * h / 2.0;
        let mut pm = ParticleMesh::new(
            stars.clone(),
            size,
            h,
            GALAXY.center[0] - half,
            GALAXY.center[1] - half,
        );
        let mut direct = CpuDirect::new(stars.clone());
        for _ in 0..steps {
            pm.update(dt);
            direct.update(dt);
        }

        let distance = |a: [f32; 2], b: [f32; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();

        let mut error = 0.0;
        let mut max_error: f32 = 0.0;
        let mut travel = 0.0;
        for ((p, d), s) in pm.stars().iter().zip(direct.stars()).zip(&stars) {
            let e = distance(p.pos, d.pos);
            error += e;
            max_error = max_error.max(e);
            travel += distance(d.pos, s.pos);
        }
        let n = stars.len() as f32;
        let mean_error = error / n;
        let mean_travel = travel / n;
        let relative = mean_error / mean_travel;

        println!("grid {size}x{size}, h = {h}, {steps} steps");
        println!("mean error = {mean_error:.4}, max error = {max_error:.4}");
        println!("mean travel = {mean_travel:.4}, relative error = {:.3}%", relative * 100.0);

        // measured ~0.3% for 64x64, h = 10
        assert!(relative < 0.01, "relative error {:.3}% is above 1%", relative * 100.0);
    }
}
