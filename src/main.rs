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
use crate::simulation::{
    CpuDirect, GpuDirect, GpuParticleMesh, ParticleMesh, Simulation, Strategy,
};

mod galaxy;
mod gpu;
mod renderer;
mod simulation;
mod star;

const STRATEGY: Strategy = Strategy::GpuParticleMeshFft { size: 127, h: 10.0 };
// const STRATEGY: Strategy = Strategy::ParticleMesh { size: 128, h: 10.0 };
// const STRATEGY: Strategy = Strategy::GpuDirect;

/// World units from the screen center to its top edge.
const VIEW_RADIUS: f32 = 800.0;

/// Simulated time per second of wall time.
const TIME_SCALE: f32 = 0.1;

const GALAXY: GalaxyConfig = GalaxyConfig {
    center: [0.0, 0.0],
    radius: 400.0,
    star_count: 50000,
    core_radius: 2.0,
    core_mass: 500000.0,
    star_mass: 10.0,
    star_radius: 1.5,
    gap: 20.0,
};

/// Lower-left corner of a PM grid centered on the galaxy.
fn grid_origin(size: i32, h: f32) -> (f32, f32) {
    let half = size as f32 * h / 2.0;
    (GALAXY.center[0] - half, GALAXY.center[1] - half)
}

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
            STRATEGY.name(),
        );
        let simulation: Box<dyn Simulation> = match STRATEGY {
            Strategy::GpuDirect => Box::new(GpuDirect::new(&gpu, &star_buffer, stars.len())),
            Strategy::CpuDirect => Box::new(CpuDirect::new(stars)),
            Strategy::ParticleMesh { size, h } => {
                let (x0, y0) = grid_origin(size, h);
                Box::new(ParticleMesh::new(stars, size, h, x0, y0))
            }
            Strategy::ParticleMeshFft { size, h } => {
                let (x0, y0) = grid_origin(size, h);
                Box::new(ParticleMesh::new_fft(stars, size, h, x0, y0))
            }
            Strategy::GpuParticleMeshFft { size, h } => {
                let (x0, y0) = grid_origin(size, h);
                Box::new(GpuParticleMesh::new(&gpu, &star_buffer, &stars, size, h, x0, y0))
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
                    .map_or(0.0, |t| (now - t).as_secs_f32().min(0.016))
                    * TIME_SCALE;
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
    use crate::star::Star;

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

        let (x0, y0) = grid_origin(size, h);
        let mut pm = ParticleMesh::new(stars.clone(), size, h, x0, y0);
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

    /// Step time of PM with the pair-sum node solver vs the FFT one, on the
    /// app's galaxy. Run with:
    /// cargo test --release pm_speed -- --ignored --nocapture
    #[test]
    #[ignore]
    fn pm_speed() {
        let dt = 0.016;
        let steps = 20;
        let stars = generate_galaxy(&GALAXY);

        let time = |pm: &mut ParticleMesh| {
            pm.update(dt);
            let start = std::time::Instant::now();
            for _ in 0..steps {
                pm.update(dt);
            }
            start.elapsed().as_secs_f64() * 1000.0 / steps as f64
        };

        // same area (640 x 640) with finer and finer grids
        for (size, h) in [(64, 10.0), (128, 5.0), (256, 2.5), (512, 1.25)] {
            let (x0, y0) = grid_origin(size, h);
            let fft = time(&mut ParticleMesh::new_fft(stars.clone(), size, h, x0, y0));
            let direct = if size <= 256 {
                let ms = time(&mut ParticleMesh::new(stars.clone(), size, h, x0, y0));
                format!("{ms:9.2} ms")
            } else {
                "  too slow".to_string()
            };
            println!("grid {size:3}x{size:3}, h = {h:5.2}: direct {direct}, fft {fft:7.2} ms");
        }
    }

    /// How PM (FFT) error against CpuDirect grows over time, for several grids.
    /// As a baseline, also direct started from positions shifted by 0.001:
    /// the part of the growth that comes from the system being chaotic, not
    /// from PM. Uses the app's galaxy with 10x fewer but 10x heavier stars
    /// (same total mass, same potential) so direct stays affordable. Run with:
    /// cargo test --release pm_error_growth -- --ignored --nocapture
    #[test]
    #[ignore]
    fn pm_error_growth() {
        let galaxy = GalaxyConfig {
            star_count: GALAXY.star_count / 10,
            star_mass: GALAXY.star_mass * 10.0,
            ..GALAXY
        };
        let dt = 0.016;
        let steps = 300;
        let every = 10;
        let print_every = 30;

        let stars = generate_galaxy(&galaxy);

        let mean_distance = |a: &[Star], b: &[[f32; 2]]| {
            let total: f32 = a
                .iter()
                .zip(b)
                .map(|(s, p)| ((s.pos[0] - p[0]).powi(2) + (s.pos[1] - p[1]).powi(2)).sqrt())
                .sum();
            total / a.len() as f32
        };

        // direct positions at every checkpoint
        let mut direct = CpuDirect::new(stars.clone());
        let mut reference = Vec::new();
        for step in 1..=steps {
            direct.update(dt);
            if step % every == 0 {
                reference.push(direct.stars().iter().map(|s| s.pos).collect::<Vec<_>>());
            }
        }

        // mean error against direct at every checkpoint
        let run = |update: &mut dyn FnMut() -> Vec<Star>| {
            let mut row = Vec::new();
            for step in 1..=steps {
                let stars = update();
                if step % every == 0 {
                    row.push(mean_distance(&stars, &reference[row.len()]));
                }
            }
            row
        };

        let mut names = Vec::new();
        let mut errors = Vec::new();

        let mut shifted = stars.clone();
        for s in shifted.iter_mut().skip(1) {
            s.pos[0] += 0.001;
        }
        let mut perturbed = CpuDirect::new(shifted);
        names.push("direct +0.001".to_string());
        errors.push(run(&mut || {
            perturbed.update(dt);
            perturbed.stars().to_vec()
        }));

        for (size, h) in [(63, 10.0), (127, 5.0), (255, 2.5)] {
            let (x0, y0) = grid_origin(size, h);
            let mut pm = ParticleMesh::new_fft(stars.clone(), size, h, x0, y0);
            names.push(format!("fft {size}x{size} h={h}"));
            errors.push(run(&mut || {
                pm.update(dt);
                pm.stars().to_vec()
            }));
        }

        println!("mean error in % of galaxy radius ({})", galaxy.radius);
        print!("step    time");
        for name in &names {
            print!(" | {name:>18}");
        }
        println!();
        for c in (print_every / every - 1..reference.len()).step_by(print_every / every) {
            let step = (c + 1) * every;
            print!("{step:4} {:6.2}s", step as f32 * dt);
            for row in &errors {
                print!(" | {:17.3}%", row[c] / galaxy.radius * 100.0);
            }
            println!();
        }

        // error ~ e^(t / tau) until it saturates at the galaxy scale: fit the
        // slope of ln(error) over checkpoints below 5% of the radius
        println!();
        for (name, row) in names.iter().zip(&errors) {
            let points: Vec<(f32, f32)> = row
                .iter()
                .enumerate()
                .filter(|&(_, &e)| e > 0.0 && e < 0.05 * galaxy.radius)
                .map(|(c, &e)| (((c + 1) * every) as f32, e.ln()))
                .collect();
            if points.len() < 2 {
                println!("{name:>18}: not enough points before saturation");
                continue;
            }
            let k = points.len() as f32;
            let (sx, sy) = points.iter().fold((0.0, 0.0), |(sx, sy), &(x, y)| (sx + x, sy + y));
            let (mx, my) = (sx / k, sy / k);
            let (num, den) = points
                .iter()
                .fold((0.0, 0.0), |(n, d), &(x, y)| (n + (x - mx) * (y - my), d + (x - mx).powi(2)));
            let slope = num / den;
            let doubling = 2f32.ln() / slope;
            let reaches = points.last().unwrap().0;
            println!(
                "{name:>18}: error doubles every {doubling:4.1} steps ({:.2}s), fitted over steps {}..{reaches}",
                doubling * dt,
                points[0].0,
            );
        }
    }
}
