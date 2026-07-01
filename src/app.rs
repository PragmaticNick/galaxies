use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;

use winit::{
    application::ApplicationHandler,
    event::{KeyEvent, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

use crate::galaxy::{generate_galaxy, GalaxyConfig, G};
use crate::renderer::Renderer;
use crate::star::Star;

#[derive(Clone, Copy, PartialEq, Eq)]
enum PhysicsStrategy {
    PlainLoop,
    Rayon,
}

impl PhysicsStrategy {
    fn name(&self) -> &'static str {
        match self {
            PhysicsStrategy::PlainLoop => "plain loop",
            PhysicsStrategy::Rayon => "rayon",
        }
    }
}

pub struct App {
    renderer: Option<Renderer>,
    stars: Vec<Star>,
    last_frame: Option<Instant>,
    time: f32,
    strategy: PhysicsStrategy,
}

impl App {
    pub fn new() -> Self {
        let config = GalaxyConfig {
            center: [0.0, 0.0],
            radius: 600.0,
            star_count: 1000,
            core_mass: 50000.0,
            arm_count: 4,
            arm_rotation_factor: 4.0,
            arm_max_offset: 0.4,
            star_mass: 0.1,
            star_radius: 2.0,
            gap: 20.0,
        };

        Self {
            renderer: None,
            stars: generate_galaxy(&config),
            last_frame: None,
            time: 0.0,
            strategy: PhysicsStrategy::PlainLoop,
        }
    }

    fn handle_key(&mut self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        match (code, is_pressed) {
            (KeyCode::Escape, true) => event_loop.exit(),
            (KeyCode::Digit1, true) => self.set_strategy(PhysicsStrategy::PlainLoop),
            (KeyCode::Digit2, true) => self.set_strategy(PhysicsStrategy::Rayon),
            _ => {}
        }
    }

    fn set_strategy(&mut self, strategy: PhysicsStrategy) {
        if self.strategy != strategy {
            self.strategy = strategy;
            log::info!("physics strategy: {}", strategy.name());
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes();
        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        self.renderer = Some(pollster::block_on(Renderer::new(window)).unwrap());
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let renderer = match &mut self.renderer {
            Some(r) => r,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => renderer.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = match self.last_frame {
                    Some(t) => now.duration_since(t).as_secs_f32().min(0.016),
                    None => 0.0,
                };
                self.last_frame = Some(now);
                self.time += dt;
                update_physics(&mut self.stars, dt, self.strategy);
                match renderer.render(&self.stars) {
                    Ok(_) => {}
                    Err(e) => {
                        log::error!("{e}");
                        event_loop.exit();
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => self.handle_key(event_loop, code, key_state.is_pressed()),
            _ => {}
        }
    }
}

fn update_physics(stars: &mut [Star], dt: f32, strategy: PhysicsStrategy) {
    let accels = match strategy {
        PhysicsStrategy::PlainLoop => accels_plain(stars),
        PhysicsStrategy::Rayon => accels_rayon(stars),
    };

    for i in 1..stars.len() {
        stars[i].vel[0] += accels[i][0] * dt;
        stars[i].vel[1] += accels[i][1] * dt;
        stars[i].pos[0] += stars[i].vel[0] * dt;
        stars[i].pos[1] += stars[i].vel[1] * dt;
    }
}

// O(n^2) accel, single-threaded
fn accels_plain(stars: &[Star]) -> Vec<[f32; 2]> {
    let n = stars.len();
    (0..n)
        .map(|i| body_accel(stars, i))
        .collect()
}

fn accels_rayon(stars: &[Star]) -> Vec<[f32; 2]> {
    let n = stars.len();
    (0..n)
        .into_par_iter()
        .map(|i| body_accel(stars, i))
        .collect()
}

fn body_accel(stars: &[Star], i: usize) -> [f32; 2] {
    if i == 0 {
        return [0.0, 0.0];
    }
    let si = &stars[i];
    let mut ax = 0.0;
    let mut ay = 0.0;
    for j in 0..stars.len() {
        if i == j {
            continue;
        }
        let [fx, fy] = gravity(si, &stars[j]);
        ax += fx / si.mass;
        ay += fy / si.mass;
    }
    [ax, ay]
}

fn gravity(s: &Star, t: &Star) -> [f32; 2] {
    const SOFTENING2: f32 = 400.0;
    let dx = t.pos[0] - s.pos[0];
    let dy = t.pos[1] - s.pos[1];
    let r2 = dx * dx + dy * dy + SOFTENING2;
    let r = r2.sqrt();
    let f = G * s.mass * t.mass / r2;
    [f * dx / r, f * dy / r]
}
