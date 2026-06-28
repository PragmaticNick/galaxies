use std::sync::Arc;
use std::time::Instant;

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

pub struct App {
    renderer: Option<Renderer>,
    stars: Vec<Star>,
    last_frame: Option<Instant>,
}

impl App {
    pub fn new() -> Self {
        let config = GalaxyConfig {
            center: [0.0, 0.0],
            radius: 300.0,
            star_count: 1000,
            star_mass: 0.1,
            star_radius: 2.0,
            arm_count: 4,
            arm_rotation_factor: 2.5,
            arm_max_offset: 0.25,
            gap: 10.0,
            core_mass: 50000.0,
            bulge_fraction: 0.25,
        };

        Self {
            renderer: None,
            stars: generate_galaxy(&config),
            last_frame: None,
        }
    }

    fn handle_key(&self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        match (code, is_pressed) {
            (KeyCode::Escape, true) => event_loop.exit(),
            _ => {}
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

                update_physics(&mut self.stars, dt);

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

fn update_physics(stars: &mut Vec<Star>, dt: f32) {
    let n = stars.len();
    let mut ax = vec![0.0f32; n];
    let mut ay = vec![0.0f32; n];

    for i in 1..n {
        for j in 0..n {
            if i == j { continue; }
            let [fx, fy] = gravity(&stars[i], &stars[j]);
            ax[i] += fx / stars[i].mass;
            ay[i] += fy / stars[i].mass;
        }
    }

    for i in 1..n {
        stars[i].vel[0] += ax[i] * dt;
        stars[i].vel[1] += ay[i] * dt;
        stars[i].pos[0] += stars[i].vel[0] * dt;
        stars[i].pos[1] += stars[i].vel[1] * dt;
    }
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
