use std::sync::Arc;
use std::time::Instant;

use winit::{
    application::ApplicationHandler,
    event::{KeyEvent, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

use crate::galaxy::{GalaxyConfig, generate_galaxy};
use crate::physics::{PhysicsStrategy, update_physics};
use crate::renderer::Renderer;
use crate::star::Star;

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
            star_count: 4000,
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
            strategy: PhysicsStrategy::Rayon,
        }
    }

    fn handle_key(&mut self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        match (code, is_pressed) {
            (KeyCode::Escape, true) => event_loop.exit(),
            (KeyCode::Digit1, true) => self.set_strategy(PhysicsStrategy::PlainLoop),
            (KeyCode::Digit2, true) => self.set_strategy(PhysicsStrategy::Rayon),
            (KeyCode::Digit3, true) => self.set_strategy(PhysicsStrategy::FmmSerial),
            (KeyCode::Digit4, true) => self.set_strategy(PhysicsStrategy::FmmRayon),
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
        self.renderer = Some(pollster::block_on(Renderer::new(window, &self.stars)).unwrap());
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
                match renderer.render(self.stars.len(), self.strategy.name()) {
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
