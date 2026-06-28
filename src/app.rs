use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    event::{KeyEvent, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

use crate::{galaxy::GalaxyConfig, renderer::Renderer};
use crate::{galaxy::generate_galaxy, square::Square};

pub struct App {
    renderer: Option<Renderer>,
    squares: Vec<Square>,
}

impl App {
    pub fn new() -> Self {
        let config = GalaxyConfig {
            center: [0.0, 0.0],
            radius: 300.0,
            star_count: 3000,
            star_mass: 10.0,
            star_radius: 2.0,
            arm_count: 10,
            arm_rotation_factor: 3.0,
            arm_max_offset: 0.6,
            gap: 10.0,
            core_mass: 1000000.0,
        };

        Self {
            renderer: None,
            squares: generate_galaxy(&config),
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
            WindowEvent::RedrawRequested => match renderer.render(&self.squares) {
                Ok(_) => {}
                Err(e) => {
                    log::error!("{e}");
                    event_loop.exit();
                }
            },
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
