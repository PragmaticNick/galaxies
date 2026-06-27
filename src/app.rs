use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    event::{KeyEvent, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

use crate::renderer::Renderer;
use crate::square::Square;

pub struct App {
    renderer: Option<Renderer>,
    squares: Vec<Square>,
}

impl App {
    pub fn new() -> Self {
        Self {
            renderer: None,
            squares: vec![
                Square {
                    center: [-80.0, 0.0],
                    radius: 150.0,
                    color: [1.0, 0.2, 0.2],
                },
                Square {
                    center: [80.0, 0.0],
                    radius: 150.0,
                    color: [1.0, 0.2, 0.2],
                },
                Square {
                    center: [0.0, 100.0],
                    radius: 100.0,
                    color: [1.0, 0.2, 0.2],
                },
            ],
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
