use std::sync::Arc;
use std::time::Instant;

use winit::{
    application::ApplicationHandler,
    event::{KeyEvent, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Fullscreen, Window},
};

use crate::galaxy::{GalaxyConfig, generate_galaxy};
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
            radius: 200.0,
            star_count: 50000,
            core_radius: 2.0,
            core_mass: 500000.0,
            star_mass: 10.0,
            star_radius: 1.5,
            gap: 20.0,
        };

        Self {
            renderer: None,
            stars: generate_galaxy(&config),
            last_frame: None,
        }
    }

    fn handle_key(&mut self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        match (code, is_pressed) {
            (KeyCode::Escape, true) => event_loop.exit(),
            _ => {}
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
        let window_attributes = Window::default_attributes();
        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        window.set_fullscreen(Some(Fullscreen::Borderless(None)));
        self.renderer = Some(pollster::block_on(Renderer::new(window, &self.stars)).unwrap());
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(r) = &self.renderer {
            r.request_redraw();
        }
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
                match renderer.render(self.stars.len(), dt) {
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
