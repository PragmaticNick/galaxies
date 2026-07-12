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
use crate::physics::{EPS, G};
use crate::renderer::Renderer;
use crate::star::Star;

struct GalaxyPreset {
    name: &'static str,
    config: GalaxyConfig,
    eps: f32,
    g: f32,
}

fn presets() -> Vec<GalaxyPreset> {
    vec![
        GalaxyPreset {
            name: "Classic Spiral",
            config: GalaxyConfig {
                center: [0.0, 0.0],
                radius: 200.0,
                star_count: 50000,
                core_radius: 2.0,
                core_mass: 500000.0,
                star_mass: 10.0,
                star_radius: 1.5,
                gap: 20.0,
            },
            eps: EPS,
            g: G,
        },
        GalaxyPreset {
            name: "Dense Core",
            config: GalaxyConfig {
                center: [0.0, 0.0],
                radius: 120.0,
                star_count: 30000,
                core_radius: 1.5,
                core_mass: 2_000_000.0,
                star_mass: 8.0,
                star_radius: 1.2,
                gap: 10.0,
            },
            eps: 20.0,
            g: 100.0,
        },
        GalaxyPreset {
            name: "Sparse Giant",
            config: GalaxyConfig {
                center: [0.0, 0.0],
                radius: 400.0,
                star_count: 20000,
                core_radius: 3.0,
                core_mass: 300000.0,
                star_mass: 15.0,
                star_radius: 2.0,
                gap: 60.0,
            },
            eps: 50.0,
            g: 100.0,
        },
        GalaxyPreset {
            name: "Chaotic High-G",
            config: GalaxyConfig {
                center: [0.0, 0.0],
                radius: 180.0,
                star_count: 40000,
                core_radius: 2.0,
                core_mass: 400000.0,
                star_mass: 12.0,
                star_radius: 1.5,
                gap: 15.0,
            },
            eps: 15.0,
            g: 250.0,
        },
    ]
}

pub struct App {
    renderer: Option<Renderer>,
    stars: Vec<Star>,
    presets: Vec<GalaxyPreset>,
    current: usize,
    last_frame: Option<Instant>,
}

impl App {
    pub fn new() -> Self {
        let presets = presets();
        let current = 0;
        let stars = generate_galaxy(&presets[current].config, presets[current].eps, presets[current].g);

        Self {
            renderer: None,
            stars,
            presets,
            current,
            last_frame: None,
        }
    }

    fn switch_preset(&mut self, delta: i32) {
        let len = self.presets.len() as i32;
        self.current = (self.current as i32 + delta).rem_euclid(len) as usize;
        let preset = &self.presets[self.current];
        self.stars = generate_galaxy(&preset.config, preset.eps, preset.g);
        if let Some(r) = &mut self.renderer {
            r.set_stars(&self.stars);
            r.set_physics(preset.eps, preset.g, preset.config.radius, preset.config.center);
            r.set_preset_name(preset.name);
        }
    }

    fn handle_key(&mut self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        match (code, is_pressed) {
            (KeyCode::Escape, true) => event_loop.exit(),
            (KeyCode::ArrowLeft, true) => self.switch_preset(-1),
            (KeyCode::ArrowRight, true) => self.switch_preset(1),
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
        let preset = &self.presets[self.current];
        self.renderer = Some(
            pollster::block_on(Renderer::new(
                window,
                &self.stars,
                preset.eps,
                preset.g,
                preset.config.radius,
                preset.config.center,
                preset.name,
            ))
            .unwrap(),
        );
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
