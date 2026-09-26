use std::time::Instant;

use glyphon::{
    Attrs, Buffer, Cache, Color, Family, FontSystem, Metrics, Resolution, Shaping, SwashCache,
    TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};

use crate::gpu::GpuContext;

pub struct TextOverlay {
    font_system: FontSystem,
    swash_cache: SwashCache,
    viewport: Viewport,
    atlas: TextAtlas,
    renderer: TextRenderer,
    buffer: Buffer,
    info: String,
    fps_frames: u32,
    fps_since: Instant,
}

impl TextOverlay {
    pub fn new(gpu: &GpuContext, star_count: usize, strategy_name: &str) -> Self {
        let device = &gpu.device;
        let mut font_system = FontSystem::new();
        let cache = Cache::new(device);
        let mut atlas = TextAtlas::new(device, &gpu.queue, &cache, gpu.config.format);
        let renderer =
            TextRenderer::new(&mut atlas, device, wgpu::MultisampleState::default(), None);
        let mut buffer = Buffer::new(&mut font_system, Metrics::new(32.0, 40.0));
        buffer.set_size(&mut font_system, Some(600.0), Some(400.0));

        let mut overlay = Self {
            font_system,
            swash_cache: SwashCache::new(),
            viewport: Viewport::new(device, &cache),
            atlas,
            renderer,
            buffer,
            info: format!("Stars: {star_count}\nStrategy: {strategy_name}"),
            fps_frames: 0,
            fps_since: Instant::now(),
        };
        overlay.set_fps(0.0);
        overlay
    }

    pub fn prepare(&mut self, gpu: &GpuContext) {
        self.fps_frames += 1;
        let elapsed = self.fps_since.elapsed().as_secs_f32();
        if elapsed >= 1.0 {
            self.set_fps(self.fps_frames as f32 / elapsed);
            self.fps_frames = 0;
            self.fps_since = Instant::now();
        }

        let resolution = Resolution {
            width: gpu.config.width,
            height: gpu.config.height,
        };
        self.viewport.update(&gpu.queue, resolution);
        self.renderer
            .prepare(
                &gpu.device,
                &gpu.queue,
                &mut self.font_system,
                &mut self.atlas,
                &self.viewport,
                [TextArea {
                    buffer: &self.buffer,
                    left: 10.0,
                    top: 10.0,
                    scale: 1.0,
                    bounds: TextBounds {
                        left: 0,
                        top: 0,
                        right: 600,
                        bottom: 400,
                    },
                    default_color: Color::rgb(255, 255, 255),
                    custom_glyphs: &[],
                }],
                &mut self.swash_cache,
            )
            .unwrap();
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass) {
        self.renderer
            .render(&self.atlas, &self.viewport, pass)
            .unwrap();
    }

    fn set_fps(&mut self, fps: f32) {
        self.buffer.set_text(
            &mut self.font_system,
            &format!("FPS: {fps:.0}\n{}", self.info),
            &Attrs::new().family(Family::Monospace),
            Shaping::Basic,
            None,
        );
        self.buffer.shape_until_scroll(&mut self.font_system, false);
    }
}
