use std::sync::Arc;

use winit::window::Window;

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    surface: Option<wgpu::Surface<'static>>,
    window: Option<Arc<Window>>,
}

impl GpuContext {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let instance = instance();
        let surface = instance.create_surface(window.clone())?;
        let (adapter, device, queue) = request_device(&instance, Some(&surface)).await?;

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let config = wgpu::SurfaceConfiguration {
            format: caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]),
            present_mode: caps.present_modes[0],
            alpha_mode: caps.alpha_modes[0],
            ..surface_config(size.width.max(1), size.height.max(1))
        };
        surface.configure(&device, &config);

        Ok(Self { device, queue, config, surface: Some(surface), window: Some(window) })
    }

    #[cfg(test)]
    pub async fn headless() -> anyhow::Result<Self> {
        let instance = instance();
        let (_, device, queue) = request_device(&instance, None).await?;
        let config = surface_config(1, 1);
        Ok(Self { device, queue, config, surface: None, window: None })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if let Some(surface) = &self.surface
            && width > 0
            && height > 0
        {
            self.config.width = width;
            self.config.height = height;
            surface.configure(&self.device, &self.config);
        }
    }

    pub fn request_redraw(&self) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    pub fn acquire(&self) -> Option<wgpu::SurfaceTexture> {
        match self.surface.as_ref()?.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => Some(t),
            _ => None,
        }
    }
}

async fn request_device(
    instance: &wgpu::Instance,
    surface: Option<&wgpu::Surface<'_>>,
) -> anyhow::Result<(wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: surface,
            force_fallback_adapter: false,
        })
        .await?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor { required_limits: adapter.limits(), ..Default::default() })
        .await?;
    Ok((adapter, device, queue))
}

fn surface_config(width: u32, height: u32) -> wgpu::SurfaceConfiguration {
    wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        width,
        height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    }
}

fn instance() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        flags: Default::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: Default::default(),
    })
}
