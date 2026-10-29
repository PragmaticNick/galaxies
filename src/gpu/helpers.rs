use bytemuck::NoUninit;
use wgpu::util::DeviceExt as _;

use crate::gpu::GpuContext;

// Plain additive blending: every shader that draws with it outputs alpha 1.
pub const ADDITIVE: wgpu::BlendState = {
    let add = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    };
    wgpu::BlendState { color: add, alpha: add }
};

impl GpuContext {
    pub fn storage_buffer<T: NoUninit>(&self, data: &[T]) -> wgpu::Buffer {
        self.buffer(data, wgpu::BufferUsages::STORAGE)
    }

    pub fn uniform_buffer<T: NoUninit>(&self, data: &[T]) -> wgpu::Buffer {
        self.buffer(data, wgpu::BufferUsages::UNIFORM)
    }

    fn buffer<T: NoUninit>(&self, data: &[T], usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: usage | wgpu::BufferUsages::COPY_DST,
            })
    }

    pub fn write<T: NoUninit>(&self, buffer: &wgpu::Buffer, data: &[T]) {
        self.queue
            .write_buffer(buffer, 0, bytemuck::cast_slice(data));
    }

    pub fn bind_group(
        &self,
        layout: &wgpu::BindGroupLayout,
        buffers: &[&wgpu::Buffer],
    ) -> wgpu::BindGroup {
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &entries,
        })
    }

    // textures at bindings 0.., then the sampler
    pub fn texture_bind_group(
        &self,
        layout: &wgpu::BindGroupLayout,
        views: &[&wgpu::TextureView],
        sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        let mut entries: Vec<_> = views
            .iter()
            .enumerate()
            .map(|(i, view)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: wgpu::BindingResource::TextureView(view),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: views.len() as u32,
            resource: wgpu::BindingResource::Sampler(sampler),
        });
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &entries,
        })
    }

    // a texture to render into and then sample from
    pub fn render_target(&self, width: u32, height: u32, format: wgpu::TextureFormat) -> wgpu::TextureView {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        texture.create_view(&Default::default())
    }

    pub fn compute_bind_group_layout(
        &self,
        types: &[wgpu::BufferBindingType],
    ) -> wgpu::BindGroupLayout {
        let entries: Vec<_> = types
            .iter()
            .enumerate()
            .map(|(i, &ty)| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        self.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &entries,
            })
    }

    pub fn compute_pipeline(
        &self,
        shader: &wgpu::ShaderModule,
        layout: &wgpu::BindGroupLayout,
        entry: &str,
    ) -> wgpu::ComputePipeline {
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
    }

    pub fn render_pipeline(
        &self,
        shader: &wgpu::ShaderModule,
        format: wgpu::TextureFormat,
        blend: wgpu::BlendState,
        constants: &[(&str, f64)],
    ) -> wgpu::RenderPipeline {
        self.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: None,
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants,
                        ..Default::default()
                    },
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants,
                        ..Default::default()
                    },
                }),
                primitive: wgpu::PrimitiveState {
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
    }
}

pub fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    workgroups: u32,
) {
    let mut pass = encoder.begin_compute_pass(&Default::default());
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(workgroups, 1, 1);
}

pub fn render_pass<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
) -> wgpu::RenderPass<'a> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
        })],
        ..Default::default()
    })
}
