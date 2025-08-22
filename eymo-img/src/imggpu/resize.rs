use super::gpu::GpuExecutor;
use super::vertex::Vertex;
use anyhow::Result;
use tracing::{Level, span};
use wgpu::util::DeviceExt;

pub fn resize_texture(
    gpu: &mut GpuExecutor,
    tex: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Result<wgpu::Texture> {
    let span = span!(Level::DEBUG, "resize_texture");
    let _guard = span.enter();

    let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let shader_code = wgpu::include_wgsl!("resize.wgsl");
    let shader = gpu.load_shader("resize", shader_code);

    let render_pipeline = gpu
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render_pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vert_main"),
                compilation_options: Default::default(),
                buffers: &[Vertex::desc()],
            },
            primitive: wgpu::PrimitiveState {
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("frag_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

    let out_dims = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out_dims"),
        size: 8,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(
        &out_dims,
        0,
        &bytemuck::cast_slice(&[(width as f32), (height as f32)]),
    );

    let render_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("render_bind_group"),
        layout: &render_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&tex.create_view(&Default::default())),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_dims.as_entire_binding(),
            },
        ],
    });

    let resize_output_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("resized_texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::TEXTURE_BINDING,
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoder"),
        });

    let vertices = Vertex::triangles_for_full_coverage();
    let vertex_buffer = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertex_buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("render_pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &resize_output_tex.create_view(&Default::default()),
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(Default::default()),
                store: wgpu::StoreOp::Store, // overwrite with fragment output
            },
        })],
        ..Default::default()
    });

    render_pass.set_pipeline(&render_pipeline);
    render_pass.set_bind_group(0, &render_bg, &[]);
    render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
    render_pass.draw(0..vertices.len() as u32, 0..1);
    drop(render_pass);

    gpu.queue.submit(std::iter::once(encoder.finish()));

    Ok(resize_output_tex)
}
