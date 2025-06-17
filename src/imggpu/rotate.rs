pub use super::gpu::GpuExecutor;
use image::{EncodableLayout, RgbImage};
use tracing::{span, Level};

pub fn rotate(gpu: &GpuExecutor, img: &RgbImage, theta: f32, default: [f32; 4]) -> RgbImage {
    let span = span!(Level::INFO, "rotate");
    let _guard = span.enter();

    let width = img.width();
    let height = img.height();
    let mut input_buffer = gpu.create_input_image_buffer(width, height);
    let rotation_uniform = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rotation_uniform"),
        size: 8, // 2 f32s
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let color_uniform = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rotation_uniform"),
        size: 16, // 4 f32s
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let (output_texture, mut output_buffer) = gpu.create_output_texture_pair(width, height);

    let shader = gpu
        .device
        .create_shader_module(wgpu::include_wgsl!("rotate.wgsl"));

    let pipeline = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("rotate_image_nearest"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind_group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: rotation_uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: color_uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(
                    &output_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                ),
            },
        ],
    });

    gpu.queue
        .write_buffer(&rotation_uniform, 0, &[theta.cos(), theta.sin()].as_bytes());
    gpu.queue
        .write_buffer(&color_uniform, 0, &default.as_bytes());

    gpu.load_image(img, &mut input_buffer);
    gpu.execute(
        &pipeline,
        &bind_group,
        &output_texture,
        &mut output_buffer,
        width,
        height,
    )
}
