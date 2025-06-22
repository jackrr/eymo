use super::gpu::GpuExecutor;
use super::util::int_div_round_up;
use anyhow::Result;
use image::Rgb;
use ort::value::Tensor;
use tracing::{debug, info, span, Level};

pub fn texture_to_tensor(gpu: &mut GpuExecutor, texture: &wgpu::Texture) -> Result<Tensor<f32>> {
    let span = span!(Level::INFO, "texture_to_tensor");
    let _guard = span.enter();

    let shader_code = wgpu::include_wgsl!("rgb.wgsl");
    let shader = gpu.load_shader("tex_to_rgb", shader_code);

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoder"),
        });

    // w x h x rgb x size(f32)
    let buffer_size = texture.width() * texture.height() * 3 * 4;
    let output_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output_buffer"),
        size: buffer_size.into(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let compute_pipeline = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compute_pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("tex_to_rgb_buf"),
            compilation_options: Default::default(),
            cache: None,
        });

    let compute_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("render_bind_group"),
        layout: &compute_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &texture.create_view(&Default::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("compute"),
        timestamp_writes: None,
    });

    compute_pass.set_pipeline(&compute_pipeline);
    compute_pass.set_bind_group(0, &compute_bg, &[]);
    compute_pass.dispatch_workgroups(
        int_div_round_up(texture.width(), 8),
        int_div_round_up(texture.height(), 8),
        1,
    );
    drop(compute_pass);

    let map_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("map_buf"),
        size: buffer_size.into(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &map_buffer, 0, buffer_size.into());

    gpu.queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = map_buffer.slice(..);
    debug!("Buffer size {buffer_size:?}");
    buffer_slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());

    gpu.device.poll(wgpu::PollType::Wait)?;

    let buffer_data = buffer_slice.get_mapped_range();
    let res = bytemuck::cast_slice::<u8, f32>(&*buffer_data).to_vec();
    debug!("First pixel: {:?} {:?} {:?}", res[0], res[1], res[2]);

    let tensor = Tensor::from_array((
        [1, texture.height() as usize, texture.width() as usize, 3],
        res,
    ))?;
    debug!("{tensor:?}");

    Ok(tensor)
}
