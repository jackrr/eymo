use super::gpu::GpuExecutor;
use super::util::{int_div_round_up, padded_bytes_per_row};
use anyhow::Result;
use image::RgbaImage;
use ort::value::Tensor;
use tracing::{debug, span, Level};

pub enum OutputRange {
    ZeroToOne,
    NegOneToOne,
}

impl OutputRange {
    fn entry_point(&self) -> &str {
        match self {
            Self::NegOneToOne => "tex_to_rgb_buf_neg1_1",
            Self::ZeroToOne => "tex_to_rgb_buf_0_1",
        }
    }
}

pub fn texture_to_rgba(gpu: &GpuExecutor, texture: &wgpu::Texture) -> RgbaImage {
    // ~9ms
    let span = span!(Level::DEBUG, "texture_to_rgba");
    let _guard = span.enter();

    let width = texture.width();
    let height = texture.height();

    let padded_bytes_per_row = padded_bytes_per_row(width);
    let unpadded_bytes_per_row = width as usize * 4;

    let buffer_size =
        padded_bytes_per_row as u64 * height as u64 * std::mem::size_of::<u8>() as u64;
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("snapshot_buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoder"),
        });

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            aspect: wgpu::TextureAspect::All,
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: (padded_bytes_per_row as u32).into(),
                rows_per_image: height.into(),
            },
        },
        texture.size(),
    );

    gpu.queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());

    gpu.device.poll(wgpu::PollType::Wait).unwrap();

    let padded_data = buffer_slice.get_mapped_range();
    let mut pixels: Vec<u8> = vec![0; unpadded_bytes_per_row * height as usize];
    for (padded, pixels) in padded_data
        .chunks_exact(padded_bytes_per_row)
        .zip(pixels.chunks_exact_mut(unpadded_bytes_per_row))
    {
        pixels.copy_from_slice(&padded[..unpadded_bytes_per_row]);
    }
    drop(padded_data);

    RgbaImage::from_raw(width, height, pixels).unwrap()
}

pub fn texture_to_tensor(
    gpu: &mut GpuExecutor,
    texture: &wgpu::Texture,
    output_range: OutputRange,
) -> Result<Tensor<f32>> {
    let span = span!(Level::DEBUG, "texture_to_tensor");
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
            entry_point: Some(output_range.entry_point()),
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
    let tensor = Tensor::from_array((
        [1, texture.height() as usize, texture.width() as usize, 3],
        res,
    ))?;
    debug!("{tensor:?}");

    Ok(tensor)
}
