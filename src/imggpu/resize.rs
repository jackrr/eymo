use image::{RgbImage, RgbaImage, DynamicImage};
use pollster::FutureExt;
use wgpu::{wgt::SamplerDescriptor, AddressMode, Device, FilterMode, Queue, SamplerBindingType, TextureDescriptor};

pub enum ResizeAlgo {
    Nearest,
    Linear,
}

impl ResizeAlgo {
    fn shader_name(&self) -> &str {
        match *self {
            Self::Nearest => "resize_image_nearest",
            Self::Linear => "resize_image_sampler",
        }
    }
}

fn padded_bytes_per_row(width: u32) -> usize {
    let bytes_per_row = width as usize * 4;
    let padding = (256 - bytes_per_row % 256) % 256;
    bytes_per_row + padding
}

fn int_div_round_up(divisor: u32, dividend: u32) -> u32 {
	(divisor / dividend) + match divisor % dividend {
		0 => 0,
		_ => 1
	}
}

async fn init_wgpu() -> (Device, Queue) {
    // TODO: less unwrap, more proper error bubbling
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
			  backends: wgpu::Backends::all(),
			  flags: wgpu::InstanceFlags::VALIDATION,
			  backend_options: wgpu::BackendOptions::default()
		});
    
		let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();

		let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
			required_features: wgpu::Features::empty(),
			required_limits: wgpu::Limits::default(),
			memory_hints: wgpu::MemoryHints::Performance,
			label: Some("device"),
			trace: wgpu::Trace::Off
		}).await.unwrap();

    return (device, queue)
}

pub fn resize(img: &RgbImage, width: u32, height: u32, algo: ResizeAlgo) -> RgbImage {
    // TODO: Avoid costly? alpha conversions
    let img = DynamicImage::ImageRgb8(img.clone()).into_rgba8();

    let (device, queue) = init_wgpu().block_on();
    
    let input_texture_size = wgpu::Extent3d {
        width: img.width(),
        height: img.height(),
        depth_or_array_layers: 1,
    };

    let output_texture_size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    let input_texture = device.create_texture(
        &TextureDescriptor {
            label: Some("input image"),
            size: input_texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST
        }
    );

    queue.write_texture(
        input_texture.as_image_copy(),
        img.as_raw(),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4*img.width()),
            rows_per_image: None,
        },
        input_texture_size);



    // let mut sampler_descriptor = SamplerDescriptor::default();
    // sampler_descriptor.min_filter = FilterMode::Linear;
    // sampler_descriptor.mag_filter = FilterMode::Linear;
    // sampler_descriptor.address_mode_u = AddressMode::ClampToEdge;
    // sampler_descriptor.address_mode_v = AddressMode::ClampToEdge;

    // let sampler = device.create_sampler(&sampler_descriptor);

    let output_texture = device.create_texture(
        &TextureDescriptor {
            label: Some("output image"),
            size: output_texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC
        }
    );

    let shader = device.create_shader_module(wgpu::include_wgsl!("resize.wgsl"));

    // let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
    //     label: Some("bind group layout"),
    //     entries: &[
    //         wgpu::BindGroupLayoutEntry {
    //             binding: 0,
    //             visibility: wgpu::ShaderStages::COMPUTE,
    //             ty: wgpu::BindingType::Texture {
    //                 sample_type: wgpu::TextureSampleType::default(),
    //                 view_dimension: wgpu::TextureViewDimension::D2,
    //                 multisampled: false
    //             },
    //             count: None
    //         },
    //         wgpu::BindGroupLayoutEntry {
    //             binding: 1,
    //             visibility: wgpu::ShaderStages::COMPUTE,
    //             ty: wgpu::BindingType::Sampler(SamplerBindingType::Filtering),
    //             count: None,
    //         },
    //         wgpu::BindGroupLayoutEntry {
    //             binding: 2,
    //             visibility: wgpu::ShaderStages::COMPUTE,
    //             ty: wgpu::BindingType::Texture {
    //                 sample_type: wgpu::TextureSampleType::default(),
    //                 view_dimension: wgpu::TextureViewDimension::D2,
    //                 multisampled: false
    //             },
    //             count: None
    //         },
    //     ]
    // });

    // let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
		// 	label: Some("pipeline_layout"),
		// 	bind_group_layouts: &[&bind_group_layout],
		// 	push_constant_ranges: &[]
		// });

		let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
			  label: Some("pipeline"),
			  layout: None,
			  module: &shader,
			  entry_point: algo.shader_name().into(),
			  compilation_options: wgpu::PipelineCompilationOptions::default(),
			  cache: None
		});

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			  label: Some("bind_group"),
			  layout: &pipeline.get_bind_group_layout(0),
			  entries: &[
				    wgpu::BindGroupEntry {
					      binding: 0,
					      resource: wgpu::BindingResource::TextureView(&input_texture.create_view(&wgpu::TextureViewDescriptor::default())),
				    },
				    // wgpu::BindGroupEntry {
					  //     binding: 1,
					  //     resource: wgpu::BindingResource::Sampler(&sampler),
				    // },
				    wgpu::BindGroupEntry {
					      binding: 1,
					      resource:  wgpu::BindingResource::TextureView(&output_texture.create_view(&wgpu::TextureViewDescriptor::default())),
				    }
			  ]
		});


    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
			  label: Some("encoder")
		});

    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("compute"),
        timestamp_writes: None
    });

    compute_pass.set_pipeline(&pipeline);
    compute_pass.set_bind_group(0, &bind_group, &[]);
    compute_pass.dispatch_workgroups(
        int_div_round_up(img.width(), 8),
        int_div_round_up(img.height(), 8),
        1
    );
    drop(compute_pass);

    let padded_bytes_per_row = padded_bytes_per_row(width);
    let unpadded_bytes_per_row = width as usize * 4;

    let output_buffer_size =
        padded_bytes_per_row as u64 * height as u64 * std::mem::size_of::<u8>() as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: output_buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            aspect: wgpu::TextureAspect::All,
            texture: &output_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        wgpu::TexelCopyBufferInfo{
            buffer: &output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: (padded_bytes_per_row as u32).into(),
                rows_per_image: height.into(),
            },
        },
        output_texture_size,
    );

    queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = output_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());

    device.poll(wgpu::PollType::Wait).unwrap();

    let padded_data = buffer_slice.get_mapped_range();

    let mut pixels: Vec<u8> = vec![0; unpadded_bytes_per_row * height as usize];
    for (padded, pixels) in padded_data
        .chunks_exact(padded_bytes_per_row)
        .zip(pixels.chunks_exact_mut(unpadded_bytes_per_row)) {
            pixels.copy_from_slice(&padded[..unpadded_bytes_per_row]);
        }

    let with_alpha = RgbaImage::from_raw(width, height, pixels).unwrap();
    DynamicImage::ImageRgba8(with_alpha).to_rgb8()
}
