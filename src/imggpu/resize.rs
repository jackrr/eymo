use image::{RgbImage, RgbaImage, DynamicImage};
use pollster::FutureExt;
use wgpu::{wgt::SamplerDescriptor, AddressMode, Device, FilterMode, Queue, SamplerBindingType, TextureDescriptor};
use tracing::{span, Level};

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

// ~65-100ms
fn rgb_to_rgba_bytes(img: &RgbImage) -> Vec<u8> {
    let rgba_bytes = (img.width() * img.height() * 4) as usize;
    let mut rgba_raw: Vec<u8> = vec![0; rgba_bytes];
    for (src, dest) in img.as_raw().chunks_exact(3).zip(rgba_raw.chunks_exact_mut(4)) {
        dest[..3].copy_from_slice(src);
    }
    rgba_raw
}

// TODO: cache wgpu init
// TODO: cache texture init somehow (1 per I/O size)
// TODO: skip alpha conversion somehow

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

// Goal: 10ms total runtime
pub fn resize(img: &RgbImage, width: u32, height: u32, algo: ResizeAlgo) -> RgbImage {
    let span = span!(Level::INFO, "resize");
    let _guard = span.enter();

    // 80-120ms
    // TODO: Avoid costly alpha conversions
    let alpha_span = span!(Level::INFO, "resize_convert_to_alpha");
    let alpha_guard = alpha_span.enter();

    // Roughly same perf...
    // let img_raw_bytes = rgb_to_rgba_bytes(img);
    // let img = DynamicImage::ImageRgb8(img.clone()).into_rgba8();
    drop(alpha_guard);

    // 30-40ms
    let init_span = span!(Level::INFO, "resize_init_gpu");
    let init_guard = init_span.enter();
    let (device, queue) = init_wgpu().block_on();
    drop(init_guard);

    // 14ms
    let textures_span = span!(Level::INFO, "resize_textures");
    let textures_guard = textures_span.enter();
    // let input_texture_size = wgpu::Extent3d {
    //     width: img.width(),
    //     height: img.height(),
    //     depth_or_array_layers: 1,
    // };

    let output_texture_size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    // let input_texture = device.create_texture(
    //     &TextureDescriptor {
    //         label: Some("input image"),
    //         size: input_texture_size,
    //         mip_level_count: 1,
    //         sample_count: 1,
    //         dimension: wgpu::TextureDimension::D2,
    //         format: wgpu::TextureFormat::Rgba8Unorm,
    //         view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
    //         usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST
    //     }
    // );

    // queue.write_texture(
    //     input_texture.as_image_copy(),
    //     img.as_raw(),
    //     // &img_raw_bytes,
    //     wgpu::TexelCopyBufferLayout {
    //         offset: 0,
    //         bytes_per_row: Some(4*img.width()),
    //         rows_per_image: None,
    //     },
    //     input_texture_size);
		let input_buf = device.create_buffer(&wgpu::BufferDescriptor {
			  label: Some("input_buf"),
        // w x h x rgb
			  size: (img.width() * img.height() * 3) as u64,
			usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
			mapped_at_creation: false
		});

		let width_uniform = device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("width_uniform"),
			size: 4,
			usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
			mapped_at_creation: false
		});

		let height_uniform = device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("height_uniform"),
			size: 4,
			usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
			mapped_at_creation: false
		});

    queue.write_buffer(&input_buf, 0, img.as_raw());
    queue.write_buffer(&width_uniform, 0, &img.width().to_ne_bytes());
    queue.write_buffer(&height_uniform, 0, &img.height().to_ne_bytes());

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

    drop(textures_guard);

    // 5ms
    let load_shader_span = span!(Level::INFO, "resize_load_shader");
    let load_shader_guard = load_shader_span.enter();
    let shader = device.create_shader_module(wgpu::include_wgsl!("resize.wgsl"));
    drop(load_shader_guard);

    // 1ms
    let compute_prep_span = span!(Level::INFO, "resize_compute_prep");
    let compute_prep_guard = compute_prep_span.enter();
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
				    // wgpu::BindGroupEntry {
					  //     binding: 0,
					  //     resource: wgpu::BindingResource::TextureView(&input_texture.create_view(&wgpu::TextureViewDescriptor::default())),
				    // },
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.as_entire_binding(),
            },
            
				    // wgpu::BindGroupEntry {
					  //     binding: 1,
					  //     resource: wgpu::BindingResource::Sampler(&sampler),
				    // },
				    wgpu::BindGroupEntry {
					      binding: 1,
					      resource:  wgpu::BindingResource::TextureView(&output_texture.create_view(&wgpu::TextureViewDescriptor::default())),
				    },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: width_uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: height_uniform.as_entire_binding(),
            },
			  ]
		});


    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
			  label: Some("encoder")
		});

    drop(compute_prep_guard);

    // 5ms (nearest, tbd linear)
    let gpu_run_span = span!(Level::INFO, "resize_gpu_run");
    let gpu_run_guard = gpu_run_span.enter();
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
    drop(gpu_run_guard);

    // 1ms
    let extract_image_span = span!(Level::INFO, "resize_extract_image");
    let extract_image_guard = extract_image_span.enter();
    let padded_data = buffer_slice.get_mapped_range();

    let mut pixels: Vec<u8> = vec![0; unpadded_bytes_per_row * height as usize];
    for (padded, pixels) in padded_data
        .chunks_exact(padded_bytes_per_row)
        .zip(pixels.chunks_exact_mut(unpadded_bytes_per_row)) {
            pixels.copy_from_slice(&padded[..unpadded_bytes_per_row]);
        }
    drop(extract_image_guard);

    // 1-2ms
    let alpha_to_rgb_span = span!(Level::INFO, "resize_alpha_to_rgb");
    let alpha_to_rgb_guard = alpha_to_rgb_span.enter();
    
    let with_alpha = RgbaImage::from_raw(width, height, pixels).unwrap();
    let res = DynamicImage::ImageRgba8(with_alpha).to_rgb8();
    drop(alpha_to_rgb_guard);

    res
}
