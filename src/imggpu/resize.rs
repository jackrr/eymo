use std::num::NonZero;
use anyhow::Result;

use image::{RgbImage, RgbaImage, DynamicImage};
use pollster::FutureExt;
use tracing::{span, Level};

pub enum ResizeAlgo {
    Nearest,
    Linear,
}

impl ResizeAlgo {
    fn shader_name(&self) -> &str {
        match *self {
            Self::Nearest => "resize_image_nearest",
            Self::Linear => "resize_image_linear",
        }
    }
}

pub fn resize(img: &RgbImage, width: u32, height: u32, algo: ResizeAlgo) -> Result<RgbImage> {
    let span = span!(Level::INFO, "resize");
    let _guard = span.enter();
    let executor = GpuExecutor::new()?;
    Ok(Resizer::new(&executor, img.width(), img.height(), width, height, algo).run(&executor, img))
}

pub fn resize_with_executor(executor: &GpuExecutor, img: &RgbImage, width: u32, height: u32, algo: ResizeAlgo) -> Result<RgbImage> {
    let span = span!(Level::INFO, "resize_with_executor");
    let _guard = span.enter();
    Ok(Resizer::new(&executor, img.width(), img.height(), width, height, algo).run(&executor, img))
}

pub struct GpuExecutor {
    queue: wgpu::Queue,
    device: wgpu::Device
}

pub struct Resizer {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    input_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    output_texture: wgpu::Texture,
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
}

pub struct CachedResizer {
    resizer: Option<Resizer>,
    gpu: GpuExecutor,
}

impl CachedResizer {
    pub fn new() -> Result<Self> {
        Ok(Self { resizer: None, gpu: GpuExecutor::new()? })
    }

    fn new_resizer(&mut self, img: &RgbImage, width: u32, height: u32, algo: ResizeAlgo) -> Resizer {
        Resizer::new(
            &self.gpu,
            img.width(),
            img.height(),
            width,
            height,
            algo
        )
    }

    pub fn run(&mut self, img: &RgbImage, width: u32, height: u32, algo: ResizeAlgo) -> RgbImage {
        match &mut self.resizer {
            Some(resizer) => {
                if resizer.input_width != img.width() || resizer.input_height != img.height() {                    // Dimensions changed -- need a new resizer
                    let mut resizer = self.new_resizer(img, width, height, algo);
                    let result = resizer.run(&self.gpu, img);
                    self.resizer = Some(resizer);
                    result
                } else {
                    resizer.run(&self.gpu, img)
                }
            }
            None => {
                let mut resizer = self.new_resizer(img, width, height, algo);
                let result = resizer.run(&self.gpu, img);
                self.resizer = Some(resizer);
                result
            }
        }
        
    }
}

fn calc_image_buffer_size(width: u32, height: u32) -> (u32, u32) {
    let input_size = width * height * 3;
    let buffer_alignment = wgpu::COPY_BUFFER_ALIGNMENT as u32;
    let input_pad = buffer_alignment - input_size % buffer_alignment;

    (input_size, input_pad)
}

impl Resizer {
    pub fn new(executor: &GpuExecutor, input_width: u32, input_height: u32, output_width: u32, output_height: u32, algo: ResizeAlgo) -> Self {
        let span = span!(Level::INFO, "Resizer#new");
        let _guard = span.enter();

        let (input_size, input_pad) = calc_image_buffer_size(input_width, input_height);

        let input_buffer = executor.device.create_buffer(&wgpu::BufferDescriptor {
			      label: Some("input_buf"),
			      size: (input_size + input_pad).into(),
			      usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
			      mapped_at_creation: false
		    });

		    let width_uniform = executor.device.create_buffer(&wgpu::BufferDescriptor {
			      label: Some("width_uniform"),
			      size: 4,
			      usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
			      mapped_at_creation: false
		    });

		    let height_uniform = executor.device.create_buffer(&wgpu::BufferDescriptor {
			      label: Some("height_uniform"),
			      size: 4,
			      usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
			      mapped_at_creation: false
		    });

        let output_texture_size = wgpu::Extent3d {
            width: output_width,
            height: output_height,
            depth_or_array_layers: 1,
        };

        let output_texture = executor.device.create_texture(
            &wgpu::TextureDescriptor {
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

        let output_buffer_size =
            padded_bytes_per_row(output_width) as u64 * output_height as u64 * std::mem::size_of::<u8>() as u64;
        let output_buffer = executor.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output_buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let shader = executor.device.create_shader_module(wgpu::include_wgsl!("resize.wgsl"));

        let pipeline = executor.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
			      label: Some("pipeline"),
			      layout: None,
			      module: &shader,
			      entry_point: algo.shader_name().into(),
			      compilation_options: wgpu::PipelineCompilationOptions::default(),
			      cache: None
		    });

        let bind_group = executor.device.create_bind_group(&wgpu::BindGroupDescriptor {
			      label: Some("bind_group"),
			      layout: &pipeline.get_bind_group_layout(0),
			      entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
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

        executor.queue.write_buffer(&width_uniform, 0, &input_width.to_ne_bytes());
        executor.queue.write_buffer(&height_uniform, 0, &input_height.to_ne_bytes());
        
        Self {
            pipeline,
            bind_group,
            input_buffer,
            output_buffer,
            output_texture,
            input_width,
            input_height,
            output_width,
            output_height,
        }
    }

    pub fn run(&mut self, executor: &GpuExecutor, img: &RgbImage) -> RgbImage {
        let span = span!(Level::INFO, "Resizer#run");
        let _guard = span.enter();

        let (input_size, _) = calc_image_buffer_size(self.input_width, self.input_height);
        let mut view = executor.queue.write_buffer_with(&self.input_buffer, 0, NonZero::new(self.input_buffer.size()).unwrap()).unwrap();
        view[..input_size as usize].copy_from_slice(img.as_raw());
        drop(view);
        
        let mut encoder = executor.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
			      label: Some("encoder")
		    });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute"),
            timestamp_writes: None
        });

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.bind_group, &[]);
        compute_pass.dispatch_workgroups(
            int_div_round_up(self.output_width, 8),
            int_div_round_up(self.output_height, 8),
            1
        );
        drop(compute_pass);

        let padded_bytes_per_row = padded_bytes_per_row(self.output_width);
        let unpadded_bytes_per_row = self.output_width as usize * 4;

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                aspect: wgpu::TextureAspect::All,
                texture: &self.output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::TexelCopyBufferInfo{
                buffer: &self.output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: (padded_bytes_per_row as u32).into(),
                    rows_per_image: self.output_height.into(),
                },
            },
            self.output_texture.size(),
        );

        executor.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = self.output_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
        
        executor.device.poll(wgpu::PollType::Wait).unwrap();

        let padded_data = buffer_slice.get_mapped_range();
        let mut pixels: Vec<u8> = vec![0; unpadded_bytes_per_row * self.output_height as usize];
        for (padded, pixels) in padded_data
            .chunks_exact(padded_bytes_per_row)
            .zip(pixels.chunks_exact_mut(unpadded_bytes_per_row)) {
                pixels.copy_from_slice(&padded[..unpadded_bytes_per_row]);
            }
        drop(padded_data);
        self.output_buffer.unmap();

        let with_alpha = RgbaImage::from_raw(self.output_width, self.output_height, pixels).unwrap();
        DynamicImage::ImageRgba8(with_alpha).to_rgb8()
    }
}

impl GpuExecutor {
    async fn init() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
			      backends: wgpu::Backends::all(),
			      flags: wgpu::InstanceFlags::VALIDATION,
			      backend_options: wgpu::BackendOptions::default()
		    });
    
		    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await?;

		    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
			      required_features: wgpu::Features::empty(),
			      required_limits: wgpu::Limits::default(),
			      memory_hints: wgpu::MemoryHints::Performance,
			      label: Some("device"),
			      trace: wgpu::Trace::Off
		    }).await?;

        Ok(Self { device, queue })
    }

    pub fn new() -> Result<Self> {
        let span = span!(Level::INFO, "GpuExecutor#new");
        let _guard = span.enter();
        Self::init().block_on()
    }
}

fn int_div_round_up(divisor: u32, dividend: u32) -> u32 {
	(divisor / dividend) + match divisor % dividend {
		0 => 0,
		_ => 1
	}
}

fn padded_bytes_per_row(width: u32) -> usize {
    let bytes_per_row = width as usize * 4;
    let padding = (256 - bytes_per_row % 256) % 256;
    bytes_per_row + padding
}
