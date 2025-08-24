use super::util::padded_bytes_per_row;
use anyhow::Result;
use image::{DynamicImage, RgbaImage};
#[cfg(not(target_arch = "wasm32"))]
use pollster::FutureExt;
use std::collections::HashMap;
use tracing::{Level, span};
use wgpu::ShaderModuleDescriptor;
#[cfg(target_arch = "wasm32")]
use wgpu::{Surface, SurfaceConfiguration};

pub struct GpuExecutor {
    pub queue: wgpu::Queue,
    pub device: wgpu::Device,
    shaders: HashMap<String, wgpu::ShaderModule>,
}

impl GpuExecutor {
    #[cfg(not(target_arch = "wasm32"))]
    async fn init() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::VALIDATION,
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: Default::default(),
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                label: Some("device"),
                trace: Default::default(),
            })
            .await?;

        Ok(Self {
            device,
            queue,
            shaders: HashMap::new(),
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Result<Self> {
        let span = span!(Level::DEBUG, "GpuExecutor#new");
        let _guard = span.enter();
        Self::init().block_on()
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn new_wasm(
        canvas: web_sys::HtmlCanvasElement,
    ) -> Result<(Self, Surface<'static>, SurfaceConfiguration)> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            flags: wgpu::InstanceFlags::VALIDATION,
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: Default::default(),
        });

        let width = canvas.width();
        let height = canvas.height();

        let surface = instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                label: Some("device"),
                trace: Default::default(),
            })
            .await
            .expect("Unable to find a suitable GPU adapter!");

        let config = wgpu::SurfaceConfiguration {
            format: wgpu::TextureFormat::Rgba8Unorm,
            view_formats: vec![wgpu::TextureFormat::Rgba8Unorm],
            usage: wgpu::TextureUsages::COPY_DST,
            width,
            height,
            present_mode: Default::default(),
            alpha_mode: Default::default(),
            desired_maximum_frame_latency: 2,
        };

        Ok((
            Self {
                device,
                queue,
                shaders: HashMap::new(),
            },
            surface,
            config,
        ))
    }

    pub fn load_shader(&mut self, name: &str, desc: ShaderModuleDescriptor) -> wgpu::ShaderModule {
        if !self.shaders.contains_key(name) {
            let shader_mod = self.device.create_shader_module(desc);
            self.shaders.insert(name.to_string(), shader_mod);
        }

        self.shaders.get(name).unwrap().clone()
    }

    #[allow(unused)]
    pub fn snapshot_texture(&self, tex: &wgpu::Texture, fname: &str) -> Result<()> {
        let width = tex.width();
        let height = tex.height();
        let buffer_size =
            padded_bytes_per_row(width) as u64 * height as u64 * std::mem::size_of::<u8>() as u64;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("snapshot_buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encoder"),
            });

        let padded_bytes_per_row = padded_bytes_per_row(width);
        let unpadded_bytes_per_row = width as usize * 4;

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                aspect: wgpu::TextureAspect::All,
                texture: tex,
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
            tex.size(),
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());

        self.device.poll(wgpu::PollType::Wait)?;

        let padded_data = buffer_slice.get_mapped_range();
        let mut pixels: Vec<u8> = vec![0; unpadded_bytes_per_row * height as usize];
        for (padded, pixels) in padded_data
            .chunks_exact(padded_bytes_per_row)
            .zip(pixels.chunks_exact_mut(unpadded_bytes_per_row))
        {
            pixels.copy_from_slice(&padded[..unpadded_bytes_per_row]);
        }
        drop(padded_data);
        buffer.unmap();

        let img = RgbaImage::from_raw(width, height, pixels).unwrap();
        DynamicImage::ImageRgba8(img).to_rgb8().save(fname)?;
        Ok(())
    }

    pub fn rgba_buffer_to_texture(
        &self,
        rgba_bytes: &[u8],
        width: u32,
        height: u32,
    ) -> wgpu::Texture {
        let span = span!(Level::DEBUG, "rgba_buffer_to_texture");
        let _guard = span.enter();

        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rgba_sourced_texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
        });

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba_bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            texture_size,
        );

        texture
    }
}
