pub use super::gpu::GpuExecutor;
use anyhow::Result;
use image::RgbImage;
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

pub fn resize_with_executor(
    executor: &GpuExecutor,
    img: &RgbImage,
    width: u32,
    height: u32,
    algo: ResizeAlgo,
) -> Result<RgbImage> {
    let span = span!(Level::INFO, "resize_with_executor");
    let _guard = span.enter();
    Ok(Resizer::new(&executor, img.width(), img.height(), width, height, algo).run(&executor, img))
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
        Ok(Self {
            resizer: None,
            gpu: GpuExecutor::new()?,
        })
    }

    fn new_resizer(
        &mut self,
        img: &RgbImage,
        width: u32,
        height: u32,
        algo: ResizeAlgo,
    ) -> Resizer {
        Resizer::new(&self.gpu, img.width(), img.height(), width, height, algo)
    }

    pub fn run(&mut self, img: &RgbImage, width: u32, height: u32, algo: ResizeAlgo) -> RgbImage {
        match &mut self.resizer {
            Some(resizer) => {
                if resizer.input_width != img.width() || resizer.input_height != img.height() {
                    // Dimensions changed -- need a new resizer
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

impl Resizer {
    pub fn new(
        executor: &GpuExecutor,
        input_width: u32,
        input_height: u32,
        output_width: u32,
        output_height: u32,
        algo: ResizeAlgo,
    ) -> Self {
        let span = span!(Level::INFO, "Resizer#new");
        let _guard = span.enter();

        let input_buffer = executor.create_input_image_buffer(input_width, input_height);

        let width_uniform = executor.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("width_uniform"),
            size: 4,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let height_uniform = executor.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("height_uniform"),
            size: 4,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (output_texture, output_buffer) =
            executor.create_output_texture_pair(output_width, output_height);

        // TODO: cache this up a level?
        let shader = executor
            .device
            .create_shader_module(wgpu::include_wgsl!("resize.wgsl"));

        let pipeline = executor
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("pipeline"),
                layout: None,
                module: &shader,
                entry_point: algo.shader_name().into(),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let bind_group = executor
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bind_group"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &output_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: width_uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: height_uniform.as_entire_binding(),
                    },
                ],
            });

        executor
            .queue
            .write_buffer(&width_uniform, 0, &input_width.to_ne_bytes());
        executor
            .queue
            .write_buffer(&height_uniform, 0, &input_height.to_ne_bytes());

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

        executor.load_image(img, &mut self.input_buffer);

        executor.execute(
            &self.pipeline,
            &self.bind_group,
            &self.output_texture,
            &mut self.output_buffer,
            self.output_width,
            self.output_height,
        )
    }
}
