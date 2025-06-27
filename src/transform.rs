use crate::imggpu::gpu::GpuExecutor;
use crate::imggpu::vertex::Vertex;
use crate::shapes::shape::Shape;
use anyhow::Result;
use tracing::{debug, info, span, Level};
use wgpu::{util::DeviceExt, ShaderStages};

#[derive(Debug, Clone, Copy)]
pub enum FlipVariant {
    Vertical,
    Horizontal,
    Both,
}

#[derive(Debug, Clone)]
pub struct Transform {
    shape: Shape,
    copy_dest: Option<Shape>,
    swap: Option<Shape>,
    rotate_deg: Option<f32>,
    flip: Option<FlipVariant>,
    scale: f32,
    tile: bool,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            shape: Default::default(),
            copy_dest: None,
            swap: None,
            rotate_deg: None,
            flip: None,
            scale: 1.,
            tile: false,
        }
    }
}

impl Transform {
    pub fn new(shape: impl Into<Shape>) -> Self {
        Self {
            shape: shape.into(),
            ..Default::default()
        }
    }

    pub fn set_flip(&mut self, f: FlipVariant) {
        self.flip = Some(f);
    }

    pub fn copy_to(&mut self, s: Shape) {
        self.copy_dest = Some(s);
    }

    pub fn swap_with(&mut self, s: Shape) {
        self.swap = Some(s);
    }

    pub fn set_scale(&mut self, s: f32) {
        self.scale = s;
    }

    pub fn set_tiling(&mut self, t: bool) {
        self.tile = t;
    }

    pub fn set_rot_degrees(&mut self, deg: f32) {
        self.rotate_deg = Some(deg);
    }

    pub fn execute(&self, gpu: &mut GpuExecutor, tex: &wgpu::Texture) -> Result<wgpu::Texture> {
        let span = span!(Level::INFO, "Transform#execute");
        let _guard = span.enter();

        let sampler = self.sampler(gpu);

        let shader_code = wgpu::include_wgsl!("transform.wgsl");
        let shader = gpu.load_shader("transform", shader_code);

        let bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("render bind group layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: Default::default(),
                                view_dimension: Default::default(),
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("render_pipeline"),
                layout: Some(&pipeline_layout),
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

        let rot = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rot"),
            size: 12,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(
            &rot,
            0,
            &bytemuck::cast_slice(&match self.rotate_deg {
                Some(d) => [1., d.to_radians().cos(), d.to_radians().sin()],
                None => [0., 0., 0.],
            }),
        );

        let render_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render_bind_group2"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &tex.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: rot.as_entire_binding(),
                },
            ],
        });

        // TODO: can we just overwrite input tex for perf?
        let output_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: tex.width(),
                height: tex.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING,
        });

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encoder"),
            });

        let vertices = self.vertices(tex.width(), tex.height());

        let vertex_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vertex_buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

        // TODO: uncomment me once working
        // encoder.copy_texture_to_texture(
        //     wgpu::TexelCopyTextureInfo {
        //         texture: &tex,
        //         mip_level: Default::default(),
        //         origin: Default::default(),
        //         aspect: Default::default(),
        //     },
        //     wgpu::TexelCopyTextureInfo {
        //         texture: &output_tex,
        //         mip_level: Default::default(),
        //         origin: Default::default(),
        //         aspect: Default::default(),
        //     },
        //     wgpu::Extent3d {
        //         width: tex.width(),
        //         height: tex.height(),
        //         depth_or_array_layers: 1,
        //     },
        // );

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &output_tex.create_view(&Default::default()),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load, // preserve underlying image
                    store: wgpu::StoreOp::Store,
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
        Ok(output_tex)
    }

    pub fn vertices(&self, width: u32, height: u32) -> Vec<Vertex> {
        let make_vtx = |x: u32, y: u32| {
            let x = x as f32 / width as f32;
            let y = y as f32 / height as f32;
            // FIXME: scale from center instead of TL
            let clip_x = self.scale * x * 2. - 1.;
            // cast y val to clip space, including inverting axis
            let clip_y = 1. - y * 2. * self.scale;

            // FIXME: handle polygon for mouth, others
            // TODO: tile?
            // TODO: swap -- vertices clip src -> tex dest for each direction
            // TODO: copy -- vertices clip src -> tex dest
            // TODO: if flip -- flip x/y tex coords appropriately
            // tex coords are points we are reading _from_
            // vertex coords are clip-spaced of where we are writing _to_
            // output, input
            // FIXME whytf is x/y always 0
            Vertex::new_with_tex(&[clip_x, clip_y], &[x, y])
        };

        let shape = self.shape.clone();
        let vertices = match shape {
            Shape::Polygon(p) => p
                .points
                .iter()
                .map(|p| make_vtx(p.x, p.y))
                .collect::<Vec<_>>(),
            Shape::Rect(sr) => {
                let r = sr.right();
                let l = sr.left();
                let t = sr.top();
                let b = sr.bottom();
                Vec::from([
                    make_vtx(r, t),
                    make_vtx(l, t),
                    make_vtx(l, b),
                    make_vtx(r, b),
                ])
            }
        };

        Vertex::to_triangles(vertices)
    }

    fn sampler(&self, gpu: &GpuExecutor) -> wgpu::Sampler {
        let address_mode = if self.tile {
            wgpu::AddressMode::Repeat
        } else {
            wgpu::AddressMode::ClampToEdge
        };

        gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: address_mode,
            address_mode_v: address_mode,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        })
    }
}
