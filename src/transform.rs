use std::time::Instant;

use crate::imggpu::vertex::Vertex;
use crate::shapes::point::Point;
use crate::shapes::shape::Shape;
use crate::{imggpu::gpu::GpuExecutor, shapes::rect::Rect};
use anyhow::Result;
use tracing::{span, trace, warn, Level};
use wgpu::util::DeviceExt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlipVariant {
    Vertical,
    Horizontal,
    Both,
}

#[derive(Debug, Clone)]
pub struct Transform {
    pub id: String,
    shape: Shape,
    copy_dests: Vec<Shape>,
    swap: Option<Shape>,
    initial_rotate_deg: Option<f32>,
    rotate_deg: Option<f32>,
    flip: Option<FlipVariant>,
    initial_translation: Option<(i32, i32)>,
    translation: Option<(i32, i32)>,
    scale: f32,
    tile: bool,
    rps: Option<f32>,
    last_tick: Option<Instant>,
    drift_vec: Option<(f32, f32)>,
    initial_drift_vec: Option<(f32, f32)>,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            id: "".to_string(),
            shape: Default::default(),
            copy_dests: Vec::new(),
            swap: None,
            initial_rotate_deg: None,
            rotate_deg: None,
            flip: None,
            initial_translation: None,
            translation: None,
            scale: 1.,
            tile: false,
            rps: None,
            last_tick: None,
            drift_vec: None,
            initial_drift_vec: None,
        }
    }
}

const TILE_HEIGHT: u32 = 100;
const TILE_WIDTH: u32 = 160;

impl Transform {
    pub fn new(s: impl Into<Shape>, id: String) -> Self {
        Self {
            id,
            shape: s.into(),
            ..Default::default()
        }
    }

    pub fn set_shape(&mut self, s: impl Into<Shape>) {
        self.shape = s.into();
    }

    pub fn set_flip(&mut self, f: FlipVariant) {
        self.flip = Some(f);
    }

    pub fn copy_to(&mut self, dests: impl Into<Vec<Shape>>) {
        // Apply transforms to self shape and dests
        self.copy_dests = dests.into();
        self.copy_dests.push(self.shape.clone());
    }

    pub fn write_to(&mut self, dests: impl Into<Vec<Shape>>) {
        // Apply transforms ONLY to dests
        self.copy_dests = dests.into();
    }

    pub fn swap_with(&mut self, s: Shape) {
        self.swap = Some(s);
    }

    pub fn set_scale(&mut self, s: f32) {
        self.scale = s;

        if self.tile {
            warn!("Scale with tile not currently supported. Skipping scale operation.");
        }
    }

    pub fn set_tiling(&mut self, t: bool) {
        self.tile = t;

        if self.tile {
            if self.scale != 1. {
                warn!("Scale with tile not currently supported. Skipping scale operation.");
            }

            if self.rotate_deg.is_some() {
                warn!("Rotate with tile not currently supported. Skipping rotate operation.");
            }

            if self.translation.is_some() {
                warn!(
                    "Translation with tile not currently supported. Skipping translate operation."
                );
            }

            if self.translation.is_some() {
                warn!(
                    "Translation with tile not currently supported. Skipping translate operation."
                );
            }
        }
    }

    pub fn set_rot_degrees(&mut self, deg: f32) {
        if self.initial_rotate_deg.is_some() && self.initial_rotate_deg.unwrap() == deg {
            trace!("Rotate already set.");
        } else {
            self.initial_rotate_deg = Some(deg);
            self.rotate_deg = Some(deg);
        }

        if self.tile {
            warn!("Rotate with tile not currently supported. Skipping rotate operation.");
        }
    }

    // rps: rotations per second. 0. = stationary, 0.5 = 180deg/s, -0.5 = -180deg/s
    pub fn set_spin(&mut self, rps: f32) {
        self.rps = Some(rps);
        self.set_rot_degrees(0.); // initialize rotation
    }

    pub fn translate_by(&mut self, x: i32, y: i32) {
        let trans = (x, y);
        if self.initial_translation.is_some() && self.initial_translation.unwrap() == trans {
            trace!("Translation already set.");
        } else {
            self.initial_translation = Some(trans);
            self.translation = Some(trans);
        }

        if self.tile {
            warn!("Translation with tile not currently supported. Skipping translate operation.");
        }
    }

    // velocity: pixels/s of travel
    // angle: clockwise degrees of initial vector
    pub fn set_drift(&mut self, velocity: f32, angle: f32) {
        let drift_vec = (velocity, angle);

        if self.initial_drift_vec.is_some() && self.initial_drift_vec.unwrap() == drift_vec {
            trace!("Drift already set.");
        } else {
            self.initial_drift_vec = Some(drift_vec);
            self.drift_vec = Some(drift_vec);
        }

        self.translate_by(0, 0); // initalize translation
    }

    pub fn execute(&mut self, gpu: &mut GpuExecutor, tex: &wgpu::Texture) -> Result<wgpu::Texture> {
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
            ],
        });

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

        self.tick(tex);
        let vertices = self.vertices(tex);

        let vertex_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vertex_buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: Default::default(),
                origin: Default::default(),
                aspect: Default::default(),
            },
            wgpu::TexelCopyTextureInfo {
                texture: &output_tex,
                mip_level: Default::default(),
                origin: Default::default(),
                aspect: Default::default(),
            },
            wgpu::Extent3d {
                width: tex.width(),
                height: tex.height(),
                depth_or_array_layers: 1,
            },
        );

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

    pub fn vertices(&self, tex: &wgpu::Texture) -> Vec<Vertex> {
        if self.tile {
            return self.tiled_vertices(tex);
        }

        let mut vertex_groups = Vec::new();

        for ds in &self.copy_dests {
            vertex_groups.push(self.vertices_for_shapes(tex, &self.shape, ds));
        }

        if self.swap.is_some() {
            let swap = self.swap.as_ref().unwrap().clone();
            vertex_groups.push(self.vertices_for_shapes(tex, &self.shape, &swap));
            vertex_groups.push(self.vertices_for_shapes(tex, &swap, &self.shape));
        }

        if vertex_groups.len() == 0 {
            vertex_groups.push(self.vertices_for_shapes(tex, &self.shape, &self.shape));
        }

        vertex_groups.concat()
    }

    fn tick(&mut self, tex: &wgpu::Texture) {
        // animate spin and drift since last iteration
        match self.last_tick {
            Some(last) => {
                if self.rps.is_some() {
                    let rps = self.rps.unwrap();
                    let old_rot = self.rotate_deg.unwrap();
                    self.rotate_deg = Some(old_rot + 360. * rps * last.elapsed().as_secs_f32());
                }

                if self.drift_vec.is_some() {
                    let (vel, ang) = self.drift_vec.unwrap();
                    let hyp = vel * last.elapsed().as_secs_f32();
                    let dy = (ang.to_radians().cos() * hyp).round() as i32;
                    let dx = (ang.to_radians().sin() * hyp).round() as i32;
                    let (tx, ty) = self.translation.unwrap();

                    let center = self.shape.center();
                    let center_x = center.x as i32;
                    let center_y = center.y as i32;
                    let cur_x = center_x + tx;
                    let cur_y = center_y + ty;

                    let mut next_x = cur_x + dx;
                    let mut next_y = cur_y + dy;

                    let mut next_ang = ang;
                    let width = tex.width() as i32;
                    let height = tex.height() as i32;

                    if next_x >= width {
                        next_x -= next_x - width;
                        next_ang = mirror_x(next_ang);
                    }

                    if next_x < 0 {
                        next_x *= -1;
                        next_ang = mirror_x(next_ang);
                    }

                    if next_y >= height {
                        next_y -= next_y - height;
                        next_ang = mirror_y(next_ang);
                    }

                    if next_y < 0 {
                        next_y *= -1;
                        next_ang = mirror_y(next_ang);
                    }

                    self.drift_vec = Some((vel, next_ang));
                    self.translation = Some((next_x - center_x, next_y - center_y));
                }
            }
            None => (),
        }

        self.last_tick = Some(Instant::now());
    }

    fn vertices_for_shapes(&self, tex: &wgpu::Texture, src: &Shape, dest: &Shape) -> Vec<Vertex> {
        let width = tex.width() as f32;
        let height = tex.height() as f32;
        let make_vtx = |(src, dest): (Point, Point)| -> Vertex {
            // tex coords are points we are reading _from_
            // vertex coords are clip-spaced of where we are writing _to_
            let x = dest.x as f32 / width;
            let y = dest.y as f32 / height;
            let clip_x = x * 2. - 1.;
            // cast y val to clip space, including inverting axis
            let clip_y = 1. - y * 2.;

            Vertex::new_with_tex(
                &[clip_x, clip_y],
                &[src.x as f32 / width, src.y as f32 / height],
            )
        };
        let mut vertices = src
            .iter_projection_onto(dest.clone())
            .map(make_vtx)
            .collect::<Vec<_>>();
        vertices = self.scale_rotate_flip(&mut vertices, tex.width(), tex.height());

        Vertex::to_triangles(vertices)
    }

    fn tiled_vertices(&self, tex: &wgpu::Texture) -> Vec<Vertex> {
        let width = tex.width();
        let height = tex.height();
        let tex_rect = Rect::from(self.shape.clone());
        let tr = tex_rect.right() as f32 / width as f32;
        let tl = tex_rect.left() as f32 / width as f32;
        let tt = tex_rect.top() as f32 / height as f32;
        let tb = tex_rect.bottom() as f32 / height as f32;
        let tex_tr = [tr, tt];
        let tex_tl = [tl, tt];
        let tex_bl = [tl, tb];
        let tex_br = [tr, tb];

        let mut rects = Vec::new();
        for ry in 0..height.div_ceil(TILE_HEIGHT) {
            for rx in 0..width.div_ceil(TILE_WIDTH) {
                let l = ((rx * TILE_WIDTH) as f32 / width as f32) * 2. - 1.;
                let r = ((rx + 1) * TILE_WIDTH).min(width) as f32 / width as f32 * 2. - 1.;
                let t = 1. - (ry * TILE_HEIGHT) as f32 / height as f32 * 2.;
                let b = 1. - ((ry + 1) * TILE_HEIGHT).min(height) as f32 / height as f32 * 2.;

                let mut vertices = Vec::from([
                    Vertex::new_with_tex(&[r, t], &tex_tr),
                    Vertex::new_with_tex(&[l, t], &tex_tl),
                    Vertex::new_with_tex(&[l, b], &tex_bl),
                    Vertex::new_with_tex(&[r, b], &tex_br),
                ]);

                vertices = self.scale_rotate_flip(&mut vertices, width, height);
                rects.push(Vertex::to_triangles(vertices));
            }
        }

        rects.concat()
    }

    // FIXME: translate + rotate causes rotation about original center, not translated center
    fn scale_rotate_flip(
        &self,
        vertices: &mut Vec<Vertex>,
        width: u32,
        height: u32,
    ) -> Vec<Vertex> {
        let mut l = f32::MAX;
        let mut r = f32::MIN;
        let mut t = f32::MAX;
        let mut b = f32::MIN;
        for v in &*vertices {
            let x = v.x();
            let y = v.y();
            if x < l {
                l = x;
            }
            if y < t {
                t = y;
            }
            if x > r {
                r = x;
            }
            if y > b {
                b = y;
            }
        }
        let clip_center = Vertex::new(&[l + (r - l) / 2., t + (b - t) / 2.]);

        // Texture bounds (for flip)
        let mut l = f32::MAX;
        let mut r = f32::MIN;
        let mut t = f32::MAX;
        let mut b = f32::MIN;
        for v in &*vertices {
            let x = v.tex_coord[0];
            let y = v.tex_coord[1];
            if x < l {
                l = x;
            }
            if y < t {
                t = y;
            }
            if x > r {
                r = x;
            }
            if y > b {
                b = y;
            }
        }

        let trans = Vertex::new(&match self.translation {
            None => [0., 0.],
            Some(t) => [t.0 as f32 / width as f32, -1. * t.1 as f32 / height as f32],
        });

        vertices
            .iter_mut()
            .map(|v| {
                self.transform_vertex(v, &clip_center, l, r, t, b, &trans);
                *v
            })
            .collect::<Vec<_>>()
    }

    fn transform_vertex(
        &self,
        v: &mut Vertex,
        c: &Vertex,
        l: f32,
        r: f32,
        t: f32,
        b: f32,
        trans: &Vertex,
    ) {
        if self.flip.is_some() {
            let flip_variant = self.flip.unwrap();

            if flip_variant == FlipVariant::Both || flip_variant == FlipVariant::Horizontal {
                v.tex_coord[0] = flip(v.tex_coord[0], l, r);
            }

            if flip_variant == FlipVariant::Both || flip_variant == FlipVariant::Vertical {
                v.tex_coord[1] = flip(v.tex_coord[1], t, b);
            }
        }

        // TODO: scale, rotate, translate support for tiling
        if !self.tile {
            if self.translation.is_some() {
                v.add(&trans);
            }

            if self.scale != 1. {
                v.sub(&c);
                v.mult_pos(self.scale);
                v.add(&c);
            }

            if self.rotate_deg.is_some() {
                let rad = self.rotate_deg.unwrap().to_radians();
                let cos = rad.cos();
                let sin = rad.sin();

                let old_x = v.position[0];
                let old_y = v.position[1];
                let trans_x = old_x - c.position[0];
                let trans_y = old_y - c.position[1];
                v.sub(c);
                v.position = [trans_x * cos - trans_y * sin, trans_x * sin + trans_y * cos];
                v.add(c);
            }
        }
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

fn flip(val: f32, min: f32, max: f32) -> f32 {
    // Invert val within range
    let res = min + max - val;
    res.min(max).max(min)
}

fn mirror_x(degrees: f32) -> f32 {
    360. - degrees
}

fn mirror_y(degrees: f32) -> f32 {
    if degrees >= 180. {
        540. - degrees
    } else {
        180. - degrees
    }
}
