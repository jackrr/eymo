use crate::imggpu::vertex::Vertex;
use crate::shapes::point::Point;
use crate::shapes::shape::Shape;
use crate::{imggpu::gpu::GpuExecutor, shapes::rect::Rect};
use std::collections::HashMap;
use tracing::{Level, span, warn};
use web_time::Instant;
use wgpu::util::DeviceExt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlipVariant {
    Vertical,
    Horizontal,
    Both,
}

#[derive(Debug)]
pub struct Transform {
    rotate_deg: Option<f32>,
    flip: Option<FlipVariant>,
    translation: Option<(i32, i32)>,
    scale: f32,
    tile: bool,
    rps: Option<f32>,
    last_tick: Instant,
    drift_vec: Option<(f32, f32)>,
    brightness_mod: f32,
    saturation_mod: f32,
    chans_mod: [f32; 4],
    cache: HashMap<String, ShapeOpState>,
    gpu_gunk: GpuGunk,
}

#[derive(Debug)]
pub struct ShapeOp {
    id: String,
    base: Shape,
    swap: Option<Shape>,
    dest: Option<Shape>,
}

#[derive(Debug)]
struct GpuGunk {
    bg_layout: wgpu::BindGroupLayout,
    render_pipeline: wgpu::RenderPipeline,
}

#[derive(Debug, Clone, Default)]
struct ShapeOpState {
    translation: Option<(i32, i32)>,
    drift_vec: Option<(f32, f32)>,
    rotate_deg: Option<f32>,
}

impl ShapeOp {
    pub fn swap(id: String, a: impl Into<Shape>, b: impl Into<Shape>) -> Self {
        Self {
            id,
            base: a.into(),
            swap: Some(b.into()),
            dest: None,
        }
    }

    pub fn copy(id: String, src: impl Into<Shape>, dest: impl Into<Shape>) -> Self {
        Self {
            id,
            base: src.into(),
            dest: Some(dest.into()),
            swap: None,
        }
    }

    pub fn on_shape(id: String, s: impl Into<Shape>) -> Self {
        Self {
            id,
            base: s.into(),
            dest: None,
            swap: None,
        }
    }
}

const TILE_HEIGHT: u32 = 100;
const TILE_WIDTH: u32 = 160;

impl Transform {
    pub fn new(gpu: &mut GpuExecutor) -> Self {
        Self {
            flip: None,
            scale: 1.,
            tile: false,
            brightness_mod: -1.,
            saturation_mod: -1.,
            chans_mod: [-1., -1., -1., -1.],
            last_tick: Instant::now(),
            rotate_deg: None,
            rps: None,
            translation: None,
            drift_vec: None,
            cache: HashMap::new(),
            gpu_gunk: GpuGunk::init(gpu),
        }
    }

    pub fn set_brightness(&mut self, b: f32) {
        self.brightness_mod = b;
    }

    pub fn set_saturation(&mut self, s: f32) {
        self.saturation_mod = s;
    }

    pub fn set_chans(&mut self, r: f32, g: f32, b: f32) {
        self.chans_mod = [r, g, b, 1.];
    }

    pub fn set_flip(&mut self, f: FlipVariant) {
        self.flip = Some(f);
    }

    pub fn set_scale(&mut self, s: f32) {
        self.scale = s;

        if self.tile {
            warn!("Scale with tile not currently supported. Skipping scale operation.");
        }
    }

    pub fn set_tiling(&mut self, t: bool) {
        self.tile = t;
    }

    pub fn set_rot_degrees(&mut self, deg: f32) {
        self.rotate_deg = Some(deg);
    }

    // rps: rotations per second. 0. = stationary, 0.5 = 180deg/s, -0.5 = -180deg/s
    pub fn set_spin(&mut self, rps: f32) {
        self.rps = Some(rps);
        self.set_rot_degrees(0.); // initialize rotation
    }

    pub fn translate_by(&mut self, x: i32, y: i32) {
        self.translation = Some((x, y));
    }

    // velocity: pixels/s of travel
    // angle: clockwise degrees of initial vector
    pub fn set_drift(&mut self, velocity: f32, angle: f32) {
        self.drift_vec = Some((velocity, angle));
        self.translate_by(0, 0); // initalize translation
    }

    pub fn execute(
        &mut self,
        gpu: &mut GpuExecutor,
        tex: &wgpu::Texture,
        shape_ops: Vec<ShapeOp>,
    ) -> wgpu::Texture {
        let span = span!(Level::DEBUG, "Transform#execute");
        let _guard = span.enter();

        let mut vertices = Vec::new();
        for op in shape_ops.into_iter() {
            let prev_val = self.cache.remove(&op.id);
            let next_cache_val = self.tick(&op.base, tex, prev_val);
            vertices.push(self.gen_vertices(tex, &op, &next_cache_val));
            self.cache.insert(op.id.clone(), next_cache_val);
        }

        let sampler = self.sampler(gpu);
        self.gpu_gunk.execute(
            gpu,
            vertices.concat(),
            tex,
            &[self.brightness_mod, self.saturation_mod],
            &self.chans_mod,
            sampler,
        )
    }

    fn tick(&self, shape: &Shape, tex: &wgpu::Texture, prev: Option<ShapeOpState>) -> ShapeOpState {
        // animate spin and drift since last iteration
        let mut next_state = ShapeOpState::default();
        let last = self.last_tick;

        let defaults = (
            self.rotate_deg.unwrap_or(0.),
            self.drift_vec.unwrap_or((0., 0.)),
            self.translation.unwrap_or((0, 0)),
        );

        let (rotate_deg, (vel, ang), (tx, ty)) = match prev {
            Some(prev) => (
                prev.rotate_deg.unwrap_or(defaults.0),
                prev.drift_vec.unwrap_or(defaults.1),
                prev.translation.unwrap_or(defaults.2),
            ),
            None => defaults,
        };

        if self.rps.is_some() {
            let rps = self.rps.unwrap();
            next_state.rotate_deg = Some(rotate_deg + 360. * rps * last.elapsed().as_secs_f32());
        } else if self.rotate_deg.is_some() {
            next_state.rotate_deg = self.rotate_deg.clone();
        }

        if self.drift_vec.is_some() {
            let hyp = vel * last.elapsed().as_secs_f32();
            let dy = (ang.to_radians().cos() * hyp).round() as i32;
            let dx = (ang.to_radians().sin() * hyp).round() as i32;

            let center = shape.center();
            let center_x = center.x as i32;
            let center_y = center.y as i32;

            let mut next_x = center_x + tx + dx;
            let mut next_y = center_y + ty + dy;

            let mut next_ang = ang;
            let width = tex.width() as i32;
            let height = tex.height() as i32;

            if next_x >= width {
                next_x = width - (next_x - width);
                next_ang = mirror_x(next_ang);
            }

            if next_x < 0 {
                next_x *= -1;
                next_ang = mirror_x(next_ang);
            }

            if next_y >= height {
                next_y = height - (next_y - height);
                next_ang = mirror_y(next_ang);
            }

            if next_y < 0 {
                next_y *= -1;
                next_ang = mirror_y(next_ang);
            }

            next_state.drift_vec = Some((vel, next_ang));
            next_state.translation = Some((next_x - center_x, next_y - center_y));
        } else if self.translation.is_some() {
            next_state.translation = self.translation.clone();
        }

        next_state
    }

    fn gen_vertices(&self, tex: &wgpu::Texture, op: &ShapeOp, s: &ShapeOpState) -> Vec<Vertex> {
        if self.tile {
            return self.tiled_vertices(&op.base, tex, s);
        }

        let mut vertex_groups = Vec::new();

        if op.swap.is_some() {
            let dest = op.swap.as_ref().unwrap();
            vertex_groups.push(self.vertices_for_shapes(tex, &op.base, &dest, s));
            vertex_groups.push(self.vertices_for_shapes(tex, &dest, &op.base, s));
        }

        if op.dest.is_some() {
            vertex_groups.push(self.vertices_for_shapes(
                tex,
                &op.base,
                op.dest.as_ref().unwrap(),
                s,
            ));
        }

        if op.swap.is_none() && op.dest.is_none() {
            vertex_groups.push(self.vertices_for_shapes(tex, &op.base, &op.base, s));
        }

        vertex_groups.concat()
    }

    fn vertices_for_shapes(
        &self,
        tex: &wgpu::Texture,
        src: &Shape,
        dest: &Shape,
        s: &ShapeOpState,
    ) -> Vec<Vertex> {
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
        vertices = self.scale_rotate_flip(&mut vertices, tex.width(), tex.height(), s);

        Vertex::to_triangles(vertices)
    }

    fn tiled_vertices(&self, shape: &Shape, tex: &wgpu::Texture, s: &ShapeOpState) -> Vec<Vertex> {
        let width = tex.width();
        let height = tex.height();
        let tex_rect = Rect::from(shape.clone());
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

                vertices = self.scale_rotate_flip(&mut vertices, width, height, s);
                rects.push(Vertex::to_triangles(vertices));
            }
        }

        rects.concat()
    }

    fn scale_rotate_flip(
        &self,
        vertices: &mut Vec<Vertex>,
        width: u32,
        height: u32,
        s: &ShapeOpState,
    ) -> Vec<Vertex> {
        // Clip space
        let mut l = f32::MAX;
        let mut r = f32::MIN;
        let mut t = f32::MIN;
        let mut b = f32::MAX;
        for v in &*vertices {
            let x = v.x();
            let y = v.y();
            if x < l {
                l = x;
            }
            if y < b {
                b = y;
            }
            if x > r {
                r = x;
            }
            if y > t {
                t = y;
            }
        }

        let mut clip_center = Vertex::new(&[l + (r - l) / 2., b + (t - b) / 2.]);

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

        // Translation is in screen pixels -- need to convert to clip space
        // wxh: 1000x1000
        // x: 100 -> 0.1
        // x: -100 -> -0.1
        // y: 100 -> -0.1
        // y: -100 -> 0.1
        let trans = Vertex::new(&match s.translation {
            None => [0., 0.],
            Some(t) => [
                2. * t.0 as f32 / width as f32,
                -2. * t.1 as f32 / height as f32,
            ],
        });

        clip_center.add(&trans);

        vertices
            .iter_mut()
            .map(|v| {
                if self.flip.is_some() {
                    let flip_variant = self.flip.unwrap();

                    if flip_variant == FlipVariant::Both || flip_variant == FlipVariant::Horizontal
                    {
                        v.tex_coord[0] = flip(v.tex_coord[0], l, r);
                    }

                    if flip_variant == FlipVariant::Both || flip_variant == FlipVariant::Vertical {
                        v.tex_coord[1] = flip(v.tex_coord[1], t, b);
                    }
                }

                // TODO: scale, rotate, translate support for tiling
                if !self.tile {
                    if s.translation.is_some() {
                        v.add(&trans);
                    }

                    if self.scale != 1. {
                        v.sub(&clip_center);
                        v.mult_pos(self.scale);
                        v.add(&clip_center);
                    }

                    if s.rotate_deg.is_some() {
                        let rad = s.rotate_deg.unwrap().to_radians();
                        let cos = rad.cos();
                        let sin = rad.sin();

                        let trans_x = v.position[0] - clip_center.position[0];
                        let trans_y = v.position[1] - clip_center.position[1];
                        v.position = [
                            clip_center.position[0] + trans_x * cos - trans_y * sin,
                            clip_center.position[1] + trans_x * sin + trans_y * cos,
                        ];
                    }
                }
                *v
            })
            .collect::<Vec<_>>()
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

impl GpuGunk {
    fn execute(
        &mut self,
        gpu: &mut GpuExecutor,
        vertices: Vec<Vertex>,
        tex: &wgpu::Texture,
        adjs: &[f32; 2],
        chans: &[f32; 4],
        sampler: wgpu::Sampler,
    ) -> wgpu::Texture {
        let adjustments = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("adjustments"),
            size: 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        gpu.queue
            .write_buffer(&adjustments, 0, &bytemuck::cast_slice(adjs));

        let chans_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chans"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        gpu.queue
            .write_buffer(&chans_buf, 0, &bytemuck::cast_slice(chans));

        let render_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render_bind_group2"),
            layout: &self.bg_layout,
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
                    resource: adjustments.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: chans_buf.as_entire_binding(),
                },
            ],
        });

        let output_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("transform output tex"),
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

        let vertex_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vertex_buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encoder"),
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

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &render_bg, &[]);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.draw(0..vertices.len() as u32, 0..1);
        drop(render_pass);

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output_tex
    }

    fn init(gpu: &mut GpuExecutor) -> Self {
        let shader_code = wgpu::include_wgsl!("transform.wgsl");
        let shader = gpu.load_shader("transform", shader_code);

        let bg_layout = gpu
            .device
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
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
                bind_group_layouts: &[&bg_layout],
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

        Self {
            bg_layout,
            render_pipeline,
        }
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
