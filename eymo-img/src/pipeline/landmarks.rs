use super::Face;
use super::detection;
use super::model::{Model, initialize_model};
use crate::imggpu;
use crate::imggpu::gpu::GpuExecutor;
use crate::imggpu::vertex::Vertex;
use crate::shapes::point::Point;
use crate::shapes::polygon::Polygon;
use crate::shapes::rect::Rect;
use anyhow::Result;
use tracing::{Level, debug, span};
use tract_nnef::prelude::tvec;
use wgpu::util::DeviceExt;

pub struct FaceLandmarker {
    model: Model,
}

const HEIGHT: u32 = 192;
const WIDTH: u32 = 192;

const FACE_IDXS: [usize; 37] = [
    10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377,
    400, 378, 379, 365, 397, 288, 435, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338,
];

const MOUTH_IDXS: [usize; 19] = [
    164, 167, 165, 216, 212, 202, 204, 194, 201, 200, 421, 418, 424, 422, 432, 436, 322, 391, 393,
];

const L_EYE_IDXS: [usize; 16] = [
    133, 173, 157, 158, 159, 160, 161, 246, 33, 7, 163, 144, 145, 153, 154, 155,
];
const L_EYE_REGION_IDXS: [usize; 17] = [
    107, 66, 105, 63, 70, 156, 35, 31, 228, 229, 230, 231, 232, 233, 245, 193, 55,
];
const R_EYE_IDXS: [usize; 16] = [
    263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249,
];
const R_EYE_REGION_IDXS: [usize; 17] = [
    300, 293, 334, 296, 336, 285, 417, 465, 453, 452, 451, 450, 449, 448, 261, 265, 353,
];

const NOSE_IDXS: [usize; 23] = [
    168, 193, 245, 188, 174, 236, 198, 209, 49, 64, 98, 97, 2, 326, 327, 294, 279, 429, 420, 456,
    419, 351, 417,
];

const MODEL: &[u8; 1435541] = include_bytes!("./face_landmark.tar.gz");

impl FaceLandmarker {
    pub fn new() -> Result<FaceLandmarker> {
        Ok(FaceLandmarker {
            model: initialize_model(MODEL)?,
        })
    }

    // FIXME: when face is notably tilted detections get
    // wonky.. something wrong with rotation in here probably
    pub async fn run_gpu(
        &mut self,
        face: &detection::Face,
        tex: &wgpu::Texture,
        gpu: &mut GpuExecutor,
    ) -> Result<Face> {
        let span = span!(Level::DEBUG, "face_landmarker");
        let _guard = span.enter();

        let theta = face.rot_theta();
        debug!("Tilt: {}", theta.to_degrees());
        let mut bounds = face.bounds.clone();
        // pad 30% vertically
        bounds = bounds.scale_y(1.6, tex.height());

        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let shader_code = wgpu::include_wgsl!("landmarks.wgsl");
        let shader = gpu.load_shader("landmarks", shader_code);

        let render_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("render_pipeline"),
                layout: None,
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
            size: 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&rot, 0, &bytemuck::cast_slice(&[theta.cos(), theta.sin()]));

        let render_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render_bind_group"),
            layout: &render_pipeline.get_bind_group_layout(0),
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

        let output_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
        });

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encoder"),
            });

        let right = bounds.right() as f32 / tex.width() as f32;
        let left = bounds.left() as f32 / tex.width() as f32;
        let top = bounds.top() as f32 / tex.height() as f32;
        let bottom = bounds.bottom() as f32 / tex.height() as f32;
        let vertices = Vec::from([
            Vertex::new_with_tex(&[1., 1.], &[right, top]),
            Vertex::new_with_tex(&[-1., 1.], &[left, top]),
            Vertex::new_with_tex(&[-1., -1.], &[left, bottom]),
            Vertex::new_with_tex(&[-1., -1.], &[left, bottom]),
            Vertex::new_with_tex(&[1., -1.], &[right, bottom]),
            Vertex::new_with_tex(&[1., 1.], &[right, top]),
        ]);
        let vertex_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vertex_buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &output_tex.create_view(&Default::default()),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(Default::default()),
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

        let gpu_span = span!(Level::DEBUG, "face_landmarker:gpu_run");
        let gpu_guard = gpu_span.enter();
        gpu.queue.submit(std::iter::once(encoder.finish()));
        drop(gpu_guard);

        let tensor =
            imggpu::rgb::texture_to_tensor(gpu, &output_tex, imggpu::rgb::OutputRange::ZeroToOne)
                .await?;

        // FIXME: this takes ~65ms on WASM!
        let model_span = span!(Level::DEBUG, "face_landmarker:model_run");
        let model_guard = model_span.enter();
        let outputs = self.model.run(tvec!(tensor.into()))?;
        drop(model_guard);

        let output = outputs[0].to_array_view::<f32>()?;
        let mesh = output.squeeze().squeeze().squeeze();
        let r = mesh.as_slice().unwrap();

        extract_results(r, WIDTH, HEIGHT, bounds, -theta)
    }
}

fn extract_results(
    r: &[f32],
    input_width: u32,
    input_height: u32,
    run_bounds: Rect,
    run_rot: f32,
) -> Result<Face> {
    let x_scale = run_bounds.w as f32 / input_width as f32;
    let y_scale = run_bounds.h as f32 / input_height as f32;
    let x_offset = run_bounds.left() as f32;
    let y_offset = run_bounds.top() as f32;
    let origin = run_bounds.center();

    Ok(Face {
        bound: run_bounds,
        face: extract_feature(
            r, &FACE_IDXS, x_offset, y_offset, x_scale, y_scale, &origin, run_rot,
        ),
        nose: extract_feature(
            r, &NOSE_IDXS, x_offset, y_offset, x_scale, y_scale, &origin, run_rot,
        ),
        mouth: extract_feature(
            r,
            &MOUTH_IDXS,
            x_offset,
            y_offset,
            x_scale,
            y_scale,
            &origin,
            run_rot,
        ),
        l_eye: extract_feature(
            r,
            &L_EYE_IDXS,
            x_offset,
            y_offset,
            x_scale,
            y_scale,
            &origin,
            run_rot,
        ),
        l_eye_region: extract_feature(
            r,
            &L_EYE_REGION_IDXS,
            x_offset,
            y_offset,
            x_scale,
            y_scale,
            &origin,
            run_rot,
        ),
        r_eye_region: extract_feature(
            r,
            &R_EYE_REGION_IDXS,
            x_offset,
            y_offset,
            x_scale,
            y_scale,
            &origin,
            run_rot,
        ),
        r_eye: extract_feature(
            r,
            &R_EYE_IDXS,
            x_offset,
            y_offset,
            x_scale,
            y_scale,
            &origin,
            run_rot,
        ),
    })
}

fn extract_feature(
    mesh: &[f32],
    kpt_idxs: &[usize],
    x_offset: f32,
    y_offset: f32,
    x_scale: f32,
    y_scale: f32,
    origin: &Point,
    rotation: f32,
) -> Polygon {
    let mut points = Vec::new();

    for i in kpt_idxs {
        let idx = i * 3;
        let x = x_offset + mesh[idx] * x_scale;
        let y = y_offset + mesh[idx + 1] * y_scale;

        let mut p = Point::new(x.round() as u32, y.round() as u32);

        p.rotate(*origin, rotation);

        points.push(p)
    }

    Polygon::new(points)
}
