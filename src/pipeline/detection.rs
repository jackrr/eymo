use super::model::{initialize_model, Session};
use crate::imggpu;
use crate::imggpu::gpu::GpuExecutor;
use crate::imggpu::vertex::Vertex;
use crate::shapes::point::PointF32;
use crate::shapes::rect::{Rect, RectF32};
use anchors::gen_anchors;
use anyhow::Result;
use ort::session::SessionOutputs;
use tracing::{span, trace, Level};
use wgpu::util::DeviceExt;

mod anchors;

const WIDTH: u32 = 128;
const HEIGHT: u32 = 128;

pub struct FaceDetector {
    model: Session,
    anchors: [RectF32; 896],
}

#[derive(Debug, Clone)]
pub struct Face {
    pub bounds: Rect,
    pub l_eye: PointF32,
    pub r_eye: PointF32,
    confidence: f32,
}

impl Face {
    pub fn with_eyes(confidence: f32, bounds: Rect, l_eye: PointF32, r_eye: PointF32) -> Face {
        Face {
            l_eye,
            r_eye,
            bounds,
            confidence,
        }
    }

    pub fn rot_theta(&self) -> f32 {
        let dx = self.r_eye.x - self.l_eye.x;
        let dy = self.r_eye.y - self.l_eye.y;
        dy.atan2(dx)
    }
}

impl FaceDetector {
    /*
    BlazeFace model wrapper using ort to run the model, then manually
    process the results into one or more faces


    Model Input: 128x128 f32 image
    Model Output:
    - 896 length array of confidence scores (classificators)
      - 896 length 2D array of detection coords (regressors)

    The first 4 values in the detection coords are centroid, width,
    height offsets applied to a particular cell among 2 predetermined
    grids. The index in the 896 length array determines which square
    in the grids is referred to.  The remaning 12 values are points
    for key features (eyes, ears, etc).

    // scale gets interpolated

     */
    pub fn new(threads: usize) -> Result<FaceDetector> {
        Ok(FaceDetector {
            model: initialize_model("mediapipe_face_detection_short_range.onnx", threads)?,
            anchors: gen_anchors(),
        })
    }

    pub fn run_gpu(&mut self, tex: &wgpu::Texture, gpu: &mut GpuExecutor) -> Result<Vec<Face>> {
        // TODO: CLEAN ME UP
        let span = span!(Level::DEBUG, "face_detector");
        let _guard = span.enter();

        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let shader_code = wgpu::include_wgsl!("detection.wgsl");
        let shader = gpu.load_shader("detection", shader_code);

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

        let out_dims = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_dims"),
            size: 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(
            &out_dims,
            0,
            &bytemuck::cast_slice(&[(WIDTH as f32), (HEIGHT as f32)]),
        );

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
                    resource: out_dims.as_entire_binding(),
                },
            ],
        });

        let resize_output_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
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

        let vertices = Vertex::triangles_for_full_coverage();
        let vertex_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vertex_buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render_pass"),
            color_attachments: &[
                // This is what @location(0) in the fragment shader targets
                Some(wgpu::RenderPassColorAttachment {
                    view: &resize_output_tex.create_view(&Default::default()),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(Default::default()),
                        // load: wgpu::LoadOp::Load,    // read previous layer
                        store: wgpu::StoreOp::Store, // overwrite with fragment output
                    },
                }),
            ],
            ..Default::default()
        });

        render_pass.set_pipeline(&render_pipeline);
        render_pass.set_bind_group(0, &render_bg, &[]);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.draw(0..vertices.len() as u32, 0..1);
        drop(render_pass);

        gpu.queue.submit(std::iter::once(encoder.finish()));

        let tensor = imggpu::rgb::texture_to_tensor(
            gpu,
            &resize_output_tex,
            imggpu::rgb::OutputRange::NegOneToOne,
        )?;
        let outputs = self.model.run(ort::inputs!["input" => tensor]?)?;
        self.extract_results(outputs, tex.width(), tex.height(), WIDTH, HEIGHT)
    }

    fn extract_results(
        &self,
        outputs: SessionOutputs,
        input_width: u32,
        input_height: u32,
        resized_width: u32,
        resized_height: u32,
    ) -> Result<Vec<Face>> {
        let regressors = outputs["regressors"].try_extract_tensor::<f32>()?;
        let classificators = outputs["classificators"].try_extract_tensor::<f32>()?;

        let scores = classificators.as_slice().unwrap();

        let detections = regressors.squeeze();
        let mut row_idx = 0;
        let mut results: Vec<Face> = Vec::new();

        for res in detections.rows() {
            let score = sigmoid_stable(scores[row_idx]);
            if score > 0.5 {
                let x_scale = input_width as f32 / resized_width as f32;
                let y_scale = input_height as f32 / resized_height as f32;

                // TODO: gen_anchor needs work...
                // let mut anchor = gen_anchor(row_idx.try_into().unwrap())?;
                let mut anchor = self.anchors[row_idx].clone();
                let ax = anchor.x.clone();
                let ay = anchor.y.clone();

                let scaled: Rect = anchor
                    .adjust(res[0], res[1], res[2], res[3])
                    .scale(x_scale, y_scale)
                    .into();

                let mut better_found = false;
                for (i, d) in results.iter().enumerate() {
                    if d.bounds.overlap_pct(&scaled) > 30. {
                        if d.confidence > score {
                            better_found = true;
                        } else {
                            results.swap_remove(i);
                        }
                        break;
                    }
                }
                if !better_found {
                    let l_eye = PointF32 {
                        x: ((ax + res[4]) * x_scale),
                        y: ((ay + res[5]) * y_scale),
                    };
                    let r_eye = PointF32 {
                        x: ((ax + res[6]) * x_scale),
                        y: ((ay + res[7]) * y_scale),
                    };
                    results.push(Face::with_eyes(score, scaled, l_eye, r_eye));
                }
            }
            row_idx += 1;
        }

        trace!("Detected {} faces", results.len());

        Ok(results)
    }
}

fn sigmoid_stable(x: f32) -> f32 {
    if x >= 0. {
        1. / (1. + (-x).exp())
    } else {
        x.exp() / (1. + x.exp())
    }
}
