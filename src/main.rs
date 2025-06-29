#![warn(unused_extern_crates)]
use crate::pipeline::Pipeline;
use ab_glyph::{FontRef, PxScale};
use anyhow::{Error, Result};
use clap::Parser;
use image::imageops::resize;
use image::{imageops, DynamicImage, Pixel, Rgb, RgbImage, Rgba, RgbaImage};
use imageproc::drawing::{draw_filled_circle, draw_text};
use imageproc::point::Point as ProcPoint;
use imggpu::resize::GpuExecutor;
use imggpu::rgb;
use imggpu::vertex::Vertex;
use nokhwa::pixel_format::RgbAFormat;
use nokhwa::{Buffer, FormatDecoder};
use num_cpus::get as get_cpu_count;
use shapes::point::Point;
use shapes::rect::Rect;
use std::time::Instant;
use tracing::{debug, error, info, span, trace, warn, Level};
use tracing_subscriber::fmt;
use tracing_subscriber::fmt::format::FmtSpan;
use transform::Transform;
use video::{create_input_stream, OutputVideoStream};
mod imggpu;
mod pipeline;
mod shapes;
mod transform;
mod triangulate;
mod video;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    threads: Option<usize>,

    #[arg(short, long, default_value = "500")]
    max_frame_lag_ms: u32,

    #[arg(short, long, default_value = "30")]
    fps: u32,
}

fn main() -> Result<()> {
    fmt::fmt()
        .with_span_events(FmtSpan::CLOSE)
        .with_target(false)
        .init();

    let args = Args::parse();

    let total_threads = get_cpu_count();
    let total_threads = args.threads.unwrap_or(total_threads).min(total_threads);
    let mut pipeline = Pipeline::new(total_threads / 2)?;

    // TODO: allow arg to specify a video source
    let mut camera = create_input_stream(args.fps)?;

    let resolution = camera.resolution();
    let mut output_stream = OutputVideoStream::new(resolution.width(), resolution.height())?;
    let mut gpu = GpuExecutor::new()?;

    loop {
        let span = span!(Level::INFO, "frame_loop_iter");
        let _guard = span.enter();

        let result = camera.frame();
        let frame = match result {
            Ok(frame) => frame,
            Err(e) => {
                error!("Failed to pull frame from webcam: {e:?}");
                break;
            }
        };

        match process_frame(frame, &mut gpu, &mut pipeline, args.max_frame_lag_ms) {
            Ok(img) => match output_stream.write_frame(img) {
                Ok(_) => trace!("Rendered frame."),
                Err(e) => error!("Failed to render frame: {e:?}"),
            },
            Err(e) => error!("Failed to process frame: {e:?}"),
        }
        break;
    }

    output_stream.close()?;
    camera.stop_stream()?;

    Ok(())
}

fn process_frame(
    frame: Buffer,
    gpu: &mut GpuExecutor,
    pipeline: &mut Pipeline,
    within_ms: u32,
) -> Result<RgbaImage> {
    let span = span!(Level::INFO, "process_frame");
    let _guard = span.enter();
    let start = Instant::now();
    let font = FontRef::try_from_slice(include_bytes!(
        "/usr/share/fonts/fira-code/FiraCode-Bold.ttf"
    ))?;

    trace!(
        "Byte len: {}, res: {}, format {:?}",
        frame.buffer().len(),
        frame.resolution(),
        frame.source_frame_format()
    );
    // TODO: can we get nokwha to give us rgba byte buffer to prevent need for decoding?
    let input_img: RgbaImage = frame.decode_image::<RgbAFormat>()?;
    // let input_img = image::open("./tmp/input_img.jpg")?;
    // let input_img: RgbaImage = input_img.into();

    let texture =
        gpu.rgba_buffer_to_texture(input_img.as_raw(), input_img.width(), input_img.height());
    let detection = pipeline.run_gpu(&texture, gpu)?;

    // TODO: make check time return image early instead of erroring
    check_time(within_ms, start, "Face Detection")?;

    let mut output = texture;

    let mut img = rgb::texture_to_rgba(gpu, &output);

    for face in detection.faces {
        info!("Handling face {:?}", face);
        // let mut t = Transform::new(face.mouth.clone());
        // t.set_scale(3.0);
        // // TODO: fix rotate
        // // t.set_rot_degrees(90.);
        // output = t.execute(gpu, &output)?;

        // let mut t = Transform::new(face.l_eye.clone());
        // t.set_scale(2.0);
        // // t.set_rot_degrees(90.);
        // output = t.execute(gpu, &output)?;

        // let mut t = Transform::new(face.r_eye.clone());
        // t.set_scale(2.);
        // // t.set_rot_degrees(90.);
        // output = t.execute(gpu, &output)?;

        let scale = 4i32;

        img = resize(
            &img,
            img.width() * scale as u32,
            img.height() * scale as u32,
            imageops::FilterType::Nearest,
        );
        for l in face.corner_sus {
            img = draw_text(
                &img,
                Rgba([255u8, 255u8, 255u8, 255u8]),
                scale * l.point.x as i32 + 10,
                scale * l.point.y as i32 + 10,
                PxScale::from(8.),
                &font,
                &format!("{}", l.idx),
            );

            img = draw_filled_circle(
                &img,
                (scale * l.point.x as i32, scale * l.point.y as i32),
                2,
                Rgba([255u8, 255u8, 255u8, 255u8]),
            );
        }

        break;

        check_time(within_ms, start, &format!("Image Manipulation TODO: index"))?;
    }

    img.save("tmp/corner_candidates.png")?;

    Ok(img)
}

fn draw_tris(tris: Vec<Vertex>, img: RgbImage) -> RgbImage {
    let mut img = img;
    let width = img.width() as f32;
    let height = img.height() as f32;
    info!("{tris:?}");
    for i in 0..tris.len() / 3 {
        let idx = i * 3;
        let points = [
            ProcPoint::new(
                (tris[idx].tex_coord[0] * width) as i32,
                (tris[idx].tex_coord[1] * height) as i32,
            ),
            ProcPoint::new(
                (tris[idx + 1].tex_coord[0] * width) as i32,
                (tris[idx + 1].tex_coord[1] * height) as i32,
            ),
            ProcPoint::new(
                (tris[idx + 2].tex_coord[0] * width) as i32,
                (tris[idx + 2].tex_coord[1] * height) as i32,
            ),
        ];
        info!("POINTS: {points:?}");
        let r = 55 + ((i * 40) % 200);
        let color = Rgb::from([r as u8, 0u8, 0u8]);
        img = imageproc::drawing::draw_polygon(&img, &points, color);
    }

    img
}

fn check_time(within_ms: u32, start: Instant, waypoint: &str) -> Result<()> {
    let elapsed_ms = start.elapsed().as_millis();
    if elapsed_ms >= within_ms.into() {
        return Err(Error::msg(format!(
            "{elapsed_ms}ms exceeds allowed time of {within_ms}ms at {waypoint}",
        )));
    }

    debug!("{elapsed_ms}ms at {waypoint}");

    Ok(())
}
