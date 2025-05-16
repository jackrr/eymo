use anyhow::{Error, Result};
use clap::Parser;
use ndarray::{s, Array, Axis, Dim, IxDynImpl, ViewRepr};
use opencv::{
    core::{self, Mat, MatTraitConst, Point, Rect, Scalar, Size, Vector},
    highgui, imgcodecs, imgproc,
    prelude::*,
};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::{
    cmp::{max, min},
    collections::HashSet,
};

const MODEL_YOLO_V11_POSE_M: &str = "yolo11m-pose.onnx";
const MODEL_YOLO_V11_POSE_S: &str = "yolo11s-pose.onnx";
const MODEL_YOLO_V11_POSE_N: &str = "yolo11n-pose.onnx";

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    model: Option<String>,

    #[arg(short, long)]
    image_path: String,

    #[arg(short, long)]
    output_path: Option<String>,

    #[arg(short, long, default_value_t = false)]
    debug: bool,
}

fn log(s: String, debug: bool) {
    if debug {
        println!("{s}");
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    let models = HashSet::from([
        MODEL_YOLO_V11_POSE_S,
        MODEL_YOLO_V11_POSE_N,
        MODEL_YOLO_V11_POSE_M,
    ]);

    let model_name: &str = &args.model.unwrap_or(MODEL_YOLO_V11_POSE_N.to_string());
    let output_path: &str = &args.output_path.unwrap_or("result.png".to_string());

    if models.contains(model_name) {
        log(format!("Using model {model_name:}"), args.debug);
    } else {
        return Err(Error::msg(format!("Unrecognized model {model_name:?}")));
    }

    let frame = imgcodecs::imread(&args.image_path, imgcodecs::IMREAD_COLOR)?;
    let mut resized = Mat::default();

    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(format!("./models/{model_name:}"))?;

    let height = 640;
    let width = 640;

    // TODO: is the resize weird?
    imgproc::resize(
        &frame,
        &mut resized,
        Size { width, height },
        0.0,
        0.0,
        imgproc::INTER_AREA,
    )?;

    let mut resized_f32 = Array::zeros((1, 3, height as usize, width as usize));
    let height = resized.rows() as usize;
    let width = resized.cols() as usize;
    let chans = resized.channels() as usize;

    // b r g -> r g b
    let chan_map = vec![2, 0, 1];

    unsafe {
        let mat_slice = std::slice::from_raw_parts(resized.data(), height * width * chans);

        for y in 0..height {
            for x in 0..width {
                for ch in 0..chans {
                    let idx = (y * width * chans) + (x * chans) + ch;
                    resized_f32[[0, chan_map[ch], y, x]] = (mat_slice[idx] as f32) / 255.;
                }
            }
        }
    }

    let input = Tensor::from_array(resized_f32)?;
    let outputs = model.run(ort::inputs!["images" => input]?)?;
    let results = outputs["output0"].try_extract_tensor::<f32>()?;

    show_results(&mut resized, results, 0, 0)?;

    imgcodecs::imwrite(output_path, &resized, &Vector::new())?;

    Ok(())
}

fn show_results(
    img: &mut Mat,
    result: ndarray::ArrayBase<ViewRepr<&f32>, Dim<IxDynImpl>>,
    x_offset: i32,
    y_offset: i32,
) -> Result<()> {
    for row in result.squeeze().columns() {
        let row: Vec<_> = row.iter().copied().collect();
        let c = row[4];
        if c < 0.8 {
            continue;
        }

        let xc = row[0] as i32 + x_offset; // centerpoint x
        let yc = row[1] as i32 + y_offset; // centerpoint y
        let w = row[2].round() as i32;
        let h = row[3].round() as i32;

        imgproc::rectangle(
            img,
            Rect::new(xc - w / 2, yc - h / 2, w, h),
            core::VecN([255., 0., 0., 0.]),
            1,
            imgproc::LINE_8,
            0,
        )?;

        for k in 0..8 {
            let k_idx = 5 + (k * 3);
            let kc = row[k_idx + 2];

            if kc < 0.8 {
                continue;
            }

            let kx = row[k_idx].round() as i32 + x_offset;
            let ky = row[k_idx + 1].round() as i32 + y_offset;

            imgproc::circle(
                img,
                Point::new(kx, ky),
                5,
                Scalar::new(0., 0., 255., 0.),
                -1, // fill
                imgproc::LINE_8,
                0,
            )?;

            imgproc::put_text(
                img,
                &format!("{k} - {kcr}", kcr = kc.round() as i32),
                Point::new(kx + 10, ky + 5),
                imgproc::FONT_HERSHEY_COMPLEX_SMALL,
                0.5,
                Scalar::new(255., 255., 255., 255.),
                1, //thickness
                imgproc::LINE_8,
                false,
            )?;
        }
    }

    Ok(())
}

fn expand_rect(rect: Rect, pad: i32, max_x: i32, max_y: i32) -> Rect {
    let tl = Point::new(max(rect.x - pad, 0), max(rect.y - pad, 0));
    let br = Point::new(
        min(rect.x + rect.width + pad, max_x),
        min(rect.y + rect.height + pad, max_y),
    );

    Rect {
        x: tl.x,
        y: tl.y,
        width: br.x - tl.x,
        height: br.y - tl.y,
    }
}
