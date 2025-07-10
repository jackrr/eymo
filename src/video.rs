use anyhow::Result;
use image::{EncodableLayout, RgbaImage};
use tracing::{debug, error};

use std::io::Write;
use std::process::{Command, Stdio};

use nokhwa::{
    nokhwa_initialize,
    pixel_format::RgbAFormat,
    query,
    utils::{ApiBackend, RequestedFormat, RequestedFormatType},
    Camera,
};

pub fn create_input_stream(fps: u32) -> Result<Camera> {
    nokhwa_initialize(|granted| {
        debug!("User said {}", granted);
    });

    let cameras = query(ApiBackend::Auto).unwrap();
    cameras
        .iter()
        .for_each(|cam| debug!("Found camera: {:?}", cam));

    let mut camera = Camera::new(
        cameras.last().unwrap().index().clone(),
        RequestedFormat::new::<RgbAFormat>(RequestedFormatType::AbsoluteHighestFrameRate),
    )?;

    camera.set_frame_rate(fps)?;
    camera.open_stream().unwrap();
    Ok(camera)
}

pub struct OutputVideoStream {
    output_proc: std::process::Child,
}

impl Drop for OutputVideoStream {
    fn drop(&mut self) {
        match self.output_proc.kill() {
            Err(e) => error!("Failed to stop output process {e:?}"),
            Ok(_) => {}
        }
    }
}

impl OutputVideoStream {
    pub fn new(width: u32, height: u32, device: Option<String>) -> Result<Self> {
        let mut command = match device {
            Some(d) => {
                let mut command = Command::new("ffmpeg");
                command.args(&[
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgba",
                    "-s",
                    &format!("{}x{}", width, height),
                    "-i",
                    "-",
                    "-map",
                    "0:v",
                    "-preset",
                    "fast",
                    "-vf",
                    "format=yuv420p",
                    "-f",
                    "v4l2",
                    &format!("/dev/{d}"),
                ]);
                command
            }
            None => {
                let mut command = Command::new("ffplay");
                command.args(&[
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgba",
                    "-video_size",
                    &format!("{}x{}", width, height),
                    "-fflags",
                    "nobuffer",
                    "-flags",
                    "low_delay",
                    "-",
                ]);
                command
            }
        };
        let output_proc = command
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()?;

        Ok(Self { output_proc })
    }

    pub fn write_frame(&mut self, img: RgbaImage) -> Result<()> {
        if let Some(stdin) = self.output_proc.stdin.as_mut() {
            stdin.write_all(img.as_bytes())?;
        }

        Ok(())
    }
}
