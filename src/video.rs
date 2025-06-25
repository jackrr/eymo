use anyhow::Result;
use image::{EncodableLayout, RgbImage};
use tracing::debug;

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
    ffplay: std::process::Child,
}

impl OutputVideoStream {
    // TODO: make configurable to enable v4loopback, whatever is used on mac
    pub fn new(width: u32, height: u32) -> Result<Self> {
        let ffplay = Command::new("ffplay")
            .args(&[
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                &format!("{}x{}", width, height),
                "-framerate",
                "7",
                "-fflags",
                "nobuffer",
                "-flags",
                "low_delay",
                "-",
            ])
            .stdin(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;

        Ok(Self { ffplay })
    }

    pub fn write_frame(&mut self, img: RgbImage) -> Result<()> {
        if let Some(stdin) = self.ffplay.stdin.as_mut() {
            stdin.write_all(img.as_bytes())?;
            stdin.flush()?;
        }

        Ok(())
    }

    pub fn close(mut self) -> Result<()> {
        drop(self.ffplay.stdin.take());
        self.ffplay.wait()?;
        Ok(())
    }
}
