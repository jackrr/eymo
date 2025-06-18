use crate::imggpu::resize::GpuExecutor;
use crate::pipeline::Detection;
use anyhow::Result;
use image::{EncodableLayout, RgbImage};
use nokhwa::Camera;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{debug, error, warn};

use std::io::Write;
use std::process::{Command, Stdio};

use crate::process_frame;

use nokhwa::{
    nokhwa_initialize,
    pixel_format::RgbFormat,
    query,
    utils::{ApiBackend, RequestedFormat, RequestedFormatType},
    CallbackCamera,
};

pub fn create_input_stream(fps: u32) -> Camera {
    nokhwa_initialize(|granted| {
        debug!("User said {}", granted);
    });

    let cameras = query(ApiBackend::Auto).unwrap();
    cameras
        .iter()
        .for_each(|cam| debug!("Found camera: {:?}", cam));

    let camera = Camera::new(
        cameras.last().unwrap().index().clone(),
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate),
    );

    camera.set_frame_rate(fps)?;
    camera.open_stream().unwrap();
    camera
}

struct OutputVideoStream {
    ffplay: std::process::Child,
}

impl OutputVideoStream {
    // TODO: make configurable to enable v4loopback, whatever is used on mac
    fn new(width: u32, height: u32) -> Result<Self> {
        let ffplay = Command::new("ffplay")
            .args(&[
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                &format!("{}x{}", width, height),
                "-framerate",
                "30",
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

    fn write_frame(&mut self, img: &RgbImage) -> Result<()> {
        if let Some(stdin) = self.ffplay.stdin.as_mut() {
            stdin.write_all(img.as_bytes())?;
            stdin.flush()?;
        }

        Ok(())
    }

    fn close(mut self) -> Result<()> {
        drop(self.ffplay.stdin.take());
        self.ffplay.wait()?;
        Ok(())
    }
}
