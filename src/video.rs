use crate::imggpu::resize::GpuExecutor;
use crate::pipeline::Detection;
use anyhow::Result;
use image::{EncodableLayout, RgbImage};
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

// TODO: make these configurable
const TARGET_FPS: u32 = 30;
const MAX_LAG_MS: u128 = 500;

pub fn process_frames(
    max_threads: usize,
    detection: Arc<RwLock<Option<Detection>>>,
    latest_frame: Arc<Mutex<Option<RgbImage>>>,
) -> Result<()> {
    let (sender, receiver) = flume::bounded(max_threads);
    let (sender, receiver) = (Arc::new(sender), Arc::new(receiver));
    let input_frame_idx = Arc::new(Mutex::new(0));

    // Input stream
    nokhwa_initialize(|granted| {
        debug!("User said {}", granted);
    });

    let cameras = query(ApiBackend::Auto).unwrap();
    cameras
        .iter()
        .for_each(|cam| debug!("Found camera: {:?}", cam));

    let s = Arc::clone(&sender);
    let input_frame_idx = Arc::clone(&input_frame_idx);

    let num_gpus = 4;
    let (push_gpu, pull_gpu) = flume::bounded(num_gpus);
    let (push_gpu, pull_gpu) = (Arc::new(push_gpu), Arc::new(pull_gpu));
    for _ in 0..num_gpus {
        push_gpu.send(GpuExecutor::new()?)?;
    }

    let mut camera = CallbackCamera::new(
        cameras.last().unwrap().index().clone(),
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate),
        move |buffer| {
            let rec_at = Instant::now();

            let mut frame_idx_mtx = input_frame_idx.lock().unwrap();
            let frame_idx = frame_idx_mtx.clone();
            *frame_idx_mtx += 1;
            drop(frame_idx_mtx);

            debug!("Processing frame {frame_idx}");

            let latest_frame = Arc::clone(&latest_frame);
            let detection = Arc::clone(&detection);
            let pull_gpu = Arc::clone(&pull_gpu);
            let push_gpu = Arc::clone(&push_gpu);

            let join = thread::spawn(move || -> (u32, RgbImage) {
                let mut image = buffer.decode_image::<RgbFormat>().unwrap();
                // Store newest image for ML model to read from
                let mut latest_frame = latest_frame.lock().unwrap();
                *latest_frame = Some(image.clone());
                drop(latest_frame);

                let mut gpu = match pull_gpu.recv() {
                    Ok(gpu) => gpu,
                    Err(err) => {
                        error!("Failed to pull GpuExecutor off channel: {:?}", err);
                        return (frame_idx, image);
                    }
                };
                match process_frame(
                    MAX_LAG_MS.saturating_sub(rec_at.elapsed().as_millis()) as u32,
                    &mut image,
                    detection,
                    &mut gpu,
                ) {
                    Ok(_) => {}
                    Err(err) => warn!("Could not complete processing frame {}: {}", frame_idx, err),
                }
                push_gpu.send(gpu);

                let total_ms = rec_at.elapsed().as_millis();
                debug!("Finished processing frame {frame_idx} in {total_ms}ms");
                if total_ms > MAX_LAG_MS {
                    warn!("Took {total_ms}ms to process frame...");
                }

                (frame_idx, image)
            });
            s.send(join).expect("Error sending thread to buffer stream");
        },
    )?;

    camera.set_frame_rate(30)?;
    camera.open_stream().unwrap();

    let resolution = camera.resolution()?;

    // Output stream
    // TODO: configurable destination from clap args
    let mut output_stream = OutputVideoStream::new(resolution.width(), resolution.height())?;

    let mut output_frame_idx = 0;
    let mut last_frame_at = Instant::now();

    while !receiver.is_disconnected() {
        let frame_thread = receiver.recv()?;
        match frame_thread.join() {
            Ok((frame_idx, image)) => {
                if output_frame_idx > frame_idx {
                    debug!(
                        "Skipping frame {} received while at index {}.",
                        frame_idx, output_frame_idx,
                    );
                    continue;
                }

                output_stream.write_frame(&image)?;
                debug!(
                    "Rendered frame {} after {:?} since previous frame",
                    frame_idx,
                    last_frame_at.elapsed()
                );
                last_frame_at = Instant::now();
                output_frame_idx += 1;

                thread::sleep(Duration::from_millis((1000 / TARGET_FPS).into()));
            }
            Err(e) => error!("Frame thread failed: {:?}", e),
        }
    }

    output_stream.close()?;

    Ok(())
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
