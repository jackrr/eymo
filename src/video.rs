use crate::pipeline::Pipeline;
use anyhow::Result;
use image::RgbImage;
use log::{debug, error, warn};
use show_image::create_window;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use crate::process_frame;

use nokhwa::{
    nokhwa_initialize,
    pixel_format::RgbFormat,
    query,
    utils::{ApiBackend, RequestedFormat, RequestedFormatType},
    CallbackCamera,
};

const MIN_FPS: u32 = 30;
const MAX_LAG_MS: u128 = 100;

pub fn process_frames(max_threads: usize, pipeline: Pipeline) -> Result<()> {
    let (sender, receiver) = flume::bounded(max_threads);
    let (sender, receiver) = (Arc::new(sender), Arc::new(receiver));
    let window = create_window("image", Default::default())?;
    let ms_per_frame_per_thread = 1000 / MIN_FPS * u32::try_from(max_threads).unwrap();
    let input_frame_idx = Arc::new(Mutex::new(0));

    nokhwa_initialize(|granted| {
        debug!("User said {}", granted);
    });

    let cameras = query(ApiBackend::Auto).unwrap();
    cameras
        .iter()
        .for_each(|cam| debug!("Found camera: {:?}", cam));

    let s = Arc::clone(&sender);
    let input_frame_idx = Arc::clone(&input_frame_idx);

    let pipeline = Arc::new(pipeline);

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

            let pipeline = Arc::clone(&pipeline);

            let join = thread::spawn(move || -> (u32, RgbImage) {
                let mut image = buffer.decode_image::<RgbFormat>().unwrap();

                match process_frame(
                    (ms_per_frame_per_thread as u128 - rec_at.elapsed().as_millis()) as u32,
                    &mut image,
                    pipeline,
                ) {
                    Ok(_) => {}
                    Err(err) => warn!("Could not complete processing frame {}: {}", frame_idx, err),
                }

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

    let mut output_frame_idx = 0;
    let mut last_frame_at = Instant::now();

    while !receiver.is_disconnected() {
        let frame_thread = receiver.recv()?;
        // TODO: frame rate logging
        match frame_thread.join() {
            Ok((frame_idx, image)) => {
                if output_frame_idx > frame_idx {
                    debug!(
                        "Skipping frame {} received while at index {}.",
                        frame_idx, output_frame_idx,
                    );
                    continue;
                }

                window.set_image("image", image)?;
                debug!(
                    "Rendered frame {} after {:?} since previous frame",
                    frame_idx,
                    last_frame_at.elapsed()
                );
                last_frame_at = Instant::now();
                output_frame_idx += 1;
            }
            Err(e) => error!("Frame thread failed: {:?}", e),
        }
    }

    Ok(())
}
