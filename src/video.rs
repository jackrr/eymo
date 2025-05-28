use anyhow::{Error, Result};
use image::RgbImage;
use log::{debug, warn};
use show_image::create_window;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use crate::{process_frame, DetectionInput, DetectionResult};

use nokhwa::{
    nokhwa_initialize,
    pixel_format::RgbFormat,
    query,
    utils::{ApiBackend, RequestedFormat, RequestedFormatType},
    CallbackCamera,
};

struct Frame {
    image: RgbImage,
    index: u32,
    rec_at: Instant,
}

const MIN_FPS: u32 = 30;

pub fn process_frames(
    max_threads: usize,
    face_detection: &Arc<RwLock<DetectionResult>>,
    latest_img: &Arc<Mutex<DetectionInput>>,
) -> Result<()> {
    let (sender, receiver) = flume::unbounded();

    let (sender, receiver) = (Arc::new(sender), Arc::new(receiver));

    let queued_frames: Arc<Mutex<HashMap<u32, RgbImage>>> = Arc::new(Mutex::new(HashMap::new()));
    let input_frame_idx = Arc::new(Mutex::new(0));

    let window = create_window("image", Default::default())?;

    // Need to keep camera in a var to prevent collection and maintain stream
    let camera = initialize_source_stream(&sender, input_frame_idx)?;

    let thread_count = max_threads - 1; // -1 for main
    debug!(
        "Initializing {:?} image processing threads...",
        thread_count
    );

    let ms_per_frame_per_thread = 1000 / MIN_FPS * u32::try_from(thread_count).unwrap();
    for i in 0..thread_count {
        let queued_frames = queued_frames.clone();
        let receiver = Arc::clone(&receiver);
        let face_detection = Arc::clone(face_detection);
        let latest_img = Arc::clone(latest_img);

        thread::spawn(move || -> Result<()> {
            while !receiver.is_disconnected() {
                let frame = receiver.recv()?;
                let mut image = frame.image;
                let mut latest_img = latest_img.lock().unwrap();
                let _ = latest_img.insert(image.clone());
                drop(latest_img);

                match process_frame(ms_per_frame_per_thread, &mut image, &face_detection) {
                    Ok(_) => {}
                    Err(err) => warn!(
                        "Thread {} could not complete processing frame {}: {}",
                        i, frame.index, err
                    ),
                }

                let mut frame_queue = queued_frames.lock().unwrap();
                frame_queue.insert(frame.index, image);
                debug!(
                    "Thread {:?} finished frame {:?} in {:?}ms",
                    i,
                    frame.index,
                    frame.rec_at.elapsed().as_millis()
                );
            }
            Ok(())
        });
    }

    let mut output_frame_idx = 0;
    let mut last_frame_at = Instant::now();
    let max_frame_delay = Duration::from_secs(10);

    loop {
        // TODO: frame rate logging
        let mut frame_queue = queued_frames.lock().unwrap();
        if frame_queue.contains_key(&output_frame_idx) {
            let image = frame_queue.remove(&output_frame_idx).unwrap();
            window.set_image("image", image)?;
            debug!(
                "Rendered frame {} after {:?}",
                output_frame_idx,
                last_frame_at.elapsed()
            );
            last_frame_at = Instant::now();
            output_frame_idx += 1;
        } else {
            if last_frame_at.elapsed() > max_frame_delay {
                return Err(Error::msg(format!(
                    "No frame ready after {:?}!",
                    max_frame_delay
                )));
            }
            // debug!("Frame {:?} not ready, waiting 1ms...", output_frame_idx);
            drop(frame_queue); // free lock
            thread::sleep(Duration::from_millis(10));
        }
    }

    // TODO: join the threads

    Ok(())
}

fn initialize_source_stream(
    sender: &Arc<flume::Sender<Frame>>,
    frame_count: Arc<Mutex<u32>>,
) -> Result<CallbackCamera> {
    // only needs to be run on OSX
    nokhwa_initialize(|granted| {
        debug!("User said {}", granted);
    });

    let cameras = query(ApiBackend::Auto).unwrap();
    cameras
        .iter()
        .for_each(|cam| debug!("Found camera: {:?}", cam));

    let s = Arc::clone(sender);
    let mut camera = CallbackCamera::new(
        cameras.last().unwrap().index().clone(),
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate),
        move |buffer| {
            let mut frame_idx = frame_count.lock().unwrap();
            let index = frame_idx.clone();

            *frame_idx += 1;
            drop(frame_idx);

            s.send(Frame {
                image: RgbImage::from(buffer.decode_image::<RgbFormat>().unwrap()),
                index,
                rec_at: Instant::now(),
            })
            .expect("Error sending frame to buffer stream");
            debug!("Sent decoded frame {index} to channel...");
        },
    )?;
    camera.set_frame_rate(30)?;

    camera.open_stream().unwrap();

    Ok(camera)
}
