use flume;
use ggez::{
    event::{run, EventHandler},
    graphics::{Canvas, Image},
    Context, ContextBuilder, GameError,
};

pub fn start_camera() -> CallbackCamera {
    let (tx, rx) = flume::unbounded();
}

// struct CaptureState {
//     receiver: Arc<Receiver<FrameBuffer>>,
//     buffer: Vec<u8>,
//     format: CameraFormat,
// }

// impl EventHandler<GameError> for CaptureState {
//     fn update(&mut self, _ctx: &mut Context) -> Result<(), GameError> {
//         Ok(())
//     }

//     fn draw(&mut self, ctx: &mut Context) -> Result<(), GameError> {
//         let buffer = self
//             .receiver
//             .recv()
//             .map_err(|why| GameError::RenderError(why.to_string()))?;
//         self.buffer
//             .resize(yuyv422_predicted_size(buffer.buffer().len(), true), 0);
//         buffer
//             .decode_image_to_buffer::<RgbAFormat>(&mut self.buffer)
//             .map_err(|why| GameError::RenderError(why.to_string()))?;
//         let image = Image::from_pixels(
//             ctx,
//             &self.buffer,
//             ImageFormat::Rgba8Uint,
//             self.format.width(),
//             self.format.height(),
//         );
//         let canvas = Canvas::from_image(ctx, image, None);
//         canvas.finish(ctx)
//     }
// }
