use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

pub async fn from_frame(frame: &web_sys::VideoFrame) -> Result<image::RgbaImage, JsValue> {
    let width = frame.coded_width();
    let height = frame.coded_height();

    let img = image::RgbaImage::new(width, height);
    let mut img_data = img.into_raw();
    let options = web_sys::VideoFrameCopyToOptions::new();
    // Need https://github.com/wasm-bindgen/wasm-bindgen/pull/4543 to release for this
    // options.set_format("RGBA");
    let obj = options.value_of();
    js_sys::Reflect::set(&obj, &js_sys::JsString::from("format"), &js_sys::JsString::from("RGBA"))?;

    JsFuture::from(frame.copy_to_with_u8_slice_and_options(&mut img_data, &options))
        .await
        .unwrap();

    Ok(image::RgbaImage::from_raw(width, height, img_data).unwrap())
}
