use crate::shapes::rect::Rect;
use anyhow::Result;
use image::{GenericImage, GenericImageView, RgbImage};

pub fn image_at(bounds: Rect, src: &RgbImage) -> Result<RgbImage> {
    let view = *src.view(
        bounds.left().try_into().unwrap(),
        bounds.top().try_into().unwrap(),
        bounds.w,
        bounds.h,
    );
    let mut img = RgbImage::new(bounds.w, bounds.h);
    img.copy_from(&view, 0, 0)?;

    Ok(img)
}
