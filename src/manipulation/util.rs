use crate::shapes::polygon::Polygon;
use crate::shapes::rect::Rect;
use anyhow::Result;
use image::{GenericImage, GenericImageView, RgbImage};
use log::debug;

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

pub fn copy_from(
    src_r: Polygon,
    src_img: &RgbImage,
    dest_p: &Polygon,
    dest_img: &mut RgbImage,
) -> Result<()> {
    for (p, src_p) in dest_p.iter_pairwise_projection_onto(src_r) {
        if src_p.x >= src_img.width()
            || src_p.y >= src_img.height()
            || p.x >= dest_img.width()
            || p.y >= dest_img.height()
        {
            // FIXME: this happens often on src.x = src_img.width / src.y = src_img.height
            debug!(
                "Out of bounds pixel swap{p:?} {src_p:?} img: {}x{} src_img: {}x{}",
                dest_img.width(),
                dest_img.height(),
                src_img.width(),
                src_img.height(),
            );
            continue;
        }
        dest_img.put_pixel(p.x, p.y, *src_img.get_pixel(src_p.x, src_p.y));
    }
    Ok(())
}
