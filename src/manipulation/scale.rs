use super::{util, Executable};
use crate::shapes::{rect::Rect, shape::Shape};
use anyhow::{Error, Result};
use image::{imageops::resize, imageops::FilterType, GenericImage, RgbImage};
use tracing::warn;

#[derive(Debug, Clone)]
pub struct Scale {
    target: Shape,
    zoom: f32,
}

impl Scale {
    pub fn new(target: Shape, zoom: f32) -> Self {
        Self { target, zoom }
    }
}

impl Executable for Scale {
    fn execute(&self, img: &mut RgbImage) -> Result<()> {
        let zoom = self.zoom;
        let src_img = util::image_at(self.target.clone().into(), img)?;
        let src_img = resize(
            &src_img,
            (src_img.width() as f32 * zoom).round() as u32,
            (src_img.height() as f32 * zoom).round() as u32,
            FilterType::Triangle,
        );

        match &self.target {
            Shape::Polygon(poly) => {
                let enlarged_poly = poly.scale(zoom, img.width(), img.height());

                for (dest_p, src_p) in enlarged_poly.iter_pairwise_projection_onto(Rect::from_tl(
                    0,
                    0,
                    src_img.width() - 1,
                    src_img.height() - 1,
                )) {
                    if src_p.x >= src_img.width()
                        || src_p.y >= src_img.height()
                        || dest_p.x >= img.width()
                        || dest_p.y >= img.height()
                    {
                        warn!("Skipping {src_p:?} -> {dest_p:?}");
                        continue;
                    }

                    img.put_pixel(dest_p.x, dest_p.y, *src_img.get_pixel(src_p.x, src_p.y));
                }
            }
            Shape::Rect(r) => {
                let enlarged = r.clone().scale(zoom, src_img.width(), src_img.height());
                img.copy_from(&src_img, enlarged.left(), enlarged.top())?;
            }
        }

        Ok(())
    }
}
