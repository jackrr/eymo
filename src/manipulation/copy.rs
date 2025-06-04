use super::Executable;
use crate::shapes::{polygon::Polygon, rect::Rect, shape::Shape};
use anyhow::Result;
use image::{GenericImage, RgbImage};

#[derive(Debug, Clone)]
pub struct Copy {
    src: Shape,
    dest: Shape,
}

impl Copy {
    pub fn new(src: Shape, dest: Shape) -> Copy {
        Copy { src, dest }
    }
}

impl Executable for Copy {
    fn execute(&self, img: &mut RgbImage) -> Result<()> {
        match &self.src {
            Shape::Rect(sr) => match &self.dest {
                Shape::Rect(dr) => {
                    img.copy_within((*sr).into(), dr.left(), dr.top());
                }
                Shape::Polygon(dp) => {
                    copy_pixels(img, &dp.project(*sr), dp);
                }
            },
            Shape::Polygon(sp) => match &self.dest {
                Shape::Rect(dr) => {
                    // project shape sp onto dr, pulling only those pixels into img
                    copy_pixels(img, sp, &sp.project(*dr));
                }
                Shape::Polygon(dp) => {
                    copy_pixels(img, sp, &sp.project(dp.clone()));
                }
            },
        }

        Ok(())
    }
}

fn copy_pixels(img: &mut RgbImage, src: &Polygon, dest: &Polygon) {
    for (src_p, dest_p) in src.iter_projected_points(dest) {
        let pixel = img.get_pixel(src_p.x, src_p.y);
        if dest_p.x >= img.width() || dest_p.y >= img.height() {
            // warn!("Invalid projected coordinate {x},{y}");
            continue;
        }
        img.put_pixel(dest_p.x, dest_p.y, pixel.clone());
    }
}
