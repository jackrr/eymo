use super::{util, Executable};
use crate::shapes::{polygon::Polygon, rect::Rect, shape::Shape};
use anyhow::Result;
use image::RgbImage;
use log::warn;

#[derive(Debug, Clone)]
pub struct Rotate {
    target: Shape,
    theta: f32,
}

impl Rotate {
    pub fn new(target: Shape, deg: f32) -> Self {
        Self {
            target,
            theta: -deg.to_radians(), // images have inverted Y space
        }
    }
}

impl Executable for Rotate {
    fn execute(&self, img: &mut RgbImage) -> Result<()> {
        let src_img = util::image_at(self.target.clone().into(), img)?;
        let poly = Polygon::from(self.target.clone());
        let trans_x = poly.min_x();
        let trans_y = poly.min_y();
        let center = poly.center();

        let poly_src = poly.project(Rect::from_tl(
            0,
            0,
            src_img.width() - 1,
            src_img.height() - 1,
        ));
        let mut updated = Vec::new();
        for p in poly_src.iter_inner_points() {
            let mut rot_p = p.clone();
            rot_p.x += trans_x;
            rot_p.y += trans_y;
            let rot_p = rot_p.rotate(center, self.theta);

            if p.x >= src_img.width()
                || p.y >= src_img.height()
                || rot_p.x >= img.width()
                || rot_p.y >= img.height()
            {
                warn!("Skipping {p:?} -> {rot_p:?}");
                continue;
            }
            updated.push(rot_p);

            img.put_pixel(rot_p.x, rot_p.y, *src_img.get_pixel(p.x, p.y));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shapes::point::Point;
    use image::ImageReader;

    #[test]
    fn test_90_deg() -> Result<()> {
        let mut img = ImageReader::open("assets/steve.png")?.decode()?.into_rgb8();

        let mut orig_pixel_vals = Vec::new();
        for p in [
            Point::new(100, 100), // TL (becomes TR)
            Point::new(125, 150), // center -- stays same
            Point::new(149, 199), // BR (becomes BL)
            Point::new(149, 150), // midpoint on R (becomes midpoint on B)
        ] {
            orig_pixel_vals.push(img.get_pixel(p.x, p.y).clone());
        }

        // Src:
        // - Center 125, 150
        // - Width 50
        // - Height 100
        let target = Rect::from_tl(100, 100, 50, 100);
        let rot = Rotate::new(target.into(), 90_f32);
        rot.execute(&mut img)?;

        // Dest:
        // - Center: 125, 50
        // - width 100
        // - height 50
        for (idx, p) in [
            Point::new(175, 125), // TR
            Point::new(125, 150), // center
            Point::new(76, 174),  // BL
            Point::new(125, 174), // midpoint on B
        ]
        .iter()
        .enumerate()
        {
            assert_eq!(img.get_pixel(p.x, p.y).clone(), orig_pixel_vals[idx]);
        }

        Ok(())
    }
}
