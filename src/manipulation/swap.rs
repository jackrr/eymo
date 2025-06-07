use super::Executable;
use crate::manipulation::util;
use crate::shapes::{rect::Rect, shape::Shape};
use anyhow::Result;
use image::{GenericImage, RgbImage};

#[derive(Debug, Clone)]
pub struct Swap {
    a: Shape,
    b: Shape,
}

impl Swap {
    pub fn new(a: Shape, b: Shape) -> Swap {
        Swap { a, b }
    }
}

impl Executable for Swap {
    fn execute(&self, img: &mut RgbImage) -> Result<()> {
        match &self.a {
            Shape::Rect(a) => match &self.b {
                Shape::Rect(b) => {
                    swap_rects(img, *a, *b)?;
                }
                Shape::Polygon(b) => {
                    swap_rects(img, *a, (*b).clone())?;
                }
            },
            Shape::Polygon(a) => match &self.b {
                Shape::Rect(b) => {
                    swap_rects(img, (*a).clone(), *b)?;
                }
                Shape::Polygon(b) => {
                    // for (ap, bp) in a.iter_points_with_projection((*b).clone()) {
                    //     let bpx = img.get_pixel(bp.x, bp.y).clone();
                    //     let apx = img.get_pixel(ap.x, ap.y).clone();

                    //     img.put_pixel(bp.x, bp.y, apx);
                    //     img.put_pixel(ap.x, ap.y, bpx);
                    // }
                }
            },
        }

        Ok(())
    }
}

fn swap_rects(
    img: &mut RgbImage,
    a: impl Into<Rect> + Clone,
    b: impl Into<Rect> + Clone,
) -> Result<()> {
    // TODO: scaling
    // 1. scale smaller to the larger

    let a_bound: Rect = a.into();
    // 1. copy pixels w/in a
    let a_img = util::image_at(a_bound, img)?;

    // 2. write pixels from b to a
    let b_bound = b.into();
    img.copy_within(b_bound.into(), a_bound.left(), a_bound.top());
    // 3. write pixels from a to b
    img.copy_from(&a_img, b_bound.left(), b_bound.top())?;

    Ok(())
}
