use super::Executable;
use crate::manipulation::util;
use crate::shapes::{rect::Rect, shape::Shape};
use anyhow::Result;
use image::imageops::{resize, FilterType};
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
                    // copy a -> pending
                    // copy b -> a
                    // copy pending -> b
                    swap_rects(img, *a, (*b).clone())?;
                }
            },
            Shape::Polygon(a) => match &self.b {
                Shape::Rect(b) => {
                    swap_rects(img, (*a).clone(), *b)?;
                }
                Shape::Polygon(b) => {
                    let a_rect: Rect = a.into();
                    let b_rect: Rect = b.into();
                    let a_img = util::image_at(a_rect, img);
                    let b_img = util::image_at(b_rect, img);
                    let a_img = resize(&a_img, b_rect.w, b_rect.h, FilterType::Triangle);
                    let b_img = resize(&b_img, a_rect.w, a_rect.h, FilterType::Triangle);

                    let a_scaled = a.resize(b_rect.w, b_rect.h);
                    let b_scaled = b.resize(a_rect.w, a_rect.h);

                    // TODO: finish this!! hurting head right now
                    // key takeaway that a_img has pixel data for b dest in target image, and so b_img has pixel data for a dest in target image
                    for (a, p) in a_scaled.iter_pairwise_projection_onto(b) {
                        img.put_pixel(p.x, p.y, b.get_pixel())
                    }

                    for (ap, bp) in a.iter_pairwise_projection_onto(b.clone()) {
                        let bpx = img.get_pixel(bp.x, bp.y).clone();
                        let apx = img.get_pixel(ap.x, ap.y).clone();

                        img.put_pixel(bp.x, bp.y, apx);
                        img.put_pixel(ap.x, ap.y, bpx);
                    }
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
