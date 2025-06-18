use super::GpuExecutable;
use crate::imggpu::resize::{resize_with_executor, GpuExecutor, ResizeAlgo};
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

impl GpuExecutable for Swap {
    fn execute(&self, gpu: &mut GpuExecutor, img: &mut RgbImage) -> Result<()> {
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
                    let a_rect = Rect::from(a.clone());
                    let b_rect = Rect::from(b.clone());
                    let a_img = util::image_at(a_rect, img)?;
                    let b_img = util::image_at(b_rect, img)?;
                    let a_img =
                        resize_with_executor(gpu, &a_img, b_rect.w, b_rect.h, ResizeAlgo::Linear)?;
                    let b_img =
                        resize_with_executor(gpu, &b_img, a_rect.w, a_rect.h, ResizeAlgo::Linear)?;

                    let a_scaled = a.project(Rect::from_tl(0, 0, a_img.width(), a_img.height()));
                    let b_scaled = b.project(Rect::from_tl(0, 0, b_img.width(), b_img.height()));

                    util::copy_from(b_scaled, &b_img, a, img)?;
                    util::copy_from(a_scaled, &a_img, b, img)?;
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
