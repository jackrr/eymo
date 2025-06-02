use super::Executable;
use crate::shapes::{rect::Rect, shape::Shape};
use anyhow::Result;
use image::{GenericImage, GenericImageView, RgbImage};

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
        // TODO: support polygon operations
        // Right now this converts all polygons to containing rect

        // 1. copy pixels w/in a
        let a_bound: Rect = self.a.clone().into();
        let b_bound: Rect = self.b.clone().into();
        let a_view = *img.view(
            a_bound.x.try_into().unwrap(),
            a_bound.y.try_into().unwrap(),
            a_bound.w,
            a_bound.h,
        );
        let mut a_img = RgbImage::new(a_bound.w, a_bound.h);
        a_img.copy_from(&a_view, 0, 0)?;

        // 2. write pixels from b to a
        img.copy_within(b_bound.into(), a_bound.left(), a_bound.top());
        // 3. write pixels from a to b
        img.copy_from(&a_img, b_bound.left(), b_bound.top())?;

        Ok(())
    }
}
