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
        // TODO: support polygon operations
        // Right now this converts all polygons to containing rect

        // TODO: scale to match
        let a_bound: Rect = self.a.clone().into();

        // 1. copy pixels w/in a
        let a_img = util::image_at(a_bound, img)?;

        // 2. write pixels from b to a
        let b_bound: Rect = self.b.clone().into();
        img.copy_within(b_bound.into(), a_bound.left(), a_bound.top());
        // 3. write pixels from a to b
        img.copy_from(&a_img, b_bound.left(), b_bound.top())?;

        Ok(())
    }
}
