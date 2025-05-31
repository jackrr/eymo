use super::Executable;
use crate::shapes::{rect::Rect, shape::Shape};
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
        // TODO: support polygon operations
        // Right now this converts all polygons to containing rect

        // 1. copy pixels w/in a
        let src: Rect = self.src.clone().into();
        let dest: Rect = self.dest.clone().into();

        img.copy_within(src.into(), dest.left(), dest.top());

        Ok(())
    }
}
