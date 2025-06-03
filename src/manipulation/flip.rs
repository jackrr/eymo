use super::Executable;
use crate::shapes::{rect::Rect, shape::Shape};
use anyhow::{Error, Result};
use image::{GenericImage, GenericImageView, RgbImage};

#[derive(Debug, Clone, Copy)]
pub enum Variant {
    Vertical,
    Horizontal,
}

#[derive(Debug, Clone)]
pub struct Flip {
    target: Shape,
    variant: Variant,
}

impl Flip {
    pub fn new(target: Shape, variant: Variant) -> Flip {
        Flip { target, variant }
    }
}

impl Executable for Flip {
    fn execute(&self, img: &mut RgbImage) -> Result<()> {
        // TODO: support polygon operations
        // Right now this converts all polygons to containing rect
        Err(Error::msg("Flip not implemented."))
    }
}
