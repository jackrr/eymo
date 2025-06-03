use super::Executable;
use crate::shapes::{rect::Rect, shape::Shape};
use anyhow::{Error, Result};
use image::{GenericImage, GenericImageView, RgbImage};

#[derive(Debug, Clone)]
pub struct Rotate {
    target: Shape,
    theta: f32,
}

impl Rotate {
    pub fn new(target: Shape, theta: f32) -> Self {
        Self { target, theta }
    }
}

impl Executable for Rotate {
    fn execute(&self, img: &mut RgbImage) -> Result<()> {
        // TODO: support polygon operations
        // Right now this converts all polygons to containing rect
        Err(Error::msg("Rotate not implemented."))
    }
}
