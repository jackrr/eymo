use super::Executable;
use crate::shapes::{rect::Rect, shape::Shape};
use anyhow::{Error, Result};
use image::{GenericImage, GenericImageView, RgbImage};

#[derive(Debug, Clone)]
pub struct Scale {
    target: Shape,
    zoom: f32,
}

impl Scale {
    pub fn new(target: Shape, zoom: f32) -> Self {
        Self { target, zoom }
    }
}

impl Executable for Scale {
    fn execute(&self, img: &mut RgbImage) -> Result<()> {
        // Expands feature by zoom level within image, taking over a
        // larger footprint
        // TODO: implement me
        Err(Error::msg("Scale not implemented."))
    }
}
