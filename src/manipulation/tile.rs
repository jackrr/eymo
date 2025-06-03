use super::Executable;
use crate::manipulation::util;
use crate::shapes::{rect::Rect, shape::Shape};
use anyhow::Result;
use image::imageops::{resize, FilterType};
use image::{GenericImage, RgbImage};

#[derive(Debug, Clone)]
pub struct Tile {
    src: Shape,
    scale: f32,
}

impl Tile {
    pub fn new(src: Shape, scale: f32) -> Self {
        Self { src, scale }
    }
}

impl Executable for Tile {
    fn execute(&self, img: &mut RgbImage) -> Result<()> {
        let src: Rect = self.src.clone().into();
        let src_img = util::image_at(src, img)?;

        let tileable = if self.scale == 1. {
            src_img
        } else {
            resize(
                &src_img,
                (src_img.width() as f32 * self.scale).round() as u32,
                (src_img.height() as f32 * self.scale).round() as u32,
                FilterType::Triangle,
            )
        };

        let mut x_offset = 0;
        let mut y_offset = 0;
        let src_width = tileable.width();
        let src_height = tileable.height();
        let target_width = img.width();
        let target_height = img.height();

        while (y_offset + src_height) < target_height {
            img.copy_from(&tileable, x_offset, y_offset)?;
            x_offset += tileable.width();
            if (x_offset + src_width) > target_width {
                x_offset = 0;
                y_offset += tileable.height();
            }
        }

        Ok(())
    }
}
