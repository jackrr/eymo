use super::{util, GpuExecutable};
use crate::imggpu::resize::{resize_with_executor, GpuExecutor, ResizeAlgo};
use crate::shapes::{polygon::ProjectedPolygonIter, rect::Rect, shape::Shape};
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

impl GpuExecutable for Copy {
    fn execute(&self, gpu: &GpuExecutor, img: &mut RgbImage) -> Result<()> {
        let src_img = util::image_at(self.src.clone().into(), img)?;
        let dest_rect: Rect = self.dest.clone().into();
        let r_src =
            resize_with_executor(gpu, &src_img, dest_rect.w, dest_rect.h, ResizeAlgo::Linear)?;

        match &self.src {
            Shape::Rect(sr) => match &self.dest {
                Shape::Rect(dr) => {
                    img.copy_from(&r_src, dr.left(), dr.top())?;
                }
                Shape::Polygon(dp) => {
                    // TODO: verify this works
                    let sr = Rect::from_tl(0, 0, r_src.width(), r_src.height());
                    let mut iter = dp.iter_pairwise_projection_onto(sr);
                    iter.invert();
                    copy_pixels(r_src, img, iter);
                }
            },
            Shape::Polygon(sp) => match &self.dest {
                Shape::Rect(dr) => {
                    // TODO: verify this works
                    let res = sp.project(Rect::from_tl(0, 0, r_src.width(), r_src.height()));
                    copy_pixels(r_src, img, res.iter_pairwise_projection_onto(*dr));
                }
                Shape::Polygon(dp) => {
                    let res = sp.project(Rect::from_tl(0, 0, r_src.width(), r_src.height()));
                    copy_pixels(r_src, img, res.iter_pairwise_projection_onto(dp.clone()));
                }
            },
        }

        Ok(())
    }
}

fn copy_pixels(src: RgbImage, dest: &mut RgbImage, iter: ProjectedPolygonIter) {
    for (src_p, dest_p) in iter {
        if src_p.x >= src.width()
            || src_p.y >= src.height()
            || dest_p.x >= dest.width()
            || dest_p.y >= dest.height()
        {
            continue;
        }
        let pixel = src.get_pixel(src_p.x, src_p.y);
        dest.put_pixel(dest_p.x, dest_p.y, pixel.clone());
    }
}
