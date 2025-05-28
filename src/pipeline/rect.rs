use log::debug;

use crate::pipeline::Point;

#[derive(Debug, Copy, Clone)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl Into<image::math::Rect> for Rect {
    fn into(self) -> image::math::Rect {
        image::math::Rect {
            x: self.x,
            y: self.y,
            width: self.w,
            height: self.h,
        }
    }
}

impl Into<imageproc::rect::Rect> for Rect {
    fn into(self) -> imageproc::rect::Rect {
        imageproc::rect::Rect::at(self.x as i32, self.y as i32).of_size(self.w, self.h)
    }
}

impl Rect {
    pub fn left(&self) -> u32 {
        self.x
    }
    pub fn right(&self) -> u32 {
        self.x + self.w
    }
    pub fn top(&self) -> u32 {
        self.y
    }
    pub fn bottom(&self) -> u32 {
        self.y + self.h
    }
    pub fn area(&self) -> u32 {
        (self.right() - self.left()) * (self.bottom() - self.top())
    }

    pub fn new(x: u32, y: u32, w: u32, h: u32) -> Rect {
        Rect { x, y, w, h }
    }

    pub fn from_center(center: Point, w: u32, h: u32) -> Rect {
        Rect {
            x: center.x - (w / 2),
            y: center.y - (h / 2),
            w,
            h,
        }
    }

    pub fn overlap_pct(&self, other: &Rect) -> f32 {
        let x_min = self.x.max(other.x);
        let x_max = self.right().min(other.right());
        let y_min = self.y.max(other.y);
        let y_max = self.bottom().min(other.bottom());

        let overlap_area = if x_min < x_max && y_min < y_max {
            (x_max - x_min) * (y_max - y_min)
        } else {
            0
        };

        let area_delta = self.area() + other.area() - overlap_area;

        if area_delta > 0 {
            overlap_area as f32 / area_delta as f32 * 100.
        } else {
            0.
        }
    }

    pub fn adjust(&mut self, dx: i32, dy: i32, dw: i32, dh: i32) -> Rect {
        debug!("before adjust {:?}", self);
        self.x = (self.x as i32 + dx).abs() as u32;
        self.y = (self.y as i32 + dy).abs() as u32;
        self.w = (self.w as i32 + dw).abs() as u32;
        self.h = (self.h as i32 + dh).abs() as u32;
        debug!("after adjust {:?}", self);

        *self
    }

    pub fn scale(&mut self, scale_x: f32, scale_y: f32) -> Rect {
        debug!("before scale {:?}", self);
        self.x = (self.x as f32 * scale_x).round() as u32;
        self.y = (self.y as f32 * scale_y).round() as u32;
        self.w = (self.w as f32 * scale_x).round() as u32;
        self.h = (self.h as f32 * scale_y).round() as u32;
        debug!("after scale {:?}", self);

        *self
    }
}
