use super::point::{Point, PointF32};
use super::polygon::Polygon;

// TODO: make rect generic to u32 or f32
#[derive(Debug, Copy, Clone)]
pub struct Rect {
    // centerpoint
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

#[derive(Debug, Copy, Clone)]
pub struct RectF32 {
    // centerpoint
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Into<Rect> for RectF32 {
    fn into(self) -> Rect {
        Rect {
            x: self.x.round() as u32,
            y: self.y.round() as u32,
            w: self.w.round() as u32,
            h: self.h.round() as u32,
        }
    }
}

#[allow(dead_code)]
impl RectF32 {
    pub fn from_center(xc: f32, yc: f32, w: f32, h: f32) -> RectF32 {
        RectF32 { x: xc, y: yc, w, h }
    }

    pub fn default() -> RectF32 {
        RectF32 {
            x: 1.,
            y: 1.,
            w: 1.,
            h: 1.,
        }
    }

    pub fn center(&self) -> PointF32 {
        PointF32 {
            x: self.x,
            y: self.y,
        }
    }

    pub fn adjust(&mut self, dx: f32, dy: f32, dw: f32, dh: f32) -> RectF32 {
        self.x += dx;
        self.y += dy;
        self.w = dw;
        self.h = dh;

        *self
    }

    pub fn scale(&mut self, scale_x: f32, scale_y: f32) -> RectF32 {
        self.x *= scale_x;
        self.y *= scale_y;
        self.w *= scale_x;
        self.h *= scale_y;

        *self
    }
}

impl Into<image::math::Rect> for Rect {
    fn into(self) -> image::math::Rect {
        image::math::Rect {
            x: self.left(),
            y: self.top(),
            width: self.w,
            height: self.h,
        }
    }
}

impl Into<imageproc::rect::Rect> for Rect {
    fn into(self) -> imageproc::rect::Rect {
        imageproc::rect::Rect::at(self.left() as i32, self.top() as i32).of_size(self.w, self.h)
    }
}

impl From<Rect> for Polygon {
    fn from(r: Rect) -> Polygon {
        Polygon::new(Vec::from([
            Point::new(r.left(), r.top()),
            Point::new(r.right(), r.top()),
            Point::new(r.right(), r.bottom()),
            Point::new(r.left(), r.bottom()),
        ]))
    }
}

#[allow(dead_code)]
impl Rect {
    pub fn left(&self) -> u32 {
        self.x - self.w / 2
    }
    pub fn right(&self) -> u32 {
        self.x + self.w / 2
    }
    pub fn top(&self) -> u32 {
        self.y - self.h / 2
    }
    pub fn bottom(&self) -> u32 {
        self.y + self.h / 2
    }
    pub fn area(&self) -> u32 {
        self.w * self.h
    }

    pub fn center(&self) -> Point {
        Point {
            x: self.x,
            y: self.y,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Self {
        self.w = width;
        self.h = height;

        *self
    }

    pub fn scale_x(&mut self, mag: f32, max: u32) -> Rect {
        // Ensure we don't go <0 on x axis
        let new_w = self.w as f32 * mag;
        let new_l = (self.x as f32 - new_w / 2.).max(0.).round() as u32;
        let new_r = ((self.x as f32 + new_w / 2.).round() as u32).min(max);

        self.w = new_r - new_l;
        self.x = new_l + self.w / 2;

        *self
    }

    pub fn scale_y(&mut self, mag: f32, max: u32) -> Rect {
        // Ensure we don't go <0 on x axis
        let new_h = self.h as f32 * mag;
        let new_t = (self.y as f32 - new_h / 2.).round().max(0.) as u32;
        let new_b = ((self.y as f32 + new_h / 2.).round() as u32).min(max);

        self.h = new_b - new_t;
        self.y = new_t + self.h / 2;

        *self
    }

    pub fn scale(&mut self, mag: f32, max_x: u32, max_y: u32) -> Rect {
        self.scale_x(mag, max_x);
        self.scale_y(mag, max_y);

        *self
    }

    pub fn from_tl(x: u32, y: u32, w: u32, h: u32) -> Rect {
        Rect {
            x: x + w / 2,
            y: y + h / 2,
            w,
            h,
        }
    }

    pub fn from_center(xc: u32, yc: u32, w: u32, h: u32) -> Rect {
        Rect { x: xc, y: yc, w, h }
    }

    pub fn overlap_pct(&self, other: &Rect) -> f32 {
        let x_min = self.left().max(other.left());
        let x_max = self.right().min(other.right());
        let y_min = self.top().max(other.top());
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
}
