use super::rect::Rect;

#[derive(Debug, Clone, Copy)]
pub struct PointF32 {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Point {
    pub x: u32,
    pub y: u32,
}

impl Point {
    pub fn new(x: u32, y: u32) -> Point {
        Point { x, y }
    }

    pub fn rotate(&mut self, origin: Point, theta: f32) -> Point {
        let theta = -1. * theta;
        let x: f32 = (self.x as i32 - origin.x as i32) as f32;
        let y: f32 = (self.y as i32 - origin.y as i32) as f32;

        let rot_x = x * theta.cos() - y * theta.sin();
        let rot_y = x * theta.sin() + y * theta.cos();

        self.x = coerce_u32(rot_x + origin.x as f32);
        self.y = coerce_u32(rot_y + origin.y as f32);

        *self
    }

    pub fn project(self, src: &Rect, target: &Rect) -> Self {
        if src == target {
            return self;
        }

        let x_offset_pct = (self.x - src.left()) as f32 / src.w as f32;
        let y_offset_pct = (self.y - src.top()) as f32 / src.h as f32;

        Self {
            x: target.left() + (x_offset_pct * target.w as f32).round() as u32,
            y: target.top() + (y_offset_pct * target.h as f32).round() as u32,
        }
    }
}

fn coerce_u32(n: f32) -> u32 {
    if n < 0. {
        0
    } else {
        n.round() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotate() {
        let origin = Point::new(1, 1);
        let mut p = Point::new(2, 2);
        let clock90 = 90_f32.to_radians();
        let counter90 = (-90_f32).to_radians();

        assert_eq!(p.clone().rotate(origin, clock90), Point::new(2, 0));
        assert_eq!(p.rotate(origin, counter90), Point::new(0, 2));
    }

    #[test]
    fn test_project_center() {
        let p = Point::new(1, 1);
        let src = Rect::from_tl(0, 0, 2, 2);
        let dest = Rect::from_tl(0, 0, 4, 4);

        assert_eq!(p.project(&src, &dest), Point::new(2, 2));
    }

    #[test]
    fn test_project_corners() {
        let tl = Point::new(0, 0);
        let br = Point::new(2, 2);

        let src = Rect::from_tl(0, 0, 2, 2);
        let dest = Rect::from_tl(0, 0, 4, 4);

        assert_eq!(tl.project(&src, &dest), Point::new(0, 0));
        assert_eq!(br.project(&src, &dest), Point::new(4, 4));
    }

    #[test]
    fn test_project_inner() {
        let tl = Point::new(1, 1);
        let br = Point::new(3, 3);

        let src = Rect::from_tl(0, 0, 4, 4);
        let dest = Rect::from_tl(0, 0, 8, 8);

        assert_eq!(tl.project(&src, &dest), Point::new(2, 2));
        assert_eq!(br.project(&src, &dest), Point::new(6, 6));
    }
}
