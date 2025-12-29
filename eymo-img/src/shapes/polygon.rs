use super::point::Point;

use super::rect::Rect;

#[derive(Debug, Clone)]
pub struct Polygon {
    pub points: Vec<Point>,
}

impl Polygon {
    pub fn new(points: Vec<Point>) -> Self {
        Self { points }
    }

    pub fn center(&self) -> Point {
        let max_x = self.max_x();
        let min_x = self.min_x();
        let max_y = self.max_y();
        let min_y = self.min_y();

        Point {
            x: min_x + rounded_div(max_x - min_x, 2),
            y: min_y + rounded_div(max_y - min_y, 2),
        }
    }

    pub fn min_y(&self) -> u32 {
        self.points
            .iter()
            .fold(self.points[0].y, |min, p| min.min(p.y))
    }

    pub fn max_y(&self) -> u32 {
        self.points
            .iter()
            .fold(self.points[0].y, |max, p| max.max(p.y))
    }

    pub fn min_x(&self) -> u32 {
        self.points
            .iter()
            .fold(self.points[0].x, |min, p| min.min(p.x))
    }

    pub fn max_x(&self) -> u32 {
        self.points
            .iter()
            .fold(self.points[0].x, |max, p| max.max(p.x))
    }

    pub fn stretch(&mut self, mags: [f32; 4]) -> &mut Self {
        let [dxl, dxr, dyt, dyb] = mags;
        let center = self.center();

        for p in self.points.iter_mut() {
            if p.x < center.x {
                p.x = center.x - mult(center.x - p.x, dxl);
            }

            if p.x > center.x {
                p.x = center.x + mult(p.x - center.x, dxr);
            }

            if p.y < center.y {
                p.y = center.y - mult(center.y - p.y, dyt);
            }

            if p.y > center.y {
                p.y = center.y + mult(p.y - center.y, dyb);
            }
        }

        self
    }
}

impl From<Polygon> for Rect {
    fn from(poly: Polygon) -> Rect {
        let max_x = poly.max_x();
        let min_x = poly.min_x();
        let max_y = poly.max_y();
        let min_y = poly.min_y();

        Rect::from_tl(min_x, min_y, max_x - min_x, max_y - min_y)
    }
}

fn rounded_div(d: u32, q: u32) -> u32 {
    (d as f32 / q as f32).round() as u32
}

fn mult(v: u32, f: f32) -> u32 {
    (v as f32 * f).round() as u32
}

#[test]
fn test_rounded_div() {
    assert_eq!(rounded_div(10, 3), 3);
    assert_eq!(rounded_div(10, 4), 3);
    assert_eq!(rounded_div(5, 2), 3);
    assert_eq!(rounded_div(5, 4), 1);
    assert_eq!(rounded_div(20, 4), 5);
}
