use super::polygon::Polygon;
use super::rect::Rect;

#[derive(Debug, Clone)]
pub enum Shape {
    Rect(Rect),
    Polygon(Polygon),
}

impl Default for Shape {
    fn default() -> Self {
        Shape::Rect(Rect::from_tl(0, 0, 1, 1))
    }
}

impl From<Shape> for Rect {
    fn from(s: Shape) -> Rect {
        match s {
            Shape::Rect(s) => s,
            Shape::Polygon(s) => s.into(),
        }
    }
}

impl From<Shape> for Polygon {
    fn from(s: Shape) -> Polygon {
        match s {
            Shape::Rect(s) => s.into(),
            Shape::Polygon(s) => s,
        }
    }
}

impl From<Rect> for Shape {
    fn from(r: Rect) -> Shape {
        Shape::Rect(r)
    }
}

impl From<Polygon> for Shape {
    fn from(n: Polygon) -> Shape {
        Shape::Polygon(n)
    }
}

impl Shape {
    pub fn scale(&mut self, mag: f32, max_x: u32, max_y: u32) {
        match self {
            Shape::Polygon(p) => {
                p.scale(mag, max_x, max_y);
            }
            Shape::Rect(r) => {
                r.scale(mag, max_x, max_y);
            }
        };
    }
}
