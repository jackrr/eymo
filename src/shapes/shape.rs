use super::polygon::Polygon;
use super::rect::Rect;

#[derive(Debug, Clone)]
pub enum Shape {
    Rect(Rect),
    Polygon(Polygon),
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
    // common logic here
}
