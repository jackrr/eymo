use super::npoint::NPoint;
use super::rect::Rect;

#[derive(Debug, Clone)]
pub enum Shape {
    Rect(Rect),
    NPoint(NPoint),
}

impl From<Shape> for Rect {
    fn from(s: Shape) -> Rect {
        match s {
            Shape::Rect(s) => s,
            Shape::NPoint(s) => s.into(),
        }
    }
}

impl From<Rect> for Shape {
    fn from(r: Rect) -> Shape {
        Shape::Rect(r)
    }
}

impl From<NPoint> for Shape {
    fn from(n: NPoint) -> Shape {
        Shape::NPoint(n)
    }
}

impl Shape {
    // common logic here
}
