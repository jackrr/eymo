use super::point::Point;
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

    pub fn points(&self) -> Vec<Point> {
        match self {
            Shape::Polygon(p) => p.points.clone(),
            Shape::Rect(r) => r.points(),
        }
    }

    pub fn iter_projection_onto(&self, o: impl Into<Shape>) -> ShapeProjectionIter {
        ShapeProjectionIter::new(self.clone(), o)
    }
}

pub struct ShapeProjectionIter {
    points: Vec<Point>,
    src_rect: Rect,
    dest_rect: Rect,
    next_idx: usize,
}

impl ShapeProjectionIter {
    fn new(base: Shape, o: impl Into<Shape>) -> Self {
        Self {
            points: base.points(),
            src_rect: Rect::from(base),
            dest_rect: Rect::from(o.into()),
            next_idx: 0,
        }
    }
}

impl Iterator for ShapeProjectionIter {
    type Item = (Point, Point);

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_idx >= self.points.len() {
            return None;
        }
        let idx = self.next_idx;
        self.next_idx += 1;

        let p = self.points[idx];
        let other = p.project(&self.src_rect, &self.dest_rect);
        Some((p, other))
    }
}
