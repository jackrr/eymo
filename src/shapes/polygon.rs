use super::point::Point;

use super::rect::Rect;

#[derive(Debug, Clone)]
pub struct Polygon {
    pub points: Vec<Point>,
}

impl Polygon {
    // Points assumed to be provided in clockwise order...
    pub fn new(points: Vec<Point>) -> Self {
        Self { points }
    }
}

impl From<Polygon> for Rect {
    fn from(poly: Polygon) -> Rect {
        let max_x = poly
            .points
            .iter()
            .fold(poly.points[0].x, |max, p| max.max(p.x));
        let min_x = poly
            .points
            .iter()
            .fold(poly.points[0].x, |min, p| min.min(p.x));
        let max_y = poly
            .points
            .iter()
            .fold(poly.points[0].y, |max, p| max.max(p.y));
        let min_y = poly
            .points
            .iter()
            .fold(poly.points[0].y, |min, p| min.min(p.y));

        Rect::from_tl(min_x, min_y, max_x - min_x, max_y - min_y)
    }
}
