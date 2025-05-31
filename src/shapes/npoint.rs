use super::point::Point;

use super::rect::Rect;

#[derive(Debug, Clone)]
pub struct NPoint {
    pub points: Vec<Point>,
}

impl From<NPoint> for Rect {
    fn from(npoint: NPoint) -> Rect {
        let max_x = npoint
            .points
            .iter()
            .fold(npoint.points[0].x, |max, p| max.max(p.x));
        let min_x = npoint
            .points
            .iter()
            .fold(npoint.points[0].x, |min, p| min.min(p.x));
        let max_y = npoint
            .points
            .iter()
            .fold(npoint.points[0].y, |max, p| max.max(p.y));
        let min_y = npoint
            .points
            .iter()
            .fold(npoint.points[0].y, |min, p| min.min(p.y));

        Rect::from_tl(min_x, min_y, max_x - min_x, max_y - min_y)
    }
}
