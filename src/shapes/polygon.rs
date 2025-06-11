use super::point::{Point, Pointi32};
use log::debug;

use super::rect::Rect;
use super::shape::Shape;

#[derive(Debug, Clone)]
pub struct Polygon {
    pub points: Vec<Point>,
    points_i32: Vec<Pointi32>,
}

impl Polygon {
    pub fn new(points: Vec<Point>) -> Self {
        let points_i32 = points.iter().map(|p| p.clone().into()).collect();
        Self { points, points_i32 }
    }

    pub fn scale(&self, scale: f32, max_x: u32, max_y: u32) -> Self {
        let mut bound: Rect = self.clone().into();
        self.project(bound.scale(scale, max_x, max_y))
    }

    pub fn project(&self, project: impl Into<Shape>) -> Self {
        // remaps own coordinates to coordinate space of s
        let proj_rect: Rect = project.into().into();
        let self_rect = Rect::from(self.clone());

        let mut points = Vec::new();

        for p in &self.points {
            points.push(p.project(&self_rect, &proj_rect));
        }

        Self::new(points)
    }

    pub fn center(&self) -> Point {
        let left = self.min_x();
        let right = self.max_x();
        let top = self.min_y();
        let bottom = self.max_y();
        Point::new(
            left + ((right - left) as f32 / 2.).round() as u32,
            top + ((bottom - top) as f32 / 2.).round() as u32,
        )
    }

    pub fn rotate(&self, theta: f32) -> Self {
        let center = self.center();
        Self::new(
            self.points
                .iter()
                .map(|p| p.clone().rotate(center, theta))
                .collect(),
        )
    }

    pub fn min_x(&self) -> u32 {
        self.points.iter().map(|p| p.x).fold(u32::MAX, u32::min)
    }

    pub fn max_x(&self) -> u32 {
        self.points.iter().map(|p| p.x).fold(0, u32::max)
    }

    pub fn min_y(&self) -> u32 {
        self.points.iter().map(|p| p.y).fold(u32::MAX, u32::min)
    }

    pub fn max_y(&self) -> u32 {
        self.points.iter().map(|p| p.y).fold(0, u32::max)
    }

    // FYI contains_point (and underlying fns provided by Claude4.0)
    pub fn contains_point(&self, point: Point) -> bool {
        let point: Pointi32 = point.into();
        let n = self.points_i32.len();
        if n < 3 {
            return false;
        }

        // First check if point is on any edge or vertex
        if self.point_on_boundary(point) {
            return true;
        }

        // Ray casting algorithm for interior points
        let mut inside = false;
        let mut j = n - 1;

        for i in 0..n {
            let vi = self.points_i32[i];
            let vj = self.points_i32[j];

            if ((vi.y > point.y) != (vj.y > point.y)) && self.point_left_of_edge(point, vi, vj) {
                inside = !inside;
            }
            j = i;
        }

        inside
    }

    fn point_on_boundary(&self, point: Pointi32) -> bool {
        let n = self.points_i32.len();

        // Check if point is a vertex
        if self.points_i32.contains(&point) {
            return true;
        }

        // Check if point is on any edge
        for i in 0..n {
            let p1 = self.points_i32[i];
            let p2 = self.points_i32[(i + 1) % n];

            if self.point_on_edge(point, p1, p2) {
                return true;
            }
        }

        false
    }

    fn point_on_edge(&self, point: Pointi32, p1: Pointi32, p2: Pointi32) -> bool {
        // Check if point is collinear with p1 and p2
        let cross_product = (point.y - p1.y) * (p2.x - p1.x) - (point.x - p1.x) * (p2.y - p1.y);

        if cross_product != 0 {
            return false; // Not collinear
        }

        // Check if point is within the bounding box of the edge
        let min_x = p1.x.min(p2.x);
        let max_x = p1.x.max(p2.x);
        let min_y = p1.y.min(p2.y);
        let max_y = p1.y.max(p2.y);

        point.x >= min_x && point.x <= max_x && point.y >= min_y && point.y <= max_y
    }

    fn point_left_of_edge(&self, point: Pointi32, p1: Pointi32, p2: Pointi32) -> bool {
        // Use integer arithmetic to avoid floating point precision issues
        // Check if point.x < p1.x + (p2.x - p1.x) * (point.y - p1.y) / (p2.y - p1.y)
        // Rearranged to avoid division: (point.x - p1.x) * (p2.y - p1.y) < (p2.x - p1.x) * (point.y - p1.y)

        let dx = p2.x - p1.x;
        let dy = p2.y - p1.y;

        if dy == 0 {
            return false; // Horizontal edge, skip
        }

        let left_side = (point.x - p1.x) * dy;
        let right_side = dx * (point.y - p1.y);

        if dy > 0 {
            left_side < right_side
        } else {
            left_side > right_side
        }
    }

    // Iterate across rows, starting at "apex polygon with lowest Y"
    pub fn iter_inner_points(&self) -> PolygonInteriorIter {
        PolygonInteriorIter::new(self.clone())
    }

    pub fn iter_pairwise_projection_onto(&self, other: impl Into<Shape>) -> ProjectedPolygonIter {
        ProjectedPolygonIter::new(self.clone(), other)
    }
}

pub struct PolygonInteriorIter {
    last: Option<Point>,
    started: bool,
    min_x: u32,
    max_x: u32,
    max_y: u32,
    polygon: Polygon,
}

pub struct ProjectedPolygonIter {
    iter: PolygonInteriorIter,
    proj_rect: Rect,
    self_rect: Rect,
    inverted: bool,
}

impl ProjectedPolygonIter {
    // TODO: implement a likely faster algo; roughly:
    // 1. get all line segments between adjacent vertices
    // 2. For each "row" Y between min y and max y of poly:
    //    3. Sort line segments by intersections with Y
    //    4. Scan X vals from min x to max x, only giving back points in interior

    // Pairwise iteration of a projection and the original
    // Use larger as base iterator to prevent sparsity
    fn new(a: Polygon, b: impl Into<Shape>) -> Self {
        Self {
            iter: a.iter_inner_points(),
            self_rect: a.clone().into(),
            proj_rect: Rect::from(a.project(b)),
            inverted: false,
        }
    }

    pub fn invert(&mut self) {
        self.inverted = !self.inverted;
    }
}

impl Iterator for ProjectedPolygonIter {
    type Item = (Point, Point);

    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next() {
            Some(p) => {
                let other = p.project(&self.self_rect, &self.proj_rect);
                if self.inverted {
                    Some((other, p))
                } else {
                    Some((p, other))
                }
            }
            None => None,
        }
    }
}

impl PolygonInteriorIter {
    fn new(polygon: Polygon) -> Self {
        let points = polygon.points.clone();
        Self {
            min_x: polygon.min_x(),
            max_x: polygon.max_x(),
            max_y: polygon.max_y(),
            polygon,
            last: None,
            started: false,
        }
    }

    fn first_point(&self) -> Option<Point> {
        let res = self.polygon.points.iter().reduce(|smallest, p| {
            if (p.y == smallest.y && p.x < smallest.x) || p.y < smallest.y {
                p
            } else {
                smallest
            }
        });

        match res {
            Some(p) => p.clone().into(),
            None => None,
        }
    }

    fn point_after(&self, p: &mut Point) -> Option<Point> {
        // Scan L->R, skip row once not contained
        p.x += 1;
        while p.y <= self.max_y {
            if self.polygon.contains_point(*p) {
                return Some(*p);
            }
            p.x += 1;
            if p.x > self.max_x {
                p.y += 1;
                p.x = self.min_x;
            }
        }
        None
    }
}

impl Iterator for PolygonInteriorIter {
    type Item = Point;

    fn next(&mut self) -> Option<Self::Item> {
        let last = self.last.take();
        match last {
            Some(mut p) => match self.point_after(&mut p) {
                Some(p) => {
                    self.last = p.clone().into();
                    Some(p)
                }
                None => None,
            },
            None => {
                if self.started {
                    None
                } else {
                    self.started = true;
                    match self.first_point() {
                        Some(p) => {
                            self.last = p.clone().into();
                            p.into()
                        }
                        None => None,
                    }
                }
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::iter::zip;

    #[test]
    fn test_project() {
        let polygon = Polygon::new(Vec::from([
            Point::new(0, 0),
            Point::new(0, 1),
            Point::new(1, 1),
            Point::new(1, 0),
        ]));

        let actual = polygon.project(Rect::from_tl(1, 1, 1, 1));

        assert_eq!(actual.points[0], Point::new(1, 1));
        assert_eq!(actual.points[1], Point::new(1, 2));
        assert_eq!(actual.points[2], Point::new(2, 2));
        assert_eq!(actual.points[3], Point::new(2, 1));
    }

    #[test]
    fn test_project_bigger() {
        let polygon = Polygon::new(Vec::from([
            Point::new(0, 0),
            Point::new(0, 1),
            Point::new(1, 1),
            Point::new(1, 0),
        ]));

        let actual = polygon.project(Rect::from_tl(1, 1, 2, 2));

        assert_eq!(actual.points[0], Point::new(1, 1));
        assert_eq!(actual.points[1], Point::new(1, 3));
        assert_eq!(actual.points[2], Point::new(3, 3));
        assert_eq!(actual.points[3], Point::new(3, 1));
    }

    #[test]
    fn test_project_smaller() {
        let polygon = Polygon::new(Vec::from([
            Point::new(0, 0),
            Point::new(0, 2),
            Point::new(2, 2),
            Point::new(2, 0),
        ]));

        let actual = polygon.project(Rect::from_tl(1, 1, 1, 1));

        assert_eq!(actual.points[0], Point::new(1, 1));
        assert_eq!(actual.points[1], Point::new(1, 2));
        assert_eq!(actual.points[2], Point::new(2, 2));
        assert_eq!(actual.points[3], Point::new(2, 1));
    }

    #[test]
    fn test_project_poly() {
        let polygon = Polygon::new(Vec::from([
            Point::new(5, 0),
            Point::new(15, 0),
            Point::new(15, 5),
            Point::new(10, 10),
            Point::new(5, 5),
        ]));

        let actual = polygon.project(Rect::from_tl(50, 50, 50, 50));
        assert_eq!(actual.points[0], Point::new(50, 50));
        assert_eq!(actual.points[1], Point::new(100, 50));
        assert_eq!(actual.points[2], Point::new(100, 75));
        assert_eq!(actual.points[3], Point::new(75, 100));
        assert_eq!(actual.points[4], Point::new(50, 75));
    }

    #[test]
    fn test_contains_point() {
        let polygon = Polygon::new(Vec::from([
            Point::new(0, 0),
            Point::new(3, 0),
            Point::new(3, 3),
        ]));
        /*
        3 n n y
        2 n y y
        1 y y y
        0 1 2 3
        */

        assert!(polygon.contains_point(Point::new(0, 0)));
        assert!(polygon.contains_point(Point::new(2, 1)));
        assert!(polygon.contains_point(Point::new(1, 0)));
        assert!(polygon.contains_point(Point::new(1, 1)));
        assert!(polygon.contains_point(Point::new(2, 0)));

        assert!(polygon.contains_point(Point::new(2, 2)));
        assert!(polygon.contains_point(Point::new(3, 0)));
        assert!(polygon.contains_point(Point::new(3, 1)));
        assert!(polygon.contains_point(Point::new(3, 2)));
        assert!(polygon.contains_point(Point::new(3, 3)));

        assert!(!polygon.contains_point(Point::new(4, 4)));
        assert!(!polygon.contains_point(Point::new(3, 4)));
        assert!(!polygon.contains_point(Point::new(4, 3)));
        assert!(!polygon.contains_point(Point::new(0, 4)));
        assert!(!polygon.contains_point(Point::new(0, 1)));
        assert!(!polygon.contains_point(Point::new(1, 2)));
    }

    #[test]
    fn test_polygon_inner() {
        let polygon = Polygon::new(Vec::from([
            Point::new(0, 0),
            Point::new(0, 2),
            Point::new(3, 2),
        ]));
        let actual: Vec<Point> = polygon.iter_inner_points().collect();

        let expected = [
            Point::new(0, 0),
            Point::new(0, 1),
            Point::new(1, 1),
            Point::new(0, 2),
            Point::new(1, 2),
            Point::new(2, 2),
            Point::new(3, 2),
        ];

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in zip(actual, expected) {
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_rotate() {
        let polygon = Polygon::new(Vec::from([
            Point::new(0, 0),
            Point::new(2, 2),
            Point::new(2, 0),
        ]));

        let actual = polygon.rotate(90_f32.to_radians());
        let expected = [Point::new(0, 2), Point::new(2, 0), Point::new(0, 0)];

        for (actual, expected) in zip(actual.points.iter(), expected.iter()) {
            assert_eq!(actual, expected);
        }
    }
}
