use super::point::{Point, Pointi32};

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

    pub fn project(&self, s: impl Into<Shape>) -> Self {
        // remaps own coordinates to coordinate space of s
        let srect: Rect = s.into().into();
        let mrect = Rect::from(self.clone());

        let mut points = Vec::new();

        let dx = srect.left() as i32 - mrect.left() as i32;
        let dy = srect.top() as i32 - mrect.top() as i32;
        let x_scale = srect.w as f32 / mrect.w as f32;
        let y_scale = srect.h as f32 / mrect.h as f32;

        for p in &self.points {
            points.push(Point::new(
                (x_scale * (p.x as i32 + dx) as f32).round() as u32,
                (y_scale * (p.y as i32 + dy) as f32).round() as u32,
            ));
        }

        Self::new(points)
    }

    // FYI contains_point (and underlying fns provided by Claude4.0)
    fn contains_point(&self, point: Point) -> bool {
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

    pub fn iter_points_with_projection(&self, other: impl Into<Shape>) -> ProjectedPolygonIter {
        ProjectedPolygonIter::new(self.clone(), other)
    }

    pub fn iter_projected_points(&self, projection: &Polygon) -> ProjectedPolygonIter {
        ProjectedPolygonIter::from_projected_polygons(self.clone(), projection.clone())
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
    dx: i32,
    dy: i32,
    x_scale: f32,
    y_scale: f32,
}

impl ProjectedPolygonIter {
    // Pairwise iteration of a projection and the original
    // Uses larger as base iterator to prevent sparsity
    pub fn new(a: Polygon, b: impl Into<Shape>) -> Self {
        // TODO: conditionally set a/b depending on the "larger" of the two
        let other = a.project(b);
        let srect: Rect = a.clone().into();
        let mrect = Rect::from(other);

        let dx = mrect.left() as i32 - srect.left() as i32;
        let dy = mrect.top() as i32 - srect.top() as i32;
        let x_scale = srect.w as f32 / mrect.w as f32;
        let y_scale = srect.h as f32 / mrect.h as f32;
        Self {
            iter: a.iter_inner_points(),
            dx,
            dy,
            x_scale,
            y_scale,
        }
    }

    pub fn from_projected_polygons(a: Polygon, projection: Polygon) -> Self {
        let srect: Rect = a.clone().into();
        let mrect = Rect::from(projection);

        let dx = mrect.left() as i32 - srect.left() as i32;
        let dy = mrect.top() as i32 - srect.top() as i32;
        let x_scale = srect.w as f32 / mrect.w as f32;
        let y_scale = srect.h as f32 / mrect.h as f32;
        Self {
            iter: a.iter_inner_points(),
            dx,
            dy,
            x_scale,
            y_scale,
        }
    }
}

impl Iterator for ProjectedPolygonIter {
    type Item = (Point, Point);

    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next() {
            Some(p) => {
                let x = (self.x_scale * (p.x as i32 + self.dx) as f32).round() as u32;
                let y = (self.y_scale * (p.y as i32 + self.dy) as f32).round() as u32;

                Some((p, Point::new(x, y)))
            }
            None => None,
        }
    }
}

impl PolygonInteriorIter {
    fn new(polygon: Polygon) -> Self {
        let points = polygon.points.clone();
        Self {
            min_x: points.iter().map(|p| p.x).fold(u32::MAX, u32::min),
            max_x: points.iter().map(|p| p.x).fold(0, u32::max),
            // min_y: points.iter().map(|p| p.y).fold(u32::MAX, u32::min),
            max_y: points.iter().map(|p| p.y).fold(0, u32::max),
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
        while p.x < self.max_x || p.y < self.max_y {
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
}
