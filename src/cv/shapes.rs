pub use imageproc::rect::Rect;

#[derive(Debug, Copy, Clone)]
pub enum FaceFeatureKind {
    LeftEar,
    RightEar,
    LeftEye,
    RightEye,
    Mouth,
    Nose,
}

#[derive(Debug, Copy, Clone)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

impl Point {
    pub fn new(x: i32, y: i32) -> Point {
        Point { x, y }
    }
}

#[derive(Debug, Clone)]
pub struct FaceFeature {
    pub bounds: Rect,
    pub kind: FaceFeatureKind,
}

impl FaceFeature {
    pub fn label(&self) -> &str {
        match self.kind {
            FaceFeatureKind::LeftEye => "left_eye",
            FaceFeatureKind::Mouth => "mouth",
            FaceFeatureKind::Nose => "nose",
            FaceFeatureKind::RightEye => "right_eye",
            FaceFeatureKind::LeftEar => "left_ear",
            FaceFeatureKind::RightEar => "right_ear",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Face {
    pub bounds: Rect,
    pub features: Vec<FaceFeature>,
}

impl Face {
    pub fn new(bounds: Rect) -> Face {
        Face {
            bounds,
            features: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Detection {
    pub face: Rect,
    pub keypoints: Vec<Keypoint>,
    pub confidence: f32,
}

#[derive(Debug, Copy, Clone)]
pub struct Keypoint {
    // 0 - nose, 1 - l eye, 2 r eye, 3 l ear, 4 r ear
    pub feature_idx: u8,
    pub point: Point,
    pub confidence: f32,
}

pub fn overlap_pct(r1: &Rect, r2: &Rect) -> f32 {
    let x_min = r1.left().max(r2.left());
    let x_max = r1.right().min(r2.right());
    let y_min = r1.top().max(r2.top());
    let y_max = r1.bottom().min(r2.bottom());

    let overlap_area = if x_min < x_max && y_min < y_max {
        (x_max - x_min) * (y_max - y_min)
    } else {
        0
    };

    let area_delta = area(r1) + area(r2) - overlap_area;

    if area_delta > 0 {
        overlap_area as f32 / area_delta as f32 * 100.
    } else {
        0.
    }
}

fn area(r: &Rect) -> i32 {
    (r.right() - r.left()) * (r.bottom() - r.top())
}

pub fn make_rect(center: Point, width: i32, height: i32) -> Rect {
    Rect::at(center.x - (width / 2), center.y - (height / 2)).of_size(width as u32, height as u32)
}
