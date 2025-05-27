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
    pub x: u32,
    pub y: u32,
}

impl Point {
    pub fn new(x: u32, y: u32) -> Point {
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

#[derive(Debug, Copy, Clone)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Into<image::math::Rect> for Rect {
    fn into(self) -> image::math::Rect {
        image::math::Rect {
            x: self.x,
            y: self.y,
            width: self.width,
            height: self.height,
        }
    }
}

impl Into<imageproc::rect::Rect> for Rect {
    fn into(self) -> imageproc::rect::Rect {
        imageproc::rect::Rect::at(self.x as i32, self.y as i32).of_size(self.width, self.height)
    }
}

impl Rect {
    pub fn left(&self) -> u32 {
        self.x
    }
    pub fn right(&self) -> u32 {
        self.x + self.width
    }
    pub fn top(&self) -> u32 {
        self.y
    }
    pub fn bottom(&self) -> u32 {
        self.y + self.height
    }
    pub fn area(&self) -> u32 {
        (self.right() - self.left()) * (self.bottom() - self.top())
    }

    pub fn from_center(center: Point, width: u32, height: u32) -> Rect {
        Rect {
            x: center.x - (width / 2),
            y: center.y - (height / 2),
            width,
            height,
        }
    }

    pub fn overlap_pct(self: &Rect, other: &Rect) -> f32 {
        let x_min = self.x.max(other.x);
        let x_max = self.right().min(other.right());
        let y_min = self.y.max(other.y);
        let y_max = self.bottom().min(other.bottom());

        let overlap_area = if x_min < x_max && y_min < y_max {
            (x_max - x_min) * (y_max - y_min)
        } else {
            0
        };

        let area_delta = self.area() + other.area() - overlap_area;

        if area_delta > 0 {
            overlap_area as f32 / area_delta as f32 * 100.
        } else {
            0.
        }
    }
}
