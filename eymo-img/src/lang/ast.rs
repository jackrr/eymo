use crate::shapes::rect::Rect;
pub use crate::transform::FlipVariant;
use std::fmt;

#[derive(Debug)]
pub enum Statement {
    Transform(Transform),
}

#[derive(Debug)]
pub struct Transform {
    pub shape: Shape,
    pub operations: Vec<Operation>,
}

#[derive(Debug)]
pub enum Shape {
    FaceRef(FaceRef),
    Rect(Rect),
}

#[derive(Debug)]
pub struct FaceRef {
    pub part: FacePart,
    pub face_idx: Option<FaceIdx>,
}

impl fmt::Display for FaceRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.face_idx {
            Some(fi) => write!(f, "{}{}", self.part, fi),
            None => write!(f, "{}", self.part),
        }
    }
}

#[derive(Debug)]
pub enum FaceIdx {
    Absolute(u32),
    Relative(i32),
}

impl fmt::Display for FaceIdx {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Absolute(i) => write!(f, "abs{i}"),
            Self::Relative(i) => write!(f, "rel{i}"),
        }
    }
}

#[derive(Debug)]
pub enum FacePart {
    LEye,
    REye,
    LEyeRegion,
    REyeRegion,
    Face,
    Mouth,
    Nose,
    Forehead,
}

impl fmt::Display for FacePart {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug)]
pub enum Operation {
    Tile,
    Scale(f32),
    Rotate(f32),
    CopyTo(Vec<Shape>),
    SwapWith(Shape),
    Translate(i32, i32),
    Flip(FlipVariant),
    Drift(f32, f32),
    Spin(f32),
    Brightness(f32),
    Saturation(f32),
    Chans(f32, f32, f32),
}
