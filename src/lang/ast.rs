use crate::shapes::rect::Rect;
pub use crate::transform::FlipVariant;

// TODO: Add clear statement
// TODO: Add ability to invert shape (on transform and lang)
#[derive(Debug)]
pub struct Statement {
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
    pub face_idx: Option<u32>,
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
}

// TODO: Add fill operation + transform
#[derive(Debug)]
pub enum Operation {
    Tile,
    Scale(f32),
    Rotate(f32),
    WriteTo(Vec<Shape>),
    CopyTo(Vec<Shape>),
    SwapWith(Shape),
    Translate(i32, i32),
    Flip(FlipVariant),
}
