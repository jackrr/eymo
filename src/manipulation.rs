use image::RgbImage;

// TODO: bake out the implementations for each operation i care about
// TODO: verify each on an input image -- verify nesting behavior as well
// TODO: define the confiuration language + build corresponding parser

// #[derive(Debug, Clone)]
// pub enum Operation {
//     Move,
//     Rotate,
//     Swap,
//     Flip,
//     Scale,
//     Repeat,
// }

#[derive(Debug, Clone)]
pub struct Target {
    // shape? rect?
    // examples: mouth, eye, bounding box, arbitrary list of coordinates to form a shape
}

#[derive(Debug, Clone)]
struct MoveArgs {
    dx: i32,
    src: Target,
    dest: Target,
}

#[derive(Debug, Clone)]
struct RotateArgs {
    theta: f32,
}

#[derive(Debug, Clone)]
enum Operation {
    Move(MoveArgs),
    Rotate(RotateArgs),
}

impl Operation {
    fn type_name(&self) -> &'static str {
        match self {
            Operation::Move(_) => "move",
            Operation::Rotate(_) => "rotate",
        }
    }

    fn run(&self, img: &mut RgbImage) {
        match self {
            Operation::Move(m) => {
                // TODO: do the move
            }
            Operation::Rotate(r) => {
                // TODO: do the rotation
            }
        }
    }
    fn roi(&self) {
        match self {
            Operation::Move(m) => {
                // TODO: return target
            }
            Operation::Rotate(r) => {
                // TODO: return src
            }
        }
    }
}

#[derive(Debug, Clone)]
struct Instruction {
    operation: Operation,
    nested_operations: Vec<Instruction>,
}
