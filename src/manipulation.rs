use anyhow::Result;
pub use copy::Copy;
use image::RgbImage;
pub use swap::Swap;

mod copy;
mod swap;

// TODO: bake out the implementations for each operation i care about
// TODO: verify each on an input image -- verify nesting behavior as well
// TODO: define the confiuration language + build corresponding parser

#[derive(Debug, Clone)]
pub enum Operation {
    Swap(Swap),
    Copy(Copy),
    // Move,
    // Rotate,
    // Flip,
    // Scale,
    // Repeat,
}

#[derive(Debug, Clone)]
pub struct OperationTree {
    op: Operation,
    sub_ops: Vec<OperationTree>,
}

impl From<Operation> for OperationTree {
    fn from(op: Operation) -> OperationTree {
        OperationTree {
            op,
            sub_ops: Vec::new(),
        }
    }
}

impl From<Swap> for Operation {
    fn from(s: Swap) -> Operation {
        Operation::Swap(s)
    }
}

impl From<Copy> for Operation {
    fn from(c: Copy) -> Operation {
        Operation::Copy(c)
    }
}

pub trait Executable {
    fn execute(&self, img: &mut RgbImage) -> Result<()>;
}

impl Executable for OperationTree {
    fn execute(&self, img: &mut RgbImage) -> Result<()> {
        match &self.op {
            Operation::Swap(s) => {
                s.execute(img)?;
            }
            Operation::Copy(c) => {
                c.execute(img)?;
            }
        }

        for op in &self.sub_ops {
            // TODO: scope to roi of self.operation
            op.execute(img)?;
        }

        Ok(())
    }
}
