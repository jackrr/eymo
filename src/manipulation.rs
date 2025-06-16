use anyhow::Result;
pub use copy::Copy;
pub use flip::Flip;
use image::RgbImage;
pub use rotate::Rotate;
pub use scale::Scale;
pub use swap::Swap;
pub use tile::Tile;

use crate::imggpu::resize::GpuExecutor;

mod copy;
mod flip;
mod rotate;
mod scale;
mod swap;
mod tile;
mod util;

#[derive(Debug, Clone)]
pub enum Operation {
    Copy(Copy),
    Flip(Flip),
    Rotate(Rotate),
    Scale(Scale),
    Swap(Swap),
    Tile(Tile),
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

impl From<Copy> for Operation {
    fn from(c: Copy) -> Operation {
        Operation::Copy(c)
    }
}

impl From<Flip> for Operation {
    fn from(c: Flip) -> Operation {
        Operation::Flip(c)
    }
}

impl From<Rotate> for Operation {
    fn from(c: Rotate) -> Operation {
        Operation::Rotate(c)
    }
}

impl From<Scale> for Operation {
    fn from(c: Scale) -> Operation {
        Operation::Scale(c)
    }
}

impl From<Swap> for Operation {
    fn from(s: Swap) -> Operation {
        Operation::Swap(s)
    }
}

impl From<Tile> for Operation {
    fn from(o: Tile) -> Operation {
        Operation::Tile(o)
    }
}

trait Executable {
    fn execute(&self, img: &mut RgbImage) -> Result<()>;
}

trait GpuExecutable {
    fn execute(&self, gpu: &GpuExecutor, img: &mut RgbImage) -> Result<()>;
}

impl OperationTree {
    pub fn execute(&self, gpu: &GpuExecutor, img: &mut RgbImage) -> Result<()> {
        match &self.op {
            Operation::Rotate(o) => {
                o.execute(img)?;
            }
            Operation::Flip(o) => {
                o.execute(img)?;
            }
            Operation::Scale(o) => {
                o.execute(gpu, img)?;
            }
            Operation::Tile(o) => {
                o.execute(img)?;
            }
            Operation::Swap(s) => {
                s.execute(img)?;
            }
            Operation::Copy(c) => {
                c.execute(img)?;
            }
        }

        for op in &self.sub_ops {
            // TODO: scope to roi of self.operation
            op.execute(gpu, img)?;
        }

        Ok(())
    }
}
