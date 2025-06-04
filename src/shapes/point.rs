#[derive(Debug, Clone, Copy)]
pub struct PointF32 {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pointi32 {
    pub x: i32,
    pub y: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Point {
    pub x: u32,
    pub y: u32,
}

impl From<Point> for Pointi32 {
    fn from(p: Point) -> Pointi32 {
        Pointi32 {
            x: p.x as i32,
            y: p.y as i32,
        }
    }
}

impl Point {
    pub fn new(x: u32, y: u32) -> Point {
        Point { x, y }
    }

    pub fn rotate(&mut self, origin: Point, theta: f32) -> Point {
        // TODO: woof these typecasts are ugly... i32 as input? safer casts to/from f32?
        let x = (self.x as i32 - origin.x as i32) as f32;
        let y = (self.y as i32 - origin.y as i32) as f32;

        let rot_x = x * theta.cos() - y * theta.sin();
        let rot_y = x * theta.sin() + y * theta.cos();

        self.x = (rot_x.round() as i32 + origin.x as i32) as u32;
        self.y = (rot_y.round() as i32 + origin.y as i32) as u32;

        *self
    }
}

impl Pointi32 {
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}
