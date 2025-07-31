use crate::triangulate::Delaunator;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable, PartialEq)]
pub struct Vertex {
    pub position: [f32; 2],
    pub tex_coord: [f32; 2],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;

        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }

    pub fn new(coord: &[f32; 2]) -> Self {
        Self {
            position: coord.clone(),
            tex_coord: [0., 0.],
        }
    }

    pub fn new_with_tex(coord: &[f32; 2], tex_coord: &[f32; 2]) -> Self {
        Self {
            position: coord.clone(),
            tex_coord: tex_coord.clone(),
        }
    }

    pub fn x(&self) -> f32 {
        self.position[0]
    }

    pub fn y(&self) -> f32 {
        self.position[1]
    }

    pub fn triangles_for_full_coverage() -> Vec<Self> {
        [
            [1., 1.],
            [-1., 1.],
            [-1., -1.],
            [1., 1.],
            [-1., -1.],
            [1., -1.],
        ]
        .iter()
        .map(Self::new)
        .collect::<Vec<Self>>()
    }

    pub fn to_triangles(list: Vec<Self>) -> Vec<Self> {
        Delaunator::new(list).triangulate()
    }

    pub fn sub(&mut self, o: &Self) {
        self.position = [
            self.position[0] - o.position[0],
            self.position[1] - o.position[1],
        ];
        self.tex_coord = [
            self.tex_coord[0] - o.tex_coord[0],
            self.tex_coord[1] - o.tex_coord[1],
        ];
    }

    pub fn add(&mut self, o: &Self) {
        self.position = [
            self.position[0] + o.position[0],
            self.position[1] + o.position[1],
        ];
        self.tex_coord = [
            self.tex_coord[0] + o.tex_coord[0],
            self.tex_coord[1] + o.tex_coord[1],
        ];
    }

    pub fn mult_pos(&mut self, mag: f32) {
        self.position = [self.position[0] * mag, self.position[1] * mag];
    }
}
