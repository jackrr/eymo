use crate::shapes::shape::Shape;

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
        self.postion[0]
    }

    pub fn y(&self) -> f32 {
        self.postion[1]
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

    /*
     * Cast shape in pixel coordinate space with origin 0,0 to a
     * vector of Vertices for triangles covering the input shape in 2d
     * clip space to be used as input to a render shader.
     */
    pub fn triangles_for_shape(
        s: impl Into<Shape>,
        world_width: u32,
        world_height: u32,
    ) -> Vec<Self> {
        // cast x val to clip space
        let clip_x = |x: u32| x as f32 / world_width as f32 * 2. - 1.;

        // cast y val to clip space, including inverting axis
        let clip_y = |y: u32| 1. - y as f32 / world_height as f32 * 2.;

        let vertices = match s.into() {
            Shape::Polygon(p) => {
                let mut clockwise = p
                    .points
                    .iter()
                    .map(|p| Self::new(&[clip_x(p.x), clip_y(p.y)]))
                    .collect::<Vec<_>>();
                clockwise.reverse();
                clockwise
            }
            Shape::Rect(sr) => {
                let r = clip_x(sr.right());
                let l = clip_x(sr.left());
                let t = clip_y(sr.top());
                let b = clip_y(sr.bottom());
                Vec::from([
                    Self::new(&[r, t]),
                    Self::new(&[l, t]),
                    Self::new(&[l, b]),
                    Self::new(&[r, b]),
                ])
            }
        };

        Self::to_triangles(vertices)
    }

    pub fn to_triangles(list: Vec<Self>) -> Vec<Self> {
        let mut needed = list.len() - 2;
        let mut out_vert = Vec::new();
        let mut cur_idx = 0;
        while needed > 0 {
            for i in 0..3 {
                let idx = cur_idx + i;
                let idx = if idx < list.len() {
                    idx
                } else {
                    // use only even vertices on 2nd pass
                    (idx * 2) % list.len()
                };
                out_vert.push(list[idx].clone());
            }
            // walking even vertices
            cur_idx += 2;
            needed -= 1;
        }

        out_vert
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shapes::point::Point;
    use crate::shapes::polygon::Polygon;
    use crate::shapes::rect::Rect;

    #[test]
    fn test_triangles_for_rect() {
        let rect = Rect::from_tl(10, 0, 10, 10);

        let expected = Vec::from([
            Vertex::new(&[1., 1.]),
            Vertex::new(&[0., 1.]),
            Vertex::new(&[0., 0.]),
            Vertex::new(&[0., 0.]),
            Vertex::new(&[1., 0.]),
            Vertex::new(&[1., 1.]),
        ]);

        let actual = Vertex::triangles_for_shape(rect, 20, 20);

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_triangles_from_3_poly() {
        let poly = Polygon::new(Vec::from([
            Point::new(10, 10),
            Point::new(20, 10),
            Point::new(20, 20),
        ]));

        let expected = Vec::from([
            Vertex::new(&[1., -1.]),
            Vertex::new(&[1., 0.]),
            Vertex::new(&[0., 0.]),
        ]);

        let actual = Vertex::triangles_for_shape(poly, 20, 20);

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_triangles_from_4_poly() {
        let poly = Polygon::new(Vec::from([
            Point::new(10, 10),
            Point::new(20, 10),
            Point::new(20, 20),
            Point::new(10, 20),
        ]));

        let expected = Vec::from([
            Vertex::new(&[0., -1.]),
            Vertex::new(&[1., -1.]),
            Vertex::new(&[1., 0.]),
            Vertex::new(&[1., 0.]),
            Vertex::new(&[0., 0.]),
            Vertex::new(&[0., -1.]),
        ]);

        let actual = Vertex::triangles_for_shape(poly, 20, 20);

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_eq!(actual, expected);
        }
    }
}
