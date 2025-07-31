use core::f32;

use crate::imggpu::vertex::Vertex;
use tracing::{span, Level};

/*
ISC License

Copyright (c) 2024, Mapbox

Permission to use, copy, modify, and/or distribute this software for any purpose
with or without fee is hereby granted, provided that the above copyright notice
and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
THIS SOFTWARE.



 ************

This is a port of Mapbox's Delauney triangulation algorithm to rust.
 */
pub struct Delaunator {
    points: Vec<Vertex>,
    triangles: Vec<usize>,
    half_edges: Vec<i32>,
    triangle_len: usize,
    hull_start: usize,
    hash_size: usize,
    pub hull: Vec<Vertex>,
    edge_stack: [usize; 512],
}

impl Delaunator {
    pub fn new(points: Vec<Vertex>) -> Self {
        let n = (points.len() * 2) >> 1;
        let max_triangles = (((2 * n) as i32) - 5).max(0);
        let mut half_edges = Vec::new();
        let mut triangles = Vec::new();

        for _ in 0..(max_triangles * 3) {
            half_edges.push(0);
            triangles.push(0);
        }

        Self {
            points,
            triangles,
            half_edges,
            hash_size: (n as f32).sqrt().ceil() as usize,
            triangle_len: 0,
            hull_start: 0,
            edge_stack: [0; 512],
            hull: Vec::with_capacity(n),
        }
    }

    pub fn triangulate(&mut self) -> Vec<Vertex> {
        let span = span!(Level::TRACE, "triangulate");
        let _guard = span.enter();

        let n = (self.points.len() * 2) >> 1;

        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;

        let mut ids = Vec::with_capacity(n);
        let mut dists = Vec::with_capacity(n);
        let mut hull_prev = Vec::with_capacity(n);
        let mut hull_next = Vec::with_capacity(n);
        let mut hull_hash = Vec::with_capacity(n);
        let mut hull_tri: Vec<usize> = Vec::with_capacity(n);

        for i in 0..n {
            let x = self.points[i].x();
            let y = self.points[i].y();
            if x < min_x {
                min_x = x;
            }

            if y < min_y {
                min_y = y;
            }
            if x > max_x {
                max_x = x;
            }
            if y > max_y {
                max_y = y;
            }
            ids.push(i);
            dists.push(0.);
            hull_prev.push(0);
            hull_next.push(0);
            hull_hash.push(0);
            hull_tri.push(0);
        }

        let cx = (min_x + max_x) / 2.;
        let cy = (min_y + max_y) / 2.;
        let c = Vertex::new(&[cx, cy]);

        // pick a seed point close to the center
        let mut v0: Option<Vertex> = None;
        let mut v0_idx = 0;
        let mut min_dist = f32::MAX;
        for (i, v) in self.points.iter().enumerate() {
            let d = dist(&c, &v);
            if d < min_dist {
                v0 = Some(v.clone());
                min_dist = d;
                v0_idx = i;
            }
        }
        let v0 = v0.unwrap();

        // find the point closest to the seed
        let mut v1: Option<Vertex> = None;
        let mut v1_idx = 0;
        min_dist = f32::MAX;
        for (i, v) in self.points.iter().enumerate() {
            if *v == v0 {
                continue;
            }
            let d = dist(&v0, &v);
            if d < min_dist && d > 0. {
                v1 = Some(v.clone());
                min_dist = d;
                v1_idx = i;
            }
        }
        let mut v1 = v1.unwrap();

        // find the third point which forms the smallest circumcircle with the first two
        let mut v2: Option<Vertex> = None;
        let mut v2_idx = 0;
        let mut min_radius = f32::MAX;
        for (i, v) in self.points.iter().enumerate() {
            if *v == v0 || *v == v1 {
                continue;
            }
            let r = circumradius(&v0, &v1, &v);
            if r < min_radius {
                v2 = Some(v.clone());
                min_radius = r;
                v2_idx = i;
            }
        }
        let mut v2 = v2.unwrap();

        if min_radius == f32::MAX {
            // order collinear points by dx (or dy if all x are identical)
            // and return the list as a hull
            let first_point = &self.points[0];
            for (i, v) in self.points.iter().enumerate() {
                let dx = v.x() - first_point.x();
                dists[i] = if dx != 0. {
                    dx
                } else {
                    v.y() - first_point.y()
                }
            }

            quicksort(&mut ids, &mut dists, 0, n - 1);
            let mut d0 = f32::MIN;
            for i in 0..n {
                let id = ids[i];
                let d = dists[id];
                if d > d0 {
                    d0 = d;
                }
            }

            return Vec::new();
        }

        // swap the order of the seed points for counter-clockwise orientation
        if orient2d(&v0, &v1, &v2) < 0. {
            let tmp = v1;
            let tmp_idx = v1_idx;
            v1 = v2;
            v1_idx = v2_idx;
            v2 = tmp;
            v2_idx = tmp_idx;
        }

        let center = circumcenter(&v0, &v1, &v2);

        for i in 0..n {
            dists[i] = dist(&self.points[i], &center);
        }

        // sort the points by distance from the seed triangle circumcenter
        quicksort(&mut ids, &mut dists, 0, n - 1);

        self.hull_start = v0_idx;
        let mut hull_size = 3;

        hull_prev[v2_idx] = v1_idx;
        hull_next[v0_idx] = v1_idx;

        hull_prev[v0_idx] = v2_idx;
        hull_next[v1_idx] = v2_idx;

        hull_prev[v1_idx] = v0_idx;
        hull_next[v2_idx] = v0_idx;

        hull_tri[v0_idx] = 0;
        hull_tri[v1_idx] = 1;
        hull_tri[v2_idx] = 2;

        for _ in 0..n {
            hull_hash.push(-1);
        }

        let key = self.hash_key(&v0, &center);
        hull_hash[key] = v0_idx as i32;
        let key = self.hash_key(&v1, &center);
        hull_hash[key] = v1_idx as i32;
        let key = self.hash_key(&v2, &center);
        hull_hash[key] = v2_idx as i32;

        self.add_triangle(v0_idx, v1_idx, v2_idx, -1, -1, -1);

        let mut xp = 0.;
        let mut yp = 0.;
        'id_iter: for k in 0..ids.len() {
            let i = ids[k];
            let v = self.points[i];
            let x = v.x();
            let y = v.y();

            // skip near-duplicate points
            if k > 0 && (x - xp).abs() <= f32::EPSILON && (y - yp).abs() <= f32::EPSILON {
                continue;
            }
            xp = x;
            yp = y;

            // skip seed triangle points
            if i == v0_idx || i == v1_idx || i == v2_idx {
                continue;
            }

            // find a visible edge on the convex hull using edge hash
            let mut start = 0;
            let key = self.hash_key(&v, &center);
            for j in 0..self.hash_size {
                start = hull_hash[(key + j) % self.hash_size];
                if start != -1 && start as usize != hull_next[start as usize] {
                    break;
                }
            }

            start = hull_prev[start as usize] as i32;
            let mut e = start as usize;
            loop {
                let q = hull_next[e];
                if orient2d(&v, &self.points[e], &self.points[q]) < 0. {
                    break;
                }
                e = q;
                if e == start as usize {
                    // likely a near-duplicate point; skip it
                    continue 'id_iter;
                }
            }

            // add the first triangle from the point
            let mut t = self.add_triangle(e, i, hull_next[e], -1, -1, hull_tri[e] as i32);

            // recursively flip triangles from the point until they satisfy the Delaunay condition
            hull_tri[i] = self.legalize(t + 2, &mut hull_tri, &mut hull_prev);
            hull_tri[e] = t; // keep track of boundary triangles on the hull
            hull_size += 1;

            // walk forward through the hull, adding more triangles and flipping recursively
            let mut n = hull_next[e];
            let mut q = hull_next[n];
            while orient2d(&v, &self.points[n], &self.points[q]) < 0. {
                t = self.add_triangle(n, i, q, hull_tri[i] as i32, -1, hull_tri[n] as i32);
                hull_tri[i] = self.legalize(t + 2, &mut hull_tri, &mut hull_prev);
                hull_next[n] = n; // mark as removed
                hull_size -= 1;
                n = q;
                q = hull_next[n];
            }

            // walk backward from the other side, adding more triangles and flipping
            if e as i32 == start {
                q = hull_prev[e];
                while orient2d(&v, &self.points[q], &self.points[e]) < 0. {
                    t = self.add_triangle(q, i, e, -1, hull_tri[e] as i32, hull_tri[q] as i32);
                    self.legalize(t + 2, &mut hull_tri, &mut hull_prev);
                    hull_tri[q] = t;
                    hull_next[e] = e; // mark as removed
                    hull_size -= 1;
                    e = q;
                    q = hull_prev[e];
                }
            }

            // update the hull indices
            hull_prev[i] = e;
            self.hull_start = e;
            hull_prev[n] = i;
            hull_next[e] = i;
            hull_next[i] = n;

            // save the two new edges in the hash table
            hull_hash[self.hash_key(&v, &c)] = i as i32;
            hull_hash[self.hash_key(&self.points[e], &c)] = e as i32;
        }

        self.hull = Vec::new();
        let mut e = self.hull_start;
        for _ in 0..hull_size {
            self.hull.push(self.points[e].clone());
            e = hull_next[e];
        }

        self.triangles[..self.triangle_len]
            .iter()
            .map(|idx| self.points[*idx].clone())
            .collect::<Vec<_>>()
    }

    fn hash_key(&self, v: &Vertex, c: &Vertex) -> usize {
        (self.hash_size as f32 * pseudo_angle(v.x() - c.x(), v.y() - c.y())).floor() as usize
            % self.hash_size
    }

    fn legalize(
        &mut self,
        a: usize,
        hull_tri: &mut Vec<usize>,
        hull_prev: &mut Vec<usize>,
    ) -> usize {
        let mut a = a;
        let mut i = 0;
        #[allow(unused_assignments)]
        let mut ar = 0;

        loop {
            let b = self.half_edges[a];

            /* if the pair of triangles doesn't satisfy the Delaunay condition
             * (p1 is inside the circumcircle of [p0, pl, pr]), flip them,
             * then do the same check/flip recursively for the new pair of triangles
             *
             *           pl                    pl
             *          /||\                  /  \
             *       al/ || \bl            al/    \a
             *        /  ||  \              /      \
             *       /  a||b  \    flip    /___ar___\
             *     p0\   ||   /p1   =>   p0\---bl---/p1
             *        \  ||  /              \      /
             *       ar\ || /br             b\    /br
             *          \||/                  \  /
             *           pr                    pr
             */
            let a0 = a - a % 3;
            ar = a0 + (a + 2) % 3;

            if b == -1 {
                // convex hull edge
                if i == 0 {
                    break;
                }
                i -= 1;
                a = self.edge_stack[i];
                continue;
            }

            let b0 = b - b % 3;
            let al = a0 + (a + 1) % 3;
            let bl = (b0 + (b + 2) % 3) as usize;

            let p0 = self.triangles[ar];
            let pr = self.triangles[a];
            let pl = self.triangles[al];
            let p1 = self.triangles[bl];

            let illegal = incircle(
                &self.points[p0],
                &self.points[pr],
                &self.points[pl],
                &self.points[p1],
            );

            if illegal {
                self.triangles[a] = p1;
                self.triangles[b as usize] = p0;

                let hbl = self.half_edges[bl];

                // edge swapped on the other side of the hull (rare); fix the halfedge reference
                if hbl == -1 {
                    let mut e = self.hull_start;
                    loop {
                        if hull_tri[e] == bl {
                            hull_tri[e] = a;
                            break;
                        }
                        e = hull_prev[e];
                        if e == self.hull_start {
                            break;
                        }
                    }
                }

                self.link(a as i32, hbl);
                self.link(b, self.half_edges[ar]);
                self.link(ar as i32, bl as i32);

                let br = b0 + (b + 1) % 3;

                // don't worry about hitting the cap: it can only happen on extremely degenerate input
                if i < self.edge_stack.len() {
                    self.edge_stack[i] = br as usize;
                    i += 1;
                }
            } else {
                if i == 0 {
                    break;
                }
                i -= 1;
                a = self.edge_stack[i];
            }
        }
        return ar;
    }

    // add a new triangle given vertex indices and adjacent half-edge ids
    fn add_triangle(&mut self, i0: usize, i1: usize, i2: usize, a: i32, b: i32, c: i32) -> usize {
        let t = self.triangle_len;

        self.triangles[t] = i0;
        self.triangles[t + 1] = i1;
        self.triangles[t + 2] = i2;

        self.link(t as i32, a);
        self.link(t as i32 + 1, b);
        self.link(t as i32 + 2, c);

        self.triangle_len += 3;

        t
    }

    fn link(&mut self, a: i32, b: i32) {
        self.half_edges[a as usize] = b;
        if b != -1 {
            self.half_edges[b as usize] = a;
        }
    }
}

// monotonically increases with real angle, but doesn't need expensive trigonometry
fn pseudo_angle(dx: f32, dy: f32) -> f32 {
    let p = dx / (dx.abs() + dy.abs());
    let t = if dy > 0. { 3. - p } else { 1. + p };

    t / 4.
}

fn dist(a: &Vertex, b: &Vertex) -> f32 {
    let dx = a.x() - b.x();
    let dy = a.y() - b.y();

    // println!("{a:?} {b:?} -- {dx}..{dy}, {}+{}", dx * dx, dy * dy);

    dx * dx + dy * dy
}

fn circumcenter(a: &Vertex, b: &Vertex, c: &Vertex) -> Vertex {
    let dx = b.x() - a.x();
    let dy = b.y() - a.y();
    let ex = c.x() - a.x();
    let ey = c.y() - a.y();

    let bl = dx * dx + dy * dy;
    let cl = ex * ex + ey * ey;

    let d = 0.5 / (dx * ey - dy * ex);

    Vertex::new(&[(a.x() * bl - dy * cl) + d, a.y() + (dx * cl - ex * bl) * d])
}

fn circumradius(a: &Vertex, b: &Vertex, c: &Vertex) -> f32 {
    let dx = b.x() - a.x();
    let dy = b.y() - a.y();
    let ex = c.x() - a.x();
    let ey = c.y() - a.y();

    let bl = dx * dx + dy * dy;
    let cl = ex * ex + ey * ey;
    let d = 0.5 / (dx * ey - dy * ex);

    let x = (ey * bl - dy * cl) * d;
    let y = (dx * cl - ex * bl) * d;

    x * x + y * y
}

fn quicksort(ids: &mut Vec<usize>, dists: &mut Vec<f32>, left: usize, right: usize) {
    if right - left <= 20 {
        for i in (left + 1)..(right + 1) {
            let tmp = ids[i];
            let tmp_dist = dists[tmp];
            let mut j = i - 1;
            while j >= left && dists[ids[j]] > tmp_dist {
                ids[j + 1] = ids[j];
                if j == 0 {
                    break;
                }
                j -= 1;
            }
            ids[j + 1] = tmp;
        }
    } else {
        let median = (left + right) >> 1;
        let mut i = left + 1;
        let mut j = right;
        swap(ids, median, i);
        if dists[ids[left]] > dists[ids[right]] {
            swap(ids, left, right);
        }
        if dists[ids[i]] > dists[ids[right]] {
            swap(ids, i, right);
        }
        if dists[ids[left]] > dists[ids[i]] {
            swap(ids, left, i);
        }

        let tmp = ids[i];
        let tmp_dist = dists[tmp];
        loop {
            loop {
                i += 1;
                if dists[ids[i]] >= tmp_dist {
                    break;
                }
            }
            loop {
                j -= 1;
                if dists[ids[j]] <= tmp_dist {
                    break;
                }
            }
            if j < i {
                break;
            }
            swap(ids, i, j);
        }
        ids[left + 1] = ids[j];
        ids[j] = tmp;

        if right - i + 1 >= j - left {
            quicksort(ids, dists, i, right);
            quicksort(ids, dists, left, j - 1);
        } else {
            quicksort(ids, dists, left, j - 1);
            quicksort(ids, dists, i, right);
        }
    }
}

fn swap(ids: &mut Vec<usize>, i: usize, j: usize) {
    let tmp = ids[i];
    ids[i] = ids[j];
    ids[j] = tmp;
}

fn orient2d(a: &Vertex, b: &Vertex, c: &Vertex) -> f64 {
    robust::orient2d(
        robust::Coord { x: a.x(), y: a.y() },
        robust::Coord { x: b.x(), y: b.y() },
        robust::Coord { x: c.x(), y: c.y() },
    )
}

fn incircle(a: &Vertex, b: &Vertex, c: &Vertex, p: &Vertex) -> bool {
    let ax = a.x();
    let ay = a.y();
    let bx = b.x();
    let by = b.y();
    let cx = c.x();
    let cy = c.y();
    let px = p.x();
    let py = p.y();

    let dx = ax - px;
    let dy = ay - py;
    let ex = bx - px;
    let ey = by - py;
    let fx = cx - px;
    let fy = cy - py;

    let ap = dx * dx + dy * dy;
    let bp = ex * ex + ey * ey;
    let cp = fx * fx + fy * fy;

    dx * (ey * cp - bp * fy) - dy * (ex * cp - bp * fx) + ap * (ex * fy - ey * fx) < 0.
}
