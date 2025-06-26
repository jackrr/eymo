use core::f32;

use crate::imggpu::vertex::Vertex;

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
    edge_stack: Vec<u32>,
    triangles: Vec<usize>,
    hash_size: usize,
    hull_prev: Vec<usize>,
    hull_next: Vec<usize>,
    hull_tri: Vec<u32>,
    half_edges: Vec<i32>,
    hull_hash: Vec<i32>,
    hull: Vec<usize>,
    ids: Vec<usize>,
    dists: Vec<f32>,
}

impl Delaunator {
    pub fn new(points: Vec<Vertex>) -> Self {
        let n = (points.len() >> 1) as f32;
        Self {
            points,
            edge_stack: Vec::new(),
            hash_size: n.sqrt().ceil() as usize,
            triangles: Vec::new(),
            half_edges: Vec::new(),
            hull_prev: Vec::new(),
            hull_next: Vec::new(),
            hull_tri: Vec::new(),
            hull_hash: Vec::new(),
            hull: Vec::new(),
            ids: Vec::new(),
            dists: Vec::new(),
        }
    }

    pub fn triangulate(&mut self) {
        let n = self.points.len() >> 1;

        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = -f32::MIN;
        let mut max_y = -f32::MIN;

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
            self.ids[i] = i;
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
        let mut v0 = v0.unwrap();

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
                self.dists[i] = if dx != 0. {
                    dx
                } else {
                    v.y() - first_point.y()
                }
            }

            quicksort(&mut self.ids, &mut self.dists, 0, n - 1);
            let mut d0 = f32::MIN;
            for i in 0..n {
                let id = self.ids[i];
                let d = self.dists[id];
                if d > d0 {
                    self.hull.push(id);
                    d0 = d;
                }
            }

            return;
        }

        // swap the order of the seed points for counter-clockwise orientation
        if robust::orient2d(
            robust::Coord {
                x: v0.x(),
                y: v0.y(),
            },
            robust::Coord {
                x: v1.x(),
                y: v1.y(),
            },
            robust::Coord {
                x: v2.x(),
                y: v2.y(),
            },
        ) < 0.
        {
            let tmp = v0;
            v0 = v1;
            v1 = v2;
            v2 = tmp;
        }

        let center = circumcenter(&v0, &v1, &v2);

        for (i, v) in self.points.iter().enumerate() {
            self.dists[i] = dist(v, &center);
        }

        // sort the points by distance from the seed triangle circumcenter
        quicksort(&mut self.ids, &mut self.dists, 0, n - 1);
        let hull_start = v0_idx;
        let hull_size = 3;

        self.hull_prev[v2_idx] = v1_idx;
        self.hull_next[v0_idx] = v1_idx;

        self.hull_prev[v0_idx] = v2_idx;
        self.hull_next[v1_idx] = v2_idx;

        self.hull_prev[v1_idx] = v0_idx;
        self.hull_next[v2_idx] = v0_idx;

        self.hull_tri[v0_idx] = 0;
        self.hull_tri[v1_idx] = 1;
        self.hull_tri[v2_idx] = 2;

        for _ in 0..n {
            self.hull_hash.push(-1);
        }

        let key = self.hash_key(&v0, &center);
        self.hull_hash[key] = v0_idx as i32;
        let key = self.hash_key(&v1, &center);
        self.hull_hash[key] = v1_idx as i32;
        let key = self.hash_key(&v2, &center);
        self.hull_hash[key] = v2_idx as i32;

        let mut triangle_len = 0;
        triangle_len = self.add_triangle(triangle_len, v0_idx, v1_idx, v2_idx, -1, -1, -1);

        for k in 0..self.ids.len() {
            let i = self.ids[k];
            let v = self.points[i];
        }
    }

    fn hash_key(&self, v: &Vertex, c: &Vertex) -> usize {
        (self.hash_size as f32 * pseudo_angle(v.x() - c.x(), v.y() - c.y())).floor() as usize
            % self.hash_size
    }

    // add a new triangle given vertex indices and adjacent half-edge ids
    fn add_triangle(
        &mut self,
        t: usize,
        i0: usize,
        i1: usize,
        i2: usize,
        a: i32,
        b: i32,
        c: i32,
    ) -> usize {
        self.triangles[t] = i0;
        self.triangles[t + 1] = i1;
        self.triangles[t + 2] = i2;

        self.link(t as i32, a);
        self.link(t as i32 + 1, b);
        self.link(t as i32 + 2, c);

        t + 3
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
        for i in left + 1..right + 1 {
            let tmp = ids[i];
            let tmp_dist = dists[tmp];
            let mut j = i - 1;
            while j >= left && dists[ids[j]] > tmp_dist {
                ids[j + 1] = ids[j - 1];
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

// export default class Delaunator {

// constructor(coords) {
// this.coords = coords;

// arrays that will store the triangulation graph
// const maxTriangles = Math.max(2 * n - 5, 0);
// this._triangles = new Uint32Array(maxTriangles * 3);
// this._halfedges = new Int32Array(maxTriangles * 3);

// temporary arrays for tracking the edges of the advancing convex hull
// this._hashSize = Math.ceil(Math.sqrt(n));
// this._hullPrev = new Uint32Array(n); // edge to prev edge
// this._hullNext = new Uint32Array(n); // edge to next edge
// this._hullTri = new Uint32Array(n); // edge to adjacent triangle
// this._hullHash = new Int32Array(this._hashSize); // angular edge hash

// temporary arrays for sorting points
// this._ids = new Uint32Array(n);
// this._dists = new Float64Array(n);

// this.update();
// }

// update() {
// const {coords, _hullPrev: hullPrev, _hullNext: hullNext, _hullTri: hullTri, _hullHash: hullHash} =  this;
// const n = coords.length >> 1;

// populate an array of point indices; calculate input data bbox
//         let minX = Infinity;
//         let minY = Infinity;
//         let maxX = -Infinity;
//         let maxY = -Infinity;

//         for (let i = 0; i < n; i++) {
//             const x = coords[2 * i];
//             const y = coords[2 * i + 1];
//             if (x < minX) minX = x;
//             if (y < minY) minY = y;
//             if (x > maxX) maxX = x;
//             if (y > maxY) maxY = y;
//             this._ids[i] = i;
//         }
//         const cx = (minX + maxX) / 2;
//         const cy = (minY + maxY) / 2;

//         let i0, i1, i2;

//         // pick a seed point close to the centero
//         for (let i = 0, minDist = Infinity; i < n; i++) {
//             const d = dist(cx, cy, coords[2 * i], coords[2 * i + 1]);
//             if (d < minDist) {
//                 i0 = i;
//                 minDist = d;
//             }
//         }
//         const i0x = coords[2 * i0];
//         const i0y = coords[2 * i0 + 1];

//         // find the point closest to the seed
//         for (let i = 0, minDist = Infinity; i < n; i++) {
//             if (i === i0) continue;
//             const d = dist(i0x, i0y, coords[2 * i], coords[2 * i + 1]);
//             if (d < minDist && d > 0) {
//                 i1 = i;
//                 minDist = d;
//             }
//         }
//         let i1x = coords[2 * i1];
//         let i1y = coords[2 * i1 + 1];

//         let minRadius = Infinity;

//         // find the third point which forms the smallest circumcircle with the first two
//         for (let i = 0; i < n; i++) {
//             if (i === i0 || i === i1) continue;
//             const r = circumradius(i0x, i0y, i1x, i1y, coords[2 * i], coords[2 * i + 1]);
//             if (r < minRadius) {
//                 i2 = i;
//                 minRadius = r;
//             }
//         }
//         let i2x = coords[2 * i2];
//         let i2y = coords[2 * i2 + 1];

//         if (minRadius === Infinity) {
//             // order collinear points by dx (or dy if all x are identical)
//             // and return the list as a hull
//             for (let i = 0; i < n; i++) {
//                 this._dists[i] = (coords[2 * i] - coords[0]) || (coords[2 * i + 1] - coords[1]);
//             }
//             quicksort(this._ids, this._dists, 0, n - 1);
//             const hull = new Uint32Array(n);
//             let j = 0;
//             for (let i = 0, d0 = -Infinity; i < n; i++) {
//                 const id = this._ids[i];
//                 const d = this._dists[id];
//                 if (d > d0) {
//                     hull[j++] = id;
//                     d0 = d;
//                 }
//             }
//             this.hull = hull.subarray(0, j);
//             this.triangles = new Uint32Array(0);
//             this.halfedges = new Uint32Array(0);
//             return;
//         }

//         // swap the order of the seed points for counter-clockwise orientation
//         if (orient2d(i0x, i0y, i1x, i1y, i2x, i2y) < 0) {
//             const i = i1;
//             const x = i1x;
//             const y = i1y;
//             i1 = i2;
//             i1x = i2x;
//             i1y = i2y;
//             i2 = i;
//             i2x = x;
//             i2y = y;
//         }

//         const center = circumcenter(i0x, i0y, i1x, i1y, i2x, i2y);
//         this._cx = center.x;
//         this._cy = center.y;

//         for (let i = 0; i < n; i++) {
//             this._dists[i] = dist(coords[2 * i], coords[2 * i + 1], center.x, center.y);
//         }

//         // sort the points by distance from the seed triangle circumcenter
//         quicksort(this._ids, this._dists, 0, n - 1);

//         // set up the seed triangle as the starting hull
//         this._hullStart = i0;
//         let hullSize = 3;

//         hullNext[i0] = hullPrev[i2] = i1;
//         hullNext[i1] = hullPrev[i0] = i2;
//         hullNext[i2] = hullPrev[i1] = i0;

//         hullTri[i0] = 0;
//         hullTri[i1] = 1;
//         hullTri[i2] = 2;

//         hullHash.fill(-1);
//         hullHash[this._hashKey(i0x, i0y)] = i0;
//         hullHash[this._hashKey(i1x, i1y)] = i1;
//         hullHash[this._hashKey(i2x, i2y)] = i2;

//         this.trianglesLen = 0;
//         this._addTriangle(i0, i1, i2, -1, -1, -1);

//         for (let k = 0, xp, yp; k < this._ids.length; k++) {
//             const i = this._ids[k];
//             const x = coords[2 * i];
//             const y = coords[2 * i + 1];

//             // skip near-duplicate points
// f64::EPSILON
//             if (k > 0 && Math.abs(x - xp) <= EPSILON && Math.abs(y - yp) <= EPSILON) continue;
//             xp = x;
//             yp = y;

//             // skip seed triangle points
//             if (i === i0 || i === i1 || i === i2) continue;

//             // find a visible edge on the convex hull using edge hash
//             let start = 0;
//             for (let j = 0, key = this._hashKey(x, y); j < this._hashSize; j++) {
//                 start = hullHash[(key + j) % this._hashSize];
//                 if (start !== -1 && start !== hullNext[start]) break;
//             }

//             start = hullPrev[start];
//             let e = start, q;
//             while (q = hullNext[e], orient2d(x, y, coords[2 * e], coords[2 * e + 1], coords[2 * q], coords[2 * q + 1]) >= 0) {
//                 e = q;
//                 if (e === start) {
//                     e = -1;
//                     break;
//                 }
//             }
//             if (e === -1) continue; // likely a near-duplicate point; skip it

//             // add the first triangle from the point
//             let t = this._addTriangle(e, i, hullNext[e], -1, -1, hullTri[e]);

//             // recursively flip triangles from the point until they satisfy the Delaunay condition
//             hullTri[i] = this._legalize(t + 2);
//             hullTri[e] = t; // keep track of boundary triangles on the hull
//             hullSize++;

//             // walk forward through the hull, adding more triangles and flipping recursively
//             let n = hullNext[e];
//             while (q = hullNext[n], orient2d(x, y, coords[2 * n], coords[2 * n + 1], coords[2 * q], coords[2 * q + 1]) < 0) {
//                 t = this._addTriangle(n, i, q, hullTri[i], -1, hullTri[n]);
//                 hullTri[i] = this._legalize(t + 2);
//                 hullNext[n] = n; // mark as removed
//                 hullSize--;
//                 n = q;
//             }

//             // walk backward from the other side, adding more triangles and flipping
//             if (e === start) {
//                 while (q = hullPrev[e], orient2d(x, y, coords[2 * q], coords[2 * q + 1], coords[2 * e], coords[2 * e + 1]) < 0) {
//                     t = this._addTriangle(q, i, e, -1, hullTri[e], hullTri[q]);
//                     this._legalize(t + 2);
//                     hullTri[q] = t;
//                     hullNext[e] = e; // mark as removed
//                     hullSize--;
//                     e = q;
//                 }
//             }

//             // update the hull indices
//             this._hullStart = hullPrev[i] = e;
//             hullNext[e] = hullPrev[n] = i;
//             hullNext[i] = n;

//             // save the two new edges in the hash table
//             hullHash[this._hashKey(x, y)] = i;
//             hullHash[this._hashKey(coords[2 * e], coords[2 * e + 1])] = e;
//         }

//         this.hull = new Uint32Array(hullSize);
//         for (let i = 0, e = this._hullStart; i < hullSize; i++) {
//             this.hull[i] = e;
//             e = hullNext[e];
//         }

//         // trim typed triangle mesh arrays
//         this.triangles = this._triangles.subarray(0, this.trianglesLen);
//         this.halfedges = this._halfedges.subarray(0, this.trianglesLen);
//     }

//     _hashKey(x, y) {
//         return Math.floor(pseudoAngle(x - this._cx, y - this._cy) * this._hashSize) % this._hashSize;
//     }

//     _legalize(a) {
//         const {_triangles: triangles, _halfedges: halfedges, coords} = this;

//         let i = 0;
//         let ar = 0;

//         // recursion eliminated with a fixed-size stack
//         while (true) {
//             const b = halfedges[a];

//             /* if the pair of triangles doesn't satisfy the Delaunay condition
//              * (p1 is inside the circumcircle of [p0, pl, pr]), flip them,
//              * then do the same check/flip recursively for the new pair of triangles
//              *
//              *           pl                    pl
//              *          /||\                  /  \
//              *       al/ || \bl            al/    \a
//              *        /  ||  \              /      \
//              *       /  a||b  \    flip    /___ar___\
//              *     p0\   ||   /p1   =>   p0\---bl---/p1
//              *        \  ||  /              \      /
//              *       ar\ || /br             b\    /br
//              *          \||/                  \  /
//              *           pr                    pr
//              */
//             const a0 = a - a % 3;
//             ar = a0 + (a + 2) % 3;

//             if (b === -1) { // convex hull edge
//                 if (i === 0) break;
//                 a = EDGE_STACK[--i];
//                 continue;
//             }

//             const b0 = b - b % 3;
//             const al = a0 + (a + 1) % 3;
//             const bl = b0 + (b + 2) % 3;

//             const p0 = triangles[ar];
//             const pr = triangles[a];
//             const pl = triangles[al];
//             const p1 = triangles[bl];

//             const illegal = inCircle(
//                 coords[2 * p0], coords[2 * p0 + 1],
//                 coords[2 * pr], coords[2 * pr + 1],
//                 coords[2 * pl], coords[2 * pl + 1],
//                 coords[2 * p1], coords[2 * p1 + 1]);

//             if (illegal) {
//                 triangles[a] = p1;
//                 triangles[b] = p0;

//                 const hbl = halfedges[bl];

//                 // edge swapped on the other side of the hull (rare); fix the halfedge reference
//                 if (hbl === -1) {
//                     let e = this._hullStart;
//                     do {
//                         if (this._hullTri[e] === bl) {
//                             this._hullTri[e] = a;
//                             break;
//                         }
//                         e = this._hullPrev[e];
//                     } while (e !== this._hullStart);
//                 }
//                 this._link(a, hbl);
//                 this._link(b, halfedges[ar]);
//                 this._link(ar, bl);

//                 const br = b0 + (b + 1) % 3;

//                 // don't worry about hitting the cap: it can only happen on extremely degenerate input
//                 if (i < EDGE_STACK.length) {
//                     EDGE_STACK[i++] = br;
//                 }
//             } else {
//                 if (i === 0) break;
//                 a = EDGE_STACK[--i];
//             }
//         }

//         return ar;
//     }

//     _link(a, b) {
//         this._halfedges[a] = b;
//         if (b !== -1) this._halfedges[b] = a;
//     }

//     // add a new triangle given vertex indices and adjacent half-edge ids
//     _addTriangle(i0, i1, i2, a, b, c) {
//         const t = this.trianglesLen;

//         this._triangles[t] = i0;
//         this._triangles[t + 1] = i1;
//         this._triangles[t + 2] = i2;

//         this._link(t, a);
//         this._link(t + 1, b);
//         this._link(t + 2, c);

//         this.trianglesLen += 3;

//         return t;
//     }
// }

// // monotonically increases with real angle, but doesn't need expensive trigonometry
// function pseudoAngle(dx, dy) {
//     const p = dx / (Math.abs(dx) + Math.abs(dy));
//     return (dy > 0 ? 3 - p : 1 + p) / 4; // [0..1]
// }

// function dist(ax, ay, bx, by) {
//     const dx = ax - bx;
//     const dy = ay - by;
//     return dx * dx + dy * dy;
// }

// function inCircle(ax, ay, bx, by, cx, cy, px, py) {
//     const dx = ax - px;
//     const dy = ay - py;
//     const ex = bx - px;
//     const ey = by - py;
//     const fx = cx - px;
//     const fy = cy - py;

//     const ap = dx * dx + dy * dy;
//     const bp = ex * ex + ey * ey;
//     const cp = fx * fx + fy * fy;

//     return dx * (ey * cp - bp * fy) -
//            dy * (ex * cp - bp * fx) +
//            ap * (ex * fy - ey * fx) < 0;
// }

// function circumradius(ax, ay, bx, by, cx, cy) {
//     const dx = bx - ax;
//     const dy = by - ay;
//     const ex = cx - ax;
//     const ey = cy - ay;

//     const bl = dx * dx + dy * dy;
//     const cl = ex * ex + ey * ey;
//     const d = 0.5 / (dx * ey - dy * ex);

//     const x = (ey * bl - dy * cl) * d;
//     const y = (dx * cl - ex * bl) * d;

//     return x * x + y * y;
// }

// function circumcenter(ax, ay, bx, by, cx, cy) {
//     const dx = bx - ax;
//     const dy = by - ay;
//     const ex = cx - ax;
//     const ey = cy - ay;

//     const bl = dx * dx + dy * dy;
//     const cl = ex * ex + ey * ey;
//     const d = 0.5 / (dx * ey - dy * ex);

//     const x = ax + (ey * bl - dy * cl) * d;
//     const y = ay + (dx * cl - ex * bl) * d;

//     return {x, y};
// }

// function quicksort(ids, dists, left, right) {
//     if (right - left <= 20) {
//         for (let i = left + 1; i <= right; i++) {
//             const temp = ids[i];
//             const tempDist = dists[temp];
//             let j = i - 1;
//             while (j >= left && dists[ids[j]] > tempDist) ids[j + 1] = ids[j--];
//             ids[j + 1] = temp;
//         }
//     } else {
//         const median = (left + right) >> 1;
//         let i = left + 1;
//         let j = right;
//         swap(ids, median, i);
//         if (dists[ids[left]] > dists[ids[right]]) swap(ids, left, right);
//         if (dists[ids[i]] > dists[ids[right]]) swap(ids, i, right);
//         if (dists[ids[left]] > dists[ids[i]]) swap(ids, left, i);

//         const temp = ids[i];
//         const tempDist = dists[temp];
//         while (true) {
//             do i++; while (dists[ids[i]] < tempDist);
//             do j--; while (dists[ids[j]] > tempDist);
//             if (j < i) break;
//             swap(ids, i, j);
//         }
//         ids[left + 1] = ids[j];
//         ids[j] = temp;

//         if (right - i + 1 >= j - left) {
//             quicksort(ids, dists, i, right);
//             quicksort(ids, dists, left, j - 1);
//         } else {
//             quicksort(ids, dists, left, j - 1);
//             quicksort(ids, dists, i, right);
//         }
//     }
// }

// function swap(arr, i, j) {
//     const tmp = arr[i];
//     arr[i] = arr[j];
//     arr[j] = tmp;
// }

// function defaultGetX(p) {
//     return p[0];
// }
// function defaultGetY(p) {
//     return p[1];
// }
