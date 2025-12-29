pub fn rounded_div(d: u32, q: u32) -> u32 {
    (d as f32 / q as f32).round() as u32
}

pub fn mult(v: u32, f: f32) -> u32 {
    (v as f32 * f).round() as u32
}
