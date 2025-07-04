use crate::shapes::rect::RectF32;

const MIN_SCALE: f32 = 0.1484375;
const MAX_SCALE: f32 = 0.75;
const STRIDES: [u32; 4] = [8, 16, 16, 16];
const WIDTH: u32 = 128;
const HEIGHT: u32 = 128;
const X_ANCHOR_OFFSET: f32 = 0.5;
const Y_ANCHOR_OFFSET: f32 = 0.5;

fn calc_scale(min_scale: f32, max_scale: f32, stride: usize, strides: usize) -> f32 {
    if strides == 1 {
        (min_scale + max_scale) * 0.5
    } else {
        min_scale + (max_scale - min_scale) * stride as f32 / (strides as f32 - 1.)
    }
}

pub fn gen_anchors() -> [RectF32; 896] {
    let mut layer_id = 0;
    let mut anchors: [RectF32; 896] = [RectF32::default(); 896];
    let mut anchor_idx = 0;

    while layer_id < STRIDES.len() {
        let mut scales: Vec<f32> = Vec::new();

        let mut last_same_stride_layer = layer_id;

        while last_same_stride_layer < STRIDES.len()
            && STRIDES[last_same_stride_layer] == STRIDES[layer_id]
        {
            let scale = calc_scale(MIN_SCALE, MAX_SCALE, last_same_stride_layer, STRIDES.len());
            scales.push(scale);

            let scale_next = if last_same_stride_layer == STRIDES.len() - 1 {
                1.
            } else {
                calc_scale(
                    MIN_SCALE,
                    MAX_SCALE,
                    last_same_stride_layer + 1,
                    STRIDES.len(),
                )
            };

            scales.push((scale * scale_next).sqrt());

            last_same_stride_layer += 1;
        }

        let stride = STRIDES[layer_id];
        let feature_map_height = HEIGHT.div_ceil(stride);
        let feature_map_width = WIDTH.div_ceil(stride);

        for y in 0..feature_map_height {
            for x in 0..feature_map_width {
                for (_, _) in scales.iter().enumerate() {
                    anchors[anchor_idx].x =
                        (x as f32 + X_ANCHOR_OFFSET) / feature_map_width as f32 * WIDTH as f32;
                    anchors[anchor_idx].y =
                        (y as f32 + Y_ANCHOR_OFFSET) / feature_map_height as f32 * HEIGHT as f32;
                    anchor_idx += 1;
                }
            }
        }
        layer_id = last_same_stride_layer;
    }

    anchors
}
