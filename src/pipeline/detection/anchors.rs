use crate::pipeline::rect::Rect;
use anyhow::{Error, Result};
use log::debug;

const SCALES: [f32; 2] = [0.1484375, 0.75];
const STRIDES: [u32; 4] = [8, 16, 16, 16];
const WIDTH: u32 = 128;
const HEIGHT: u32 = 128;

// absl::Status SsdAnchorsCalculator::GenerateAnchors(
//   int layer_id = 0;
//   while (layer_id < options.num_layers()) {
//     std::vector<float> anchor_height;
//     std::vector<float> anchor_width;
//     std::vector<float> aspect_ratios;
//     std::vector<float> scales;

//     // For same strides, we merge the anchors in the same order.
//     int last_same_stride_layer = layer_id; 0
//     while (last_same_stride_layer < options.strides_size() &&
//            options.strides(last_same_stride_layer) ==
//                options.strides(layer_id)) {
//       const float scale =
//           CalculateScale(options.min_scale(), options.max_scale(),
//                          last_same_stride_layer, options.strides_size());
//       if (last_same_stride_layer == 0 &&
//           options.reduce_boxes_in_lowest_layer()) {
//         // For first layer, it can be specified to use predefined anchors.
//         aspect_ratios.push_back(1.0);
//         aspect_ratios.push_back(2.0);
//         aspect_ratios.push_back(0.5);
//         scales.push_back(0.1);
//         scales.push_back(scale);
//         scales.push_back(scale);
//       } else {
//         for (int aspect_ratio_id = 0;
//              aspect_ratio_id < options.aspect_ratios_size();
//              ++aspect_ratio_id) {
//           aspect_ratios.push_back(options.aspect_ratios(aspect_ratio_id));
//           scales.push_back(scale);
//         }
//         if (options.interpolated_scale_aspect_ratio() > 0.0) {
//           const float scale_next =
//               last_same_stride_layer == options.strides_size() - 1
//                   ? 1.0f
//                   : CalculateScale(options.min_scale(), options.max_scale(),
//                                    last_same_stride_layer + 1,
//                                    options.strides_size());
//           scales.push_back(std::sqrt(scale * scale_next));
//           aspect_ratios.push_back(options.interpolated_scale_aspect_ratio());
//         }
//       }
//       last_same_stride_layer++;
//     }

//     for (int i = 0; i < aspect_ratios.size(); ++i) {
//       const float ratio_sqrts = std::sqrt(aspect_ratios[i]);
//       anchor_height.push_back(scales[i] / ratio_sqrts);
//       anchor_width.push_back(scales[i] * ratio_sqrts);
//     }

//     int feature_map_height = 0;
//     int feature_map_width = 0;
//     if (options.feature_map_height_size()) {
//       feature_map_height = options.feature_map_height(layer_id);
//       feature_map_width = options.feature_map_width(layer_id);
//     } else {
//       const int stride = options.strides(layer_id);
//       feature_map_height =
//           std::ceil(1.0f * options.input_size_height() / stride);
//       feature_map_width = std::ceil(1.0f * options.input_size_width() / stride);
//     }

//     for (int y = 0; y < feature_map_height; ++y) {
//       for (int x = 0; x < feature_map_width; ++x) {
//         for (int anchor_id = 0; anchor_id < anchor_height.size(); ++anchor_id) {
//           // TODO: Support specifying anchor_offset_x, anchor_offset_y.
//           const float x_center =
//               (x + options.anchor_offset_x()) * 1.0f / feature_map_width;
//           const float y_center =
//               (y + options.anchor_offset_y()) * 1.0f / feature_map_height;

//           Anchor new_anchor;
//           new_anchor.set_x_center(x_center);
//           new_anchor.set_y_center(y_center);

//           if (options.fixed_anchor_size()) {
//             new_anchor.set_w(1.0f);
//             new_anchor.set_h(1.0f);
//           } else {
//             new_anchor.set_w(anchor_width[anchor_id]);
//             new_anchor.set_h(anchor_height[anchor_id]);
//           }
//           anchors->push_back(new_anchor);
//         }
//       }
//     }
//     layer_id = last_same_stride_layer;
//   }
//   return absl::OkStatus();
// }

pub fn gen_anchor(offset: u32) -> Result<Rect> {
    // layer 0 -> 8 len strides, scale = scale(scales[0] , scales[1], 0, 4)
    // layer 1 -> 8 len strides, scale = square_root(
    //       scale(scales[0] , scales[1], 0, 4) *
    //       scale(scales[0] , scales[1], 1, 4)
    //     )
    // layer 1 -> 16 len strides,
    if offset >= 896 {
        return Err(Error::msg(format!(
            "Cannot handle offset {} (must be <896)",
            offset
        )));
    }

    // Build a lookup of layers by length
    let mut layers = Vec::new();
    let mut agg_offset = 0;
    for stride in STRIDES {
        // It's crazy, but model seemingly assumes fixed size anchors
        for _ in SCALES {
            // on a layer
            layers.push((agg_offset, stride));
            agg_offset += WIDTH * HEIGHT / stride / stride;
        }
    }

    // Find layer offset falls within
    let mut stride: u32 = 0;
    let mut layer_idx: u32 = 0;
    for (i, str) in layers {
        if i > offset {
            break;
        }

        stride = str;
        layer_idx = offset - i;
    }

    // Build anchor at layer_idx
    let strides_per_row = WIDTH / stride;

    let col_num = layer_idx / strides_per_row;
    let row_num = layer_idx % strides_per_row;

    Ok(Rect {
        x: col_num * stride,
        y: row_num * stride,
        w: 1,
        h: 1,
    })
}

fn mult(v: u32, m: f32) -> u32 {
    (v as f32 * m).round() as u32
}

fn scale(min_scale: f32, max_scale: f32, stride: u32, strides: u32) -> f32 {
    if strides == 1 {
        (min_scale + max_scale) * 0.5
    } else {
        min_scale + (max_scale - min_scale) * stride as f32 / (strides as f32 - 1.)
    }
}
