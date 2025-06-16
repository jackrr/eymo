// @group(0) @binding(0) var input_texture : texture_2d<f32>;
// @group(0) @binding(1) var input_sampler : sampler;
@group(0) @binding(0) var<storage, read> input_buf : array<u32>;
@group(0) @binding(1) var output_texture
    : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> width : u32;
@group(0) @binding(3) var<uniform> height : u32;

fn float_color(c : u32) -> f32 { return f32(c) / 255.0; }

// To get 3 u8 starting at `byte_index`, converting to u32:
fn get_rgb(byte_index : u32) -> vec3<u32> {
  let word_index = byte_index / 4u;
  let byte_offset = byte_index % 4u;

  let word1 = input_buf[word_index];
  let word2 = input_buf[word_index + 1u];

  // Extract bytes using bit operations
  var bytes : vec3<u32>;

  if (byte_offset == 0u) {
    bytes.x = word1 & 0xFFu;
    bytes.y = (word1 >> 8u) & 0xFFu;
    bytes.z = (word1 >> 16u) & 0xFFu;
  } else if (byte_offset == 1u) {
    bytes.x = (word1 >> 8u) & 0xFFu;
    bytes.y = (word1 >> 16u) & 0xFFu;
    bytes.z = (word1 >> 24u) & 0xFFu;
  } else if (byte_offset == 2u) {
    bytes.x = (word1 >> 16u) & 0xFFu;
    bytes.y = (word1 >> 24u) & 0xFFu;
    bytes.z = word2 & 0xFFu;
  } else { // byte_offset == 3u
    bytes.x = (word1 >> 24u) & 0xFFu;
    bytes.y = word2 & 0xFFu;
    bytes.z = (word2 >> 8u) & 0xFFu;
  }

  return bytes;
}

fn get_rgba(coords : vec2<u32>) -> vec4<f32> {
  let array_offset = coords.y * width + coords.x;
  let rgb = get_rgb(array_offset * 3);
  return vec4<f32>(float_color(rgb.x), float_color(rgb.y), float_color(rgb.z),
                   1.0);
}

// Manual bilinear interpolation for compute shaders
fn bilinear_sample(uv : vec2<f32>) -> vec4<f32> {
  let input_size = vec2<f32>(vec2<u32>(width, height));

  let pixel_coords = uv * input_size - 0.5;
  let base_coords = floor(pixel_coords);
  let fract_coords = pixel_coords - base_coords;

  let x0 = i32(base_coords.x);
  let y0 = i32(base_coords.y);
  let x1 = x0 + 1;
  let y1 = y0 + 1;

  let input_dims = vec2<i32>(vec2<u32>(width, height));

  // Clamp coordinates to texture bounds
  let c00 = vec2<u32>(vec2<i32>(clamp(x0, 0, input_dims.x - 1),
                                clamp(y0, 0, input_dims.y - 1)));
  let c10 = vec2<u32>(vec2<i32>(clamp(x1, 0, input_dims.x - 1),
                                clamp(y0, 0, input_dims.y - 1)));
  let c01 = vec2<u32>(vec2<i32>(clamp(x0, 0, input_dims.x - 1),
                                clamp(y1, 0, input_dims.y - 1)));
  let c11 = vec2<u32>(vec2<i32>(clamp(x1, 0, input_dims.x - 1),
                                clamp(y1, 0, input_dims.y - 1)));

  // Sample the four neighboring pixels
  let p00 = get_rgba(c00);
  let p10 = get_rgba(c10);
  let p01 = get_rgba(c01);
  let p11 = get_rgba(c11);

  // Bilinear interpolation
  let top = mix(p00, p10, fract_coords.x);
  let bottom = mix(p01, p11, fract_coords.x);
  return mix(top, bottom, fract_coords.y);
}

@compute @workgroup_size(8, 8) fn
    resize_image_linear(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let output_size = textureDimensions(output_texture);
  let coords = vec2<u32>(global_id.xy);

  // Early exit if outside bounds
  if (coords.x >= output_size.x || coords.y >= output_size.y) {
    return;
  }

  // Calculate UV coordinates for the input texture
  let uv = (vec2<f32>(coords) + 0.5) / vec2<f32>(output_size);

  let color = bilinear_sample(uv);

  // Write to output texture
  textureStore(output_texture, coords, color);
}

@compute @workgroup_size(8, 8) fn
    resize_image_nearest(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let output_size = textureDimensions(output_texture);
  let coords = vec2<i32>(global_id.xy);

  if (coords.x >= i32(output_size.x) || coords.y >= i32(output_size.y)) {
    return;
  }
  let input_size = vec2<f32>(vec2<u32>(width, height));

  // Calculate corresponding input coordinates
  let scale = input_size / vec2<f32>(output_size);
  let input_coords = vec2<i32>(vec2<f32>(coords) * scale);

  // Clamp to texture bounds
  let clamped_coords =
      vec2<u32>(clamp(input_coords, vec2<i32>(0), vec2<i32>(input_size) - 1));

  // Sample and store
  let color = get_rgba(clamped_coords);
  textureStore(output_texture, coords, color);
}
