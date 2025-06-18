@group(0) @binding(0) var<storage, read> input_buf : array<u32>;
// (theta.cos, theta.sin)
@group(0) @binding(1) var<uniform> rotation : vec2<f32>;
@group(0) @binding(2) var<uniform> default_color : vec4<f32>;
@group(0) @binding(3) var output_texture
    : texture_storage_2d<rgba8unorm, write>;

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
  let img_size = textureDimensions(output_texture);
  let array_offset = coords.y * img_size.x + coords.x;
  let rgb = get_rgb(array_offset * 3);
  return vec4<f32>(float_color(rgb.x), float_color(rgb.y), float_color(rgb.z),
                   1.0);
}

@compute @workgroup_size(8, 8) fn
    rotate_image_nearest(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let img_size = vec2<i32>(textureDimensions(output_texture));
  let coords = vec2<i32>(global_id.xy);

  if (coords.x >= img_size.x || coords.y >= img_size.y) {
    return;
  }

  // Calculate corresponding input coordinates
  let center = img_size / 2;
  let trans = vec2<f32>(coords - center);
  let rot = vec2<f32>(trans.x * rotation.x - trans.y * rotation.y,
                      trans.x * rotation.y + trans.y * rotation.x);
  let input_coords = vec2<i32>(round(rot + trans));

  if (input_coords.x < 0 || input_coords.x >= img_size.x ||
      input_coords.y < 0 || input_coords.y >= img_size.y) {
    textureStore(output_texture, coords, default_color);
  } else {
    let color = get_rgba(vec2<u32>(input_coords));
    textureStore(output_texture, coords, color);
  }
}
