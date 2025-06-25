@group(0) @binding(0) var tex : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<u32>;

// Write rgb vals packed cast to u8 into output buffer
@compute @workgroup_size(8, 8) fn
    texture_to_rgb_buf_u8(@builtin(global_invocation_id) global_id : vec3u) {
  let dims = textureDimensions(tex);
	// 0-1 -> 0-255
  let color =
		vec3u(round(textureLoad(tex, global_id.xy, 0).rgb * 255));

  let pixel_index = u32(global_id.y * dims.x + global_id.x) * 3;

	let pack_index = pixel_index / 4u;
  let pack_offset = pixel_index % 4u;

	let shifts = vec4u(0, 8, 16, 24);
	let masks = vec4u(
										255u << shifts[0], // 1st 8 bits
										255u << shifts[1], // 2nd 8 bits
										255u << shifts[2], // 3rd 8 bits
										255u << shifts[3], // 4th 8 bits
										);

  if (pack_offset == 0u) {
		// 0       1       2       3
		// rrrrrrrrggggggggbbbbbbbbxxxxxxxx
		// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
		// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
		let rmask = (color.r << shifts[0]) & masks[0];
		let gmask = (color.g << shifts[1]) & masks[1];
		let bmask = (color.b << shifts[2]) & masks[2];
		output[pack_index] = output[pack_index] | rmask | gmask | bmask;
	} else if (pack_offset == 3u) {
		// xxxxxxxxxxxxxxxxxxxxxxxxrrrrrrrr
		// ggggggggbbbbbbbbxxxxxxxxxxxxxxxx
		// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
		let rmask = (color.r << shifts[3]) & masks[3];
		let gmask = (color.g << shifts[0]) & masks[0];
		let bmask = (color.b << shifts[1]) & masks[1];
		output[pack_index] = output[pack_index] | rmask;
		output[pack_index + 1] = output[pack_index + 1] | gmask | bmask;
	} else if (pack_offset == 2u) {
		// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
		// xxxxxxxxxxxxxxxxrrrrrrrrgggggggg
		// bbbbbbbbxxxxxxxxxxxxxxxxxxxxxxxx
		let rmask = (color.r << shifts[2]) & masks[2];
		let gmask = (color.g << shifts[3]) & masks[3];
		let bmask = (color.b << shifts[0]) & masks[0];
		output[pack_index] = output[pack_index] | rmask | gmask;
		output[pack_index + 1] = output[pack_index + 1] | bmask;
  } else { // (pack_offset == 1u)
		// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
		// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
		// xxxxxxxxrrrrrrrrggggggggbbbbbbbb
		let rmask = (color.r << shifts[1]) & masks[1];
		let gmask = (color.g << shifts[2]) & masks[2];
		let bmask = (color.b << shifts[3]) & masks[3];
		output[pack_index] = output[pack_index] | rmask | gmask | bmask;
	}
}

// 0,0
// pixidx = 0
// pack_idx = 0
// pack_offset = 0
// 0,1
// pixidx = 3
// pack_idx = 0
// pack_offset = 3
// 0,2
// pixidx = 6
// pack_idx = 1
// pack_offset = 2
// 0,3
// pixidx = 9
// pack_idx = 2
// pack_offset = 1
// 0,4
// pixidx = 12
// pack_idx = 3
// pack_offset = 0
