@group(0) @binding(0) var tex : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;

@compute @workgroup_size(8, 8) fn
    tex_to_rgb_buf_neg1_1(@builtin(global_invocation_id) global_id : vec3u) {
  let dims = textureDimensions(tex);
	// 0 -> 1 --> -1 -> 1
  let color =
		textureLoad(tex, global_id.xy, 0).rgb * 2. - 1.;

  let output_offset = u32(global_id.y * dims.x * 3 + (global_id.x * 3));
  output[output_offset] = color.r;
	output[output_offset+1] = color.g;
	output[output_offset+2] = color.b;
}

@compute @workgroup_size(8, 8) fn
    tex_to_rgb_buf_0_1(@builtin(global_invocation_id) global_id : vec3u) {
  let dims = textureDimensions(tex);
  let color =
		textureLoad(tex, global_id.xy, 0);

  let output_offset = u32(global_id.y * dims.x * 3 + (global_id.x * 3));
  output[output_offset] = color.r;
	output[output_offset+1] = color.g;
	output[output_offset+2] = color.b;
}
