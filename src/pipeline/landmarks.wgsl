@group(0) @binding(0) var input_tex : texture_2d<f32>;
@group(0) @binding(1) var samp : sampler;
@group(0) @binding(2) var<uniform> out_dims: vec2f;
@group(0) @binding(3) var<uniform> rot: vec2f;

struct VertexIn {
  @location(0) position : vec2f,
}

struct VertexOut {
  @builtin(position) position : vec4f,
}

@vertex fn vert_main(in : VertexIn) -> VertexOut {
  var out : VertexOut;
  out.position = vec4f(in.position, 0., 1.);
  return out;
}

@fragment fn frag_main(pos : VertexOut) -> @location(0) vec4f {
	// FIXME: scaling up not happening
	// FIXME: rotation not quite right
	let coords = pos.position.xy / out_dims; // 0 -> 1.
	let center = vec2f(0.5, 0.5);
	let trans = coords - center;
	let rot = vec2f(trans.x * rot.x - trans.y * rot.y, trans.x * rot.y + trans.y * rot.x);
	let point = rot + trans;
	return textureSample(input_tex, samp, coords);
  // return textureSample(input_tex, samp, point);
}
