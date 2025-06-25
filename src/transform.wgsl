@group(0) @binding(0) var input_tex : texture_2d<f32>;
@group(0) @binding(1) var samp : sampler;
// rot[0]: 0 = disable, nonzero = enable
// rot[1]: cos
// rot[2]: sin
@group(0) @binding(2) var<uniform> rot: vec3f;

struct VertexIn {
  @location(0) position : vec2f,
	@location(1) tex_coord: vec2f,
}

struct VertexOut {
  @builtin(position) position : vec4f,
	@location(0) tex_coord: vec2f,
}

@vertex fn vert_main(in : VertexIn) -> VertexOut {
  var out : VertexOut;
  out.position = vec4f(in.position, 0., 1.);
	out.tex_coord = in.tex_coord;
  return out;
}

@fragment fn frag_main(pos : VertexOut) -> @location(0) vec4f {
	var coords = pos.tex_coord;
	
	if (rot[0] != 0.) {
		let center = vec2f(0.5, 0.5);
		let trans = coords - center;
		let rot = vec2f(trans.x * rot[1] - trans.y * rot[2], trans.x * rot[2] + trans.y * rot[1]);
		coords = rot + center;
	}
	
  return textureSample(input_tex, samp, coords);
}
