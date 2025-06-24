@group(0) @binding(0) var input_tex : texture_2d<f32>;
@group(0) @binding(1) var samp : sampler;
@group(0) @binding(2) var<uniform> out_dims: vec2f;

struct VertexIn {
  @location(0) position : vec2f,
	@location(1) tex_coord: vec2f,
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
  return textureSample(input_tex, samp, pos.position.xy / out_dims);
}
