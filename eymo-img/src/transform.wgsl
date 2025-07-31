@group(0) @binding(0) var input_tex : texture_2d<f32>;
@group(0) @binding(1) var samp : sampler;
// adjustments[0] -> brightness pct (if > 0)
// adjustments[1] -> saturation pct (if > 0)
@group(0) @binding(2) var<uniform> adjustments: vec2f;
// R, G, B, A pct modifiers
@group(0) @binding(3) var<uniform> chans: vec4f;

fn hsv_to_rgb(hsv: vec3f) -> vec3f {
	let h = hsv.r;
	let s = hsv.g;
	let v = hsv.b;
	let i = floor(h * 6.);
	let f = h * 6 - i;
	let p = v * (1 - s);
	let q = v * (1 - f * s);
	let t = v * (1 - (1 - f) * s);

	let offset = u32(i % 6);

	switch offset {
			case 0u, default {
				return vec3f(v, t, p);
			}
			case 1u: {
				return vec3f(q, v, p);
			}
			case 2u: {
				return vec3f(p, v, t);
			}
			case 3u: {
				return vec3f(p, q, v);
			}
			case 4u: {
				return vec3f(t, p, v);
			}
			case 5u: {
				return vec3f(v, p, q);
			}
	 }
}

fn rgb_to_hsv(rgb: vec3f) -> vec3f {
	let max = max(max(rgb.r, rgb.g), rgb.b);
	let min = min(min(rgb.r, rgb.g), rgb.b);
	let delta = max - min;
	let v = max;

	var s = 0.;
	if max != 0. {
			s = delta / max;
	}

	var h = 0.;
	if s != 0. {
			if rgb.r == max {
					h = (rgb.g - rgb.b) / delta;
			} else if rgb.g == max {
					h = 2 + (rgb.b - rgb.r) / delta;
			} else if rgb.b == max {
					h = 4 + (rgb.r - rgb.g) / delta;
			}
	}

	h *= 60.;

	if h < 0. {
			h += 360.;
	}

	return vec3f(h / 360., s, v);
}

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
	var color = textureSample(input_tex, samp, pos.tex_coord);

	if chans.r >= 0. || chans.g >= 0. || chans.b >= 0. || chans.a >= 0. {
			color *= chans;
			color = max(color, vec4f(0., 0., 0., 0.));
			color = min(color, vec4f(1., 1., 1., 1.));
	}

	if adjustments.x < 0. && adjustments.y < 0. {
			return color;
	}

	var hsv = rgb_to_hsv(color.rgb);
	if adjustments.x >= 0. {
			// brightness
			hsv.b = min(hsv.b * adjustments.x, 1.0);
	}

	if adjustments.y >= 0. {
			// saturation
			hsv.g = min(hsv.g * adjustments.y, 1.0);
	}

	return vec4f(hsv_to_rgb(hsv), color.a);
}
