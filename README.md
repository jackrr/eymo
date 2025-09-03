# eymo

Ey(e)mo(uth) is a program that does fun things with a video stream of a face.

You can view the live WASM version [here](https://jackratner.com/projects/eymo) or follow the installation instructions below to run on desktop.

## Installation / first time setup

### Desktop

1. Clone this repo
2. Ensure you have [rust](https://www.rust-lang.org/tools/install) and [ffmpeg](https://ffmpeg.org/) installed
3. Run: `cargo build --release` from within the `eymo-desktop` directory

Compiled binary will be at
`./eymo-desktop/target/release/eymo-desktop`. Run the command with the `-h` or `--help` flag to see usage instructions.

By default, eymo will stream output to a child `ffplay` process for
display in a window. To stream output to a virtual webcam device see
OS-specific installation requirements as follows:

#### Linux (*optional*)

To create a virtual webcam device to stream to, the
[v4l2loopback](https://github.com/v4l2loopback/v4l2loopback) kernel
module must be installed and loaded. Run eymo with `-d video1` to send
output to a virtual camera at `/dev/video1` created by v4l2loopback.

#### MacOS

Help Wanted: https://github.com/jackrr/eymo/issues/11

### WASM

NOTE: WASM implementation is still WIP!!

1. Clone this repo
2. Ensure you have [rust](https://www.rust-lang.org/tools/install) installed
3. Install [`wasm-pack`](https://drager.github.io/wasm-pack/installer/)
4. Run: `wasm-pack build -t web --release` from within the `eymo-wasm` directory

Compiled output will be in `./eymo-wasm/pkg/`. See `demo/index.html`
for an example of how to use the generated code.

## Configuration language

Eymo uses [lalrpop](https://github.com/lalrpop/lalrpop) to implement a
minimal "language" for applying transformations to faces in the webcam
stream. A basic example:
```
//  enlarge the mouth, swap it with the left eye, and copy it to the right eye
mouth: scale(2), swap_with(leye), copy_to(reye), drift(150, 45), spin(0.5)
```

All "transformations" target a shape and perform one or more
operations on it, conforming to the following syntax:

```
<target_shape>: <operation>(, operation)*
```


By default, builtin shapes target _all_ detected faces in a
frame. Shapes can be made to target specific faces by adding a
detection index:

```
// swap first and second face mouths. this will noop if less than 2 faces are detected in a frame
mouth#0: swap_with(mouth#1)
```

Additionally transforms can target faces by relative order in the
detection list:

```
// Copy each mouth to "next" mouth
mouth: copy_to(mouth+1)

// Copy each left eye to "previous" left eye
leye+1: copy_to(leye)
```

The set of available shapes are:

- `leye` - left eye
- `reye` - right eye
- `leye_region` - left eye region (includes eyebrow)
- `reye_region` - right eye region (includes eyebrow)
- `face` - ...face
- `nose` - ...nose
- `mouth` - ...mouth

Alternatively, custom rectangles can be used anywhere a built-in shape
could be used:

```
// copy the contents in the rectangle with top-left coodinate (100,000), width 50, and height 50 to noses
rect(100, 100, 50, 50): copy(nose)
```

The available operations are:

```
tile // tile image with specified shape
scale(2.5) // grow/shrink shape by given multiplication factor
rotate(-45) // rotate shape by given degrees
copy_to(nose, mouth, ...) // copy shape to one or more destination shapes
swap_with(nose) // swap shape contents with target shape contents
translate(50, -100) // move shape 50 to the right and up 100 (Y minimum is 0 at top of frame)
flip(vertical) // flip pixels vertically in shape. accepts vertcial | horizontal | both
drift(150, 45) // move shape 150 pixels/second at a 45° angle from starting point. shape will "bounce" off edges of the frame
spin(-0.25) // rotate shape -90°/second (1.0 yields a full clockwise rotation every second)
brighten(0.5) // brighten/darken shape by given factor (in this case darken by 50%)
saturate(1.5) // increase/decrease saturation of shape by given factor (in this case saturate by 150%)
channels(1.5, 0.5, 0.8) // increase/decrease rgb levels by given factors (in this case 150% red, 50% green, 80% blue)
```

## Navigating the codebase

See [overview.md](docs/overview.md) for more implementation details

## Contributing

Contributions are welcome! For bigger features or architectural suggestions, please open a new [issues](https://github.com/jackrr/eymo/issues) describing a change you'd like to make. For smaller changes or if you want to proceed with implementation, please fork the repo, verify your fix locally, open a pull request describing the desired change, and add [me](https://github.com/jackrr) as a reviewer.

Excited by the project and unsure where to start? Check out the [issues](https://github.com/jackrr/eymo/issues) to find something to work on!

## This could not have been built without the following:

- [delaunator](https://github.com/mapbox/delaunator) reimplemented in rust within this project
- [ffmpeg](https://ffmpeg.org/) output video streaming
- [lalrpop](https://github.com/lalrpop/lalrpop) language parser
- [mediapipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) face detection and landmarker models
- [nokwha](https://github.com/l1npengtul/nokhwa) cross-platform webcam streaming
- [robust](https://github.com/georust/robust) helpers for graphics processing
- [tract](http://github.com/sonos/tract) model inference execution runtime
- [wasm-bindgen](https://github.com/wasm-bindgen/wasm-bindgen)
- [wasm-pack](https://drager.github.io/wasm-pack/installer/)
- [wgpu](https://github.com/gfx-rs/wgpu) cross-platform GPU execution of image manipulations

