# eymo

Ey(e)mo(uth) is a program that does fun things with a video stream of a face.

## Installation / first time setup

### External Dependencies

- [rust](https://www.rust-lang.org/tools/install)
- [ffmpeg](https://ffmpeg.org/)

By default, eymo will stream output to a child `ffplay` process for
display in a window. To stream output to a virtual webcam device see
operating system installation requirements...

#### Linux (*optional*)

To create a virtual webcam device to stream to, the
[v4l2loopback](https://github.com/v4l2loopback/v4l2loopback) kernel
module must be installed and loaded. Run eymo with `-d video1` to send
output to a virtual camera at `/dev/video1` created by v4l2loopback.

#### MacOS

TODO

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
copy_to(nose, mouth, ...) // copy shape to one or more destination shapes, appying transforms to shape and destinations
write_to(nose, mouth, ...) // copy shape to one or more destination shapes, applying transforms only to destinations
swap_with(nose) // swap shape contents with target shape contents
translate(50, -100) // move shape 50 to the right and up 100 (Y minimum is 0 at top of frame)
flip(vertical) // flip pixels vertically in shape. accepts vertcial | horizontal | both
drift(150, 45) // move shape 150 pixels/second at a 45° angle from starting point. shape will "bounce" off edges of the frame
spin(-0.25) // rotate shape -90°/second (1.0 yields a full clockwise rotation every second)
brighten(0.5) // brighten/darken shape by given factor (in this case darken by 50%)
saturate(1.5) // increase/decrease saturation of shape by given factor (in this case saturate by 150%)
channels(1.5, 0.5, 0.8) // increase/decrease rgb levels by given factors (in this case 150% red, 50% green, 80% blue)
```


## Built with...

- [ffmpeg](https://ffmpeg.org/) output video streaming
- [lalrpop](https://github.com/lalrpop/lalrpop) language parser
- [mediapipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) face detection and landmarker models
- [nokwha](https://github.com/l1npengtul/nokhwa) cross-platform webcam streaming
- [ort](https://github.com/pykeio/ort) model inference execution runtime
- [wgpu](https://github.com/gfx-rs/wgpu) cross-platform GPU execution of image manipulations

## License - TODO
