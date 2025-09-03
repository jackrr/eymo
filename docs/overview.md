# Architecture Overview

## eymo/eymo-desktop crate

A wrapper around `eymo/eymo-img` to enable it to run ergonomically on desktop environments (tested on Linux and MacOS).

Key pieces:

- CLAP for argument parsing
- Enables targeting an output video device or [`ffplay`](https://ffmpeg.org/) by default
- Uses [`nokwha`](https://github.com/l1npengtul/nokhwa) crate for cross-platform input video device abstraction
- Alternative execution mode to generate an image from an input image
- Allows specifying path to a configuration file


## eymo/eymo-wasm crate

Also a wrapper around `eymo/eymo-img`, enables `eymo` to run in a browser.

Key pieces:

- [wasm-bindgen](https://github.com/wasm-bindgen/wasm-bindgen) to expose APIs to JS layer
- [wasm-pack](https://github.com/drager/wasm-pack) to build a size-optimized wasm binary and surrounding JS entrypoints
- JS APIs:
  - `State` object to represent a video processor instance
  - `State#new` constructor function, hooks up to canvas in DOM and bootstraps webgpu and inference pipeline
  - `State.start` to kick off an infinite loop consuming video frames and transforming them
  - `State.set_cmd` to update configuration for transforms, hot-swapping the interpreter
  - `State.stop` to stop processing video frames

Notable implementation hurdles:

- Figuring out how to capture video frames from webcam, platform distinctions (limited platform support for [`MediaStreamTrackProcessor`](https://developer.mozilla.org/en-US/docs/Web/API/MediaStreamTrackProcessor) as well as [`HTMLVideoElement.sourceObject`](https://developer.mozilla.org/en-US/docs/Web/API/HTMLMediaElement/srcObject))
- Replacing [ort](https://docs.rs/ort/latest/ort/) w/ [tract](https://github.com/sonos/tract) in eymo-img
- Refactoring parts of eymo-img to embrace async Rust, as [pollster's](https://docs.rs/pollster/latest/pollster/) blocking mechanism is incompatible with WASM
- Debugging WebGPU memory map polling issue (see [rgb.rs](../eymo-img/src/imggpu/rgb.rs#L173-L188))
- Message passing oneshot pattern for communicating w/ async closures (resize callback and stopping long-running `start` frame-processor)
- State mutex pattern for handling interior mutability safely across async contexts (set command, resize, stop, per-frame processing)


## eymo/eymo-img crate

Core implementation of eymo.

Key pieces:

- face [detection](../eymo-img/src/pipeline/detection.rs) and [landmarking](../eymo-img/src/pipeline/landmarks.rs), leveraging exported [mediapipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) models executed on the [tract](https://github.com/sonos/tract) inference engine
- [lalrpop grammar](..//eymo-img/src/lang/grammar.lalrpop) and associated [runtime interpreter](../eymo-img/src/lang.rs) for config language
- [WebGPU transform implementations](../eymo-img/src/transform.rs) to carry out requested per-frame image transformations in a single render pass per statement
- [WebGPU format conversions](../eymo-img/src/imggpu/rgb.rs): Texture (rgba) -> Tensor (rgb) -- compute shader, 
- [WebGPU bootstrapping logic](../eymo-img/src/imggpu/gpu.rs), with forks for surface/WASM vs generic backends
- [`Shape`](../eymo-img/src/shapes/shape.rs) enum abstraction around [`Rect`](../eymo-img/src/shapes/rect.rs) and [`Polygon`](../eymo-img/src/shapes/polygon.rs) structs
- [Mapbox's Delauney triangulation](https://github.com/mapbox/delaunator) [reimplemented in rust](../eymo-img/src/triangulate.rs)

Notable implementation hurdles:

- Interpreting output tensors from face detection and face landmarker models
  - Searching github issues and forum sites
  - Anchors implementation for detection model
- Out-of-band detection when models were too slow
- Image transforms were slow until I reimplemented on WebGPU
- Maintaining state across transform runs across frames
- Converting models to final nnef format (tflite -> onnx -> nnef)
