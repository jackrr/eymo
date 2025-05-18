# eymo

Ey(e)mo(uth) is a program that does fun things with a video stream of a face.

## Getting models

1. Install `yolo` command line according to the [Ultralytics
   docs](https://docs.ultralytics.com/quickstart/) (yes, you need
   python...)
2. Run `yolo export model=yolo11s-pose.pt format=onnx imgsz=640` and
   copy the generated file to the `models/` directory
   
## TODO

- Mouth detection (maybe assume an offset from nose?)
- Write to virtual camera detectable by other apps (browser cameras, zoom, etc)
- Fun effects
  - Swap eyes and lips
  - Have eyes and lips rotate about the screen
- Tracking -- do not re-run model on frame -- instead look for
  telltale pixels to maintain a centerpoint + shape for the target
  object

## Other idea spaces to explore

- Add an image segmentation model for background replacement
