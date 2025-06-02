pub mod face_detection {
    include!(concat!(
        env!("OUT_DIR"),
        "/models/mediapipe_face_detection_short_range.rs"
    ));
    // include!("src/model/something.rs");
}

pub mod face_landmark {
    include!(concat!(
        env!("OUT_DIR"),
        "/models/mediapipe_face_landmark.rs"
    ));
    // include!("src/model/something2.rs");
}
