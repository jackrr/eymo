pub fn process_frame(model: &Session, frame: &mut Mat) -> Result<()> {
    let mut resized = Mat::default();

    let height = 640;
    let width = 640;

    // TODO: is the resize distorting?
    imgproc::resize(
        frame,
        &mut resized,
        Size { width, height },
        0.0,
        0.0,
        imgproc::INTER_AREA,
    )?;

    let mut resized_f32 = Array::zeros((1, 3, height as usize, width as usize));
    let height = resized.rows() as usize;
    let width = resized.cols() as usize;
    let chans = resized.channels() as usize;

    // b r g -> r g b
    let chan_map = vec![2, 0, 1];

    // TODO: consider threading this
    unsafe {
        let mat_slice = std::slice::from_raw_parts(resized.data(), height * width * chans);

        for y in 0..height {
            for x in 0..width {
                for ch in 0..chans {
                    let idx = (y * width * chans) + (x * chans) + ch;
                    resized_f32[[0, chan_map[ch], y, x]] = (mat_slice[idx] as f32) / 255.;
                }
            }
        }
    }

    let input = Tensor::from_array(resized_f32)?;
    let outputs = model.run(ort::inputs!["images" => input]?)?;
    let results = outputs["output0"].try_extract_tensor::<f32>()?;

    // TODO: write results to frame, projecting result coordinates
    show_results(&mut resized, results, 0, 0)?;

    Ok(())
}

fn show_results(
    img: &mut Mat,
    result: ndarray::ArrayBase<ViewRepr<&f32>, Dim<IxDynImpl>>,
    x_offset: i32,
    y_offset: i32,
) -> Result<()> {
    for row in result.squeeze().columns() {
        let row: Vec<_> = row.iter().copied().collect();
        let c = row[4];
        if c < 0.8 {
            continue;
        }

        let xc = row[0] as i32 + x_offset; // centerpoint x
        let yc = row[1] as i32 + y_offset; // centerpoint y
        let w = row[2].round() as i32;
        let h = row[3].round() as i32;

        imgproc::rectangle(
            img,
            Rect::new(xc - w / 2, yc - h / 2, w, h),
            core::VecN([255., 0., 0., 0.]),
            1,
            imgproc::LINE_8,
            0,
        )?;

        // 0 - nose, 1 - l eye, 2 r eye, 3 l ear, 4 r ear
        for k in 0..5 {
            let k_idx = 5 + (k * 3);
            let kc = row[k_idx + 2];

            if kc < 0.8 {
                continue;
            }

            let kx = row[k_idx].round() as i32 + x_offset;
            let ky = row[k_idx + 1].round() as i32 + y_offset;

            imgproc::circle(
                img,
                Point::new(kx, ky),
                5,
                Scalar::new(0., 0., 255., 0.),
                -1, // fill
                imgproc::LINE_8,
                0,
            )?;

            imgproc::put_text(
                img,
                &format!("{k} - {kcr}", kcr = kc.round() as i32),
                Point::new(kx + 10, ky + 5),
                imgproc::FONT_HERSHEY_COMPLEX_SMALL,
                0.5,
                Scalar::new(255., 255., 255., 255.),
                1, //thickness
                imgproc::LINE_8,
                false,
            )?;
        }
    }

    Ok(())
}
