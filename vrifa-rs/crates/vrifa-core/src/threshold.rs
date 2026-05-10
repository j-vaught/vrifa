use crate::{cvutil, Result, VrifaError};
use ndarray::Array2;
use opencv::core;
use opencv::imgproc;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ThresholdMode {
    Otsu,
    Triangle,
    Manual(f32),
    Percentile(f32),
    AdaptiveMean { block_size: u32, c: f32 },
    AdaptiveGaussian { block_size: u32, c: f32 },
}

impl ThresholdMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Otsu => "otsu",
            Self::Triangle => "triangle",
            Self::Manual(_) => "manual",
            Self::Percentile(_) => "percentile",
            Self::AdaptiveMean { .. } => "adaptive-mean",
            Self::AdaptiveGaussian { .. } => "adaptive-gaussian",
        }
    }

    /// Parse a SPEC of the form:
    ///   otsu
    ///   triangle
    ///   manual:VALUE          (VALUE is a 0..=255 absolute byte threshold)
    ///   percentile:P          (P is a 0..=100 percentile)
    ///   adaptive-mean:B:C     (B is the odd block size, C is a float offset)
    ///   adaptive-gaussian:B:C (same as adaptive-mean but Gaussian-weighted)
    pub fn parse(value: &str) -> Result<Self> {
        let value = value.trim();
        if value.is_empty() {
            return Err(VrifaError::InvalidParameter(
                "threshold spec is empty".into(),
            ));
        }
        let mut parts = value.split(':');
        let kind = parts
            .next()
            .ok_or_else(|| VrifaError::InvalidParameter("empty threshold spec".into()))?
            .trim()
            .to_ascii_lowercase();
        let rest: Vec<&str> = parts.collect();
        match kind.as_str() {
            "otsu" => {
                if !rest.is_empty() {
                    return Err(VrifaError::InvalidParameter(
                        "otsu does not take parameters".into(),
                    ));
                }
                Ok(Self::Otsu)
            }
            "triangle" => {
                if !rest.is_empty() {
                    return Err(VrifaError::InvalidParameter(
                        "triangle does not take parameters".into(),
                    ));
                }
                Ok(Self::Triangle)
            }
            "manual" => {
                if rest.len() != 1 {
                    return Err(VrifaError::InvalidParameter(format!(
                        "manual threshold expects manual:VALUE, got {value:?}"
                    )));
                }
                let v: f32 = rest[0].trim().parse().map_err(|e| {
                    VrifaError::InvalidParameter(format!("manual value invalid: {e}"))
                })?;
                Ok(Self::Manual(v))
            }
            "percentile" => {
                if rest.len() != 1 {
                    return Err(VrifaError::InvalidParameter(format!(
                        "percentile threshold expects percentile:P, got {value:?}"
                    )));
                }
                let p: f32 = rest[0].trim().parse().map_err(|e| {
                    VrifaError::InvalidParameter(format!("percentile invalid: {e}"))
                })?;
                if !(0.0..=100.0).contains(&p) {
                    return Err(VrifaError::InvalidParameter(format!(
                        "percentile must be in [0, 100], got {p}"
                    )));
                }
                Ok(Self::Percentile(p))
            }
            "adaptive-mean" | "adaptive-gaussian" => {
                if rest.len() != 2 {
                    return Err(VrifaError::InvalidParameter(format!(
                        "{kind} expects {kind}:BLOCK:C, got {value:?}"
                    )));
                }
                let block: u32 = rest[0].trim().parse().map_err(|e| {
                    VrifaError::InvalidParameter(format!("block size invalid: {e}"))
                })?;
                if block < 3 {
                    return Err(VrifaError::InvalidParameter(format!(
                        "{kind} block size must be >= 3, got {block}"
                    )));
                }
                let c: f32 = rest[1].trim().parse().map_err(|e| {
                    VrifaError::InvalidParameter(format!("C value invalid: {e}"))
                })?;
                if kind == "adaptive-mean" {
                    Ok(Self::AdaptiveMean { block_size: block, c })
                } else {
                    Ok(Self::AdaptiveGaussian { block_size: block, c })
                }
            }
            other => Err(VrifaError::InvalidParameter(format!(
                "unknown threshold kind {other:?}; expected otsu, triangle, manual, percentile, \
                 adaptive-mean, adaptive-gaussian"
            ))),
        }
    }
}

impl std::fmt::Display for ThresholdMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Otsu => f.write_str("otsu"),
            Self::Triangle => f.write_str("triangle"),
            Self::Manual(v) => write!(f, "manual:{v}"),
            Self::Percentile(p) => write!(f, "percentile:{p}"),
            Self::AdaptiveMean { block_size, c } => write!(f, "adaptive-mean:{block_size}:{c}"),
            Self::AdaptiveGaussian { block_size, c } => {
                write!(f, "adaptive-gaussian:{block_size}:{c}")
            }
        }
    }
}

/// Produce a binary {0, 255} mask from the normalized delta.
///
/// The `offset` is added to the global thresholds (Otsu, Triangle, Manual,
/// Percentile) before binarization. Adaptive modes ignore `offset` because
/// their `C` parameter already plays the same role.
pub fn apply_threshold(
    delta_norm: &Array2<u8>,
    roi_mask: &Array2<u8>,
    mode: ThresholdMode,
    offset: f32,
) -> Result<Array2<u8>> {
    let src = cvutil::array2_u8_to_mat(delta_norm)?;
    let dst_mat = match mode {
        ThresholdMode::Otsu => global_threshold_with_flag(&src, imgproc::THRESH_OTSU, offset)?,
        ThresholdMode::Triangle => {
            global_threshold_with_flag(&src, imgproc::THRESH_TRIANGLE, offset)?
        }
        ThresholdMode::Manual(value) => {
            let t = (value + offset).clamp(0.0, 255.0) as f64;
            apply_simple_threshold(&src, t)?
        }
        ThresholdMode::Percentile(p) => {
            let mut values: Vec<u8> = delta_norm
                .iter()
                .zip(roi_mask.iter())
                .filter_map(|(value, mask)| (*mask > 0).then_some(*value))
                .collect();
            let raw = if values.is_empty() {
                otsu_value(&src)?
            } else {
                values.sort_unstable();
                numpy_linear_percentile(&values, p.clamp(0.0, 100.0)) as f64
            };
            let t = (raw + offset as f64).clamp(0.0, 255.0);
            apply_simple_threshold(&src, t)?
        }
        ThresholdMode::AdaptiveMean { block_size, c } => {
            adaptive_threshold(&src, imgproc::ADAPTIVE_THRESH_MEAN_C, block_size, c)?
        }
        ThresholdMode::AdaptiveGaussian { block_size, c } => {
            adaptive_threshold(&src, imgproc::ADAPTIVE_THRESH_GAUSSIAN_C, block_size, c)?
        }
    };
    cvutil::mat_to_array2_u8(&dst_mat)
}

fn apply_simple_threshold(src: &core::Mat, t: f64) -> Result<core::Mat> {
    let mut dst = core::Mat::default();
    imgproc::threshold(src, &mut dst, t, 255.0, imgproc::THRESH_BINARY)?;
    Ok(dst)
}

/// Run the global auto-threshold (Otsu or Triangle) to recover its chosen
/// value, then re-binarize with that value plus the operator offset. We do
/// this in two passes because OpenCV's auto-threshold does not let us
/// supply an offset directly.
fn global_threshold_with_flag(src: &core::Mat, auto_flag: i32, offset: f32) -> Result<core::Mat> {
    let mut dst_unused = core::Mat::default();
    let raw_value = imgproc::threshold(
        src,
        &mut dst_unused,
        0.0,
        255.0,
        imgproc::THRESH_BINARY | auto_flag,
    )?;
    let t = (raw_value + offset as f64).clamp(0.0, 255.0);
    apply_simple_threshold(src, t)
}

fn otsu_value(src: &core::Mat) -> Result<f64> {
    let mut dst = core::Mat::default();
    Ok(imgproc::threshold(
        src,
        &mut dst,
        0.0,
        255.0,
        imgproc::THRESH_BINARY | imgproc::THRESH_OTSU,
    )?)
}

fn adaptive_threshold(
    src: &core::Mat,
    method: i32,
    block_size: u32,
    c: f32,
) -> Result<core::Mat> {
    let mut block = block_size as i32;
    if block % 2 == 0 {
        block += 1;
    }
    if block < 3 {
        block = 3;
    }
    let mut dst = core::Mat::default();
    imgproc::adaptive_threshold(
        src,
        &mut dst,
        255.0,
        method,
        imgproc::THRESH_BINARY,
        block,
        c as f64,
    )?;
    Ok(dst)
}

/// Compatibility shim for callers that still operate on the
/// (manual, percentile, offset) triple plus a separately-applied threshold
/// step. Returns the SCALAR threshold value that callers can hand to
/// cv::threshold. Adaptive modes cannot be expressed as a single value
/// and will error out -- those callers must use `apply_threshold` for
/// the per-pixel binary mask directly.
pub fn choose_threshold(
    delta_norm: &Array2<u8>,
    roi_mask: &Array2<u8>,
    mode: ThresholdMode,
    offset: f32,
) -> Result<f32> {
    let src = cvutil::array2_u8_to_mat(delta_norm)?;
    let raw = match mode {
        ThresholdMode::Otsu => otsu_value(&src)?,
        ThresholdMode::Triangle => triangle_value(&src)?,
        ThresholdMode::Manual(v) => v as f64,
        ThresholdMode::Percentile(p) => {
            let mut values: Vec<u8> = delta_norm
                .iter()
                .zip(roi_mask.iter())
                .filter_map(|(value, mask)| (*mask > 0).then_some(*value))
                .collect();
            if values.is_empty() {
                otsu_value(&src)?
            } else {
                values.sort_unstable();
                numpy_linear_percentile(&values, p.clamp(0.0, 100.0)) as f64
            }
        }
        ThresholdMode::AdaptiveMean { .. } | ThresholdMode::AdaptiveGaussian { .. } => {
            return Err(VrifaError::InvalidParameter(
                "adaptive thresholding cannot be expressed as a single value; \
                 callers must use threshold::apply_threshold for adaptive modes"
                    .into(),
            ));
        }
    };
    let value = raw + offset as f64;
    Ok(value.clamp(0.0, 255.0) as f32)
}

fn triangle_value(src: &core::Mat) -> Result<f64> {
    let mut dst = core::Mat::default();
    Ok(imgproc::threshold(
        src,
        &mut dst,
        0.0,
        255.0,
        imgproc::THRESH_BINARY | imgproc::THRESH_TRIANGLE,
    )?)
}

fn numpy_linear_percentile(sorted: &[u8], percentile: f32) -> f32 {
    if sorted.len() == 1 {
        return sorted[0] as f32;
    }
    let rank = percentile / 100.0 * (sorted.len() - 1) as f32;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        sorted[lower] as f32
    } else {
        let weight = rank - lower as f32;
        sorted[lower] as f32 * (1.0 - weight) + sorted[upper] as f32 * weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn count_white(mask: &Array2<u8>) -> usize {
        mask.iter().filter(|v| **v == 255).count()
    }

    #[test]
    fn parse_modes() {
        assert_eq!(ThresholdMode::parse("otsu").unwrap(), ThresholdMode::Otsu);
        assert_eq!(
            ThresholdMode::parse("triangle").unwrap(),
            ThresholdMode::Triangle
        );
        assert_eq!(
            ThresholdMode::parse("manual:42.5").unwrap(),
            ThresholdMode::Manual(42.5)
        );
        assert_eq!(
            ThresholdMode::parse("percentile:80").unwrap(),
            ThresholdMode::Percentile(80.0)
        );
        assert_eq!(
            ThresholdMode::parse("adaptive-mean:21:5").unwrap(),
            ThresholdMode::AdaptiveMean { block_size: 21, c: 5.0 }
        );
        assert_eq!(
            ThresholdMode::parse("adaptive-gaussian:31:-2.5").unwrap(),
            ThresholdMode::AdaptiveGaussian { block_size: 31, c: -2.5 }
        );
        // Round-trip through Display.
        assert_eq!(format!("{}", ThresholdMode::Otsu), "otsu");
        assert_eq!(format!("{}", ThresholdMode::Triangle), "triangle");
    }

    #[test]
    fn parse_errors() {
        assert!(ThresholdMode::parse("").is_err());
        assert!(ThresholdMode::parse("nope").is_err());
        assert!(ThresholdMode::parse("otsu:5").is_err());
        assert!(ThresholdMode::parse("manual").is_err());
        assert!(ThresholdMode::parse("adaptive-mean:21").is_err());
        assert!(ThresholdMode::parse("percentile:200").is_err());
        assert!(ThresholdMode::parse("adaptive-mean:1:0").is_err());
    }

    #[test]
    fn manual_threshold_above() {
        // 4x4: half pixels = 100, half = 200. Manual threshold 150 -> 200s win.
        let delta = array![
            [100, 100, 200, 200],
            [100, 100, 200, 200],
            [100, 100, 200, 200],
            [100, 100, 200, 200],
        ];
        let roi = Array2::<u8>::from_elem((4, 4), 1);
        let out =
            apply_threshold(&delta.mapv(|v| v as u8), &roi, ThresholdMode::Manual(150.0), 0.0)
                .unwrap();
        assert_eq!(count_white(&out), 8);
    }

    #[test]
    fn manual_offset_shifts_threshold() {
        let delta = array![
            [100, 100, 200, 200],
            [100, 100, 200, 200],
            [100, 100, 200, 200],
            [100, 100, 200, 200],
        ];
        let roi = Array2::<u8>::from_elem((4, 4), 1);
        // Manual 250 with offset -150 = effective 100. > 100 -> only the 200s.
        let out = apply_threshold(
            &delta.mapv(|v| v as u8),
            &roi,
            ThresholdMode::Manual(250.0),
            -150.0,
        )
        .unwrap();
        assert_eq!(count_white(&out), 8);
    }

    #[test]
    fn otsu_finds_bimodal_split() {
        // Bimodal distribution: a clear dark cluster around 50, a clear bright
        // cluster around 200. Otsu should put the threshold between them and
        // produce roughly half the pixels white.
        let mut delta = Array2::<u8>::zeros((10, 10));
        for ((y, x), v) in delta.indexed_iter_mut() {
            *v = if (y + x) % 2 == 0 { 50 } else { 200 };
        }
        let roi = Array2::<u8>::from_elem((10, 10), 1);
        let out = apply_threshold(&delta, &roi, ThresholdMode::Otsu, 0.0).unwrap();
        assert_eq!(count_white(&out), 50);
    }

    #[test]
    fn triangle_finds_skewed_threshold() {
        // Skewed histogram: most pixels around 30, a small bright minority at 220.
        // Triangle should isolate the bright minority.
        let mut delta = Array2::<u8>::from_elem((10, 10), 30);
        for x in 0..2 {
            for y in 0..10 {
                delta[(y, x)] = 220;
            }
        }
        let roi = Array2::<u8>::from_elem((10, 10), 1);
        let out = apply_threshold(&delta, &roi, ThresholdMode::Triangle, 0.0).unwrap();
        let white = count_white(&out);
        // Triangle should pick up exactly the 20 bright pixels (or close).
        assert!(
            (15..=30).contains(&white),
            "triangle should isolate the bright minority, got {white}"
        );
    }

    #[test]
    fn percentile_threshold() {
        // 100 pixels uniformly 0..=99. P=80 should put threshold near 79.
        let mut delta = Array2::<u8>::zeros((10, 10));
        for ((y, x), v) in delta.indexed_iter_mut() {
            *v = (y * 10 + x) as u8;
        }
        let roi = Array2::<u8>::from_elem((10, 10), 1);
        let out =
            apply_threshold(&delta, &roi, ThresholdMode::Percentile(80.0), 0.0).unwrap();
        // Roughly the top 20% should be white.
        let white = count_white(&out);
        assert!((15..=22).contains(&white), "got {white}");
    }

    #[test]
    fn adaptive_mean_local_window() {
        // Gradient from 0 (left) to 100 (right). A global Otsu would split
        // somewhere in the middle. Adaptive-mean with a small window should
        // produce roughly half-and-half within each row regardless of
        // absolute brightness.
        let mut delta = Array2::<u8>::zeros((10, 10));
        for ((_, x), v) in delta.indexed_iter_mut() {
            *v = (x * 10) as u8;
        }
        let roi = Array2::<u8>::from_elem((10, 10), 1);
        let out = apply_threshold(
            &delta,
            &roi,
            ThresholdMode::AdaptiveMean { block_size: 3, c: 0.0 },
            0.0,
        )
        .unwrap();
        // Just verify it produced a valid mask and didn't crash.
        assert_eq!(out.dim(), (10, 10));
        for v in out.iter() {
            assert!(*v == 0 || *v == 255);
        }
    }

    #[test]
    fn adaptive_gaussian_block_forced_odd() {
        // Pass even block size; must not panic and must produce valid mask.
        let delta = Array2::<u8>::from_elem((20, 20), 100);
        let roi = Array2::<u8>::from_elem((20, 20), 1);
        let out = apply_threshold(
            &delta,
            &roi,
            ThresholdMode::AdaptiveGaussian { block_size: 8, c: 1.0 },
            0.0,
        )
        .unwrap();
        assert_eq!(out.dim(), (20, 20));
    }

    #[test]
    fn percentile_with_empty_roi_falls_back_to_otsu() {
        let delta = array![[50_u8, 50, 200, 200], [50, 50, 200, 200]];
        let empty_roi = Array2::<u8>::zeros((2, 4));
        // No pixels in ROI -> percentile should fall back to global Otsu.
        let out =
            apply_threshold(&delta, &empty_roi, ThresholdMode::Percentile(50.0), 0.0).unwrap();
        // Just verify it returned a binary mask of the right shape.
        assert_eq!(out.dim(), (2, 4));
    }
}
