use crate::{cvutil, Result, VrifaError};
use ndarray::{Array2, Array3};
use opencv::core::{self, Size};
use opencv::imgproc;
use opencv::prelude::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlurKind {
    None,
    Flat,
    Gaussian,
    Triangle,
    Median,
    Bilateral,
}

impl BlurKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Flat => "flat",
            Self::Gaussian => "gaussian",
            Self::Triangle => "triangle",
            Self::Median => "median",
            Self::Bilateral => "bilateral",
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "none" | "off" | "skip" => Ok(Self::None),
            "flat" | "box" | "mean" => Ok(Self::Flat),
            "gaussian" | "gauss" => Ok(Self::Gaussian),
            "triangle" | "tent" => Ok(Self::Triangle),
            "median" => Ok(Self::Median),
            "bilateral" => Ok(Self::Bilateral),
            other => Err(VrifaError::InvalidParameter(format!(
                "unknown blur kind {other:?}; expected one of none, flat, gaussian, triangle, median, bilateral"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlurSpec {
    pub kind: BlurKind,
    pub size: usize,
}

impl BlurSpec {
    pub const NONE: Self = Self {
        kind: BlurKind::None,
        size: 0,
    };

    pub fn gaussian(size: usize) -> Self {
        Self {
            kind: BlurKind::Gaussian,
            size,
        }
    }
    pub fn flat(size: usize) -> Self {
        Self {
            kind: BlurKind::Flat,
            size,
        }
    }
    pub fn triangle(size: usize) -> Self {
        Self {
            kind: BlurKind::Triangle,
            size,
        }
    }
    pub fn median(size: usize) -> Self {
        Self {
            kind: BlurKind::Median,
            size,
        }
    }
    pub fn bilateral(size: usize) -> Self {
        Self {
            kind: BlurKind::Bilateral,
            size,
        }
    }
    pub fn none() -> Self {
        Self::NONE
    }

    pub fn parse(value: &str) -> Result<Self> {
        let value = value.trim();
        if value.is_empty() {
            return Err(VrifaError::InvalidParameter(
                "blur spec is empty; expected KIND[:SIZE]".into(),
            ));
        }
        let (kind_str, size_str) = match value.split_once(':') {
            Some((k, s)) => (k, Some(s)),
            None => (value, None),
        };
        let kind = BlurKind::parse(kind_str)?;
        if matches!(kind, BlurKind::None) {
            return Ok(Self { kind, size: 0 });
        }
        let size_str = size_str.ok_or_else(|| {
            VrifaError::InvalidParameter(format!(
                "blur kind {} requires a size, e.g. {}:9",
                kind.name(),
                kind.name()
            ))
        })?;
        let size = size_str.trim().parse::<usize>().map_err(|e| {
            VrifaError::InvalidParameter(format!(
                "blur size must be a non-negative integer, got {size_str:?}: {e}"
            ))
        })?;
        if size == 0 {
            return Err(VrifaError::InvalidParameter(
                "blur size must be > 0 for non-none kinds; pass `none` to disable".into(),
            ));
        }
        Ok(Self { kind, size })
    }

    fn odd_size(self) -> i32 {
        let mut s = self.size as i32;
        if s % 2 == 0 {
            s += 1;
        }
        if s < 1 {
            s = 1;
        }
        s
    }

    pub fn is_no_op(self) -> bool {
        matches!(self.kind, BlurKind::None) || self.size <= 1
    }
}

impl std::fmt::Display for BlurSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind {
            BlurKind::None => f.write_str("none"),
            _ => write!(f, "{}:{}", self.kind.name(), self.size),
        }
    }
}

pub fn blur_plane(plane: &Array2<f32>, spec: BlurSpec) -> Result<Array2<f32>> {
    if spec.is_no_op() {
        return Ok(plane.clone());
    }
    let src = cvutil::array2_f32_to_mat(plane)?;
    let k = spec.odd_size();
    let dst = run_blur(&src, spec.kind, k)?;
    cvutil::mat_to_array2_f32(&dst)
}

pub fn blur_frame(frame: &Array3<f32>, spec: BlurSpec) -> Result<Array3<f32>> {
    if spec.is_no_op() {
        return Ok(frame.clone());
    }
    let (h, w, c) = frame.dim();
    let mut out = Array3::<f32>::zeros((h, w, c));
    for ch in 0..c {
        let plane = blur_plane(&frame.slice(ndarray::s![.., .., ch]).to_owned(), spec)?;
        out.slice_mut(ndarray::s![.., .., ch]).assign(&plane);
    }
    Ok(out)
}

fn run_blur(src: &core::Mat, kind: BlurKind, k: i32) -> Result<core::Mat> {
    match kind {
        BlurKind::None => Ok(src.try_clone()?),
        BlurKind::Gaussian => {
            let mut dst = core::Mat::default();
            imgproc::gaussian_blur_def(src, &mut dst, Size::new(k, k), 0.0)?;
            Ok(dst)
        }
        BlurKind::Flat => {
            let mut dst = core::Mat::default();
            imgproc::blur_def(src, &mut dst, Size::new(k, k))?;
            Ok(dst)
        }
        BlurKind::Triangle => {
            let half = (k - 1) / 2;
            let mut weights = Vec::<f32>::with_capacity(k as usize);
            let mut sum = 0.0f32;
            for i in 0..k {
                let w = ((half + 1) - (i - half).abs()) as f32;
                weights.push(w);
                sum += w;
            }
            for w in weights.iter_mut() {
                *w /= sum;
            }
            let kernel = core::Mat::from_slice(&weights)?.try_clone()?;
            let mut dst = core::Mat::default();
            imgproc::sep_filter_2d_def(src, &mut dst, core::CV_32F, &kernel, &kernel)?;
            Ok(dst)
        }
        BlurKind::Median => {
            if k > 5 {
                return Err(VrifaError::InvalidParameter(format!(
                    "median blur on f32 supports only size 3 and 5; got {k}. \
                     For larger kernels, apply median to a u8 view of the data."
                )));
            }
            let mut dst = core::Mat::default();
            imgproc::median_blur(src, &mut dst, k)?;
            Ok(dst)
        }
        BlurKind::Bilateral => {
            let mut dst = core::Mat::default();
            imgproc::bilateral_filter(src, &mut dst, k, k as f64, k as f64, core::BORDER_DEFAULT)?;
            Ok(dst)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    fn arrays_close(a: &Array2<f32>, b: &Array2<f32>, tol: f32) -> bool {
        if a.dim() != b.dim() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| approx_eq(*x, *y, tol))
    }

    #[test]
    fn parse_kinds() {
        assert_eq!(BlurKind::parse("none").unwrap(), BlurKind::None);
        assert_eq!(BlurKind::parse("OFF").unwrap(), BlurKind::None);
        assert_eq!(BlurKind::parse("flat").unwrap(), BlurKind::Flat);
        assert_eq!(BlurKind::parse("box").unwrap(), BlurKind::Flat);
        assert_eq!(BlurKind::parse("Gaussian").unwrap(), BlurKind::Gaussian);
        assert_eq!(BlurKind::parse("triangle").unwrap(), BlurKind::Triangle);
        assert_eq!(BlurKind::parse("tent").unwrap(), BlurKind::Triangle);
        assert_eq!(BlurKind::parse("median").unwrap(), BlurKind::Median);
        assert_eq!(BlurKind::parse("bilateral").unwrap(), BlurKind::Bilateral);
        assert!(BlurKind::parse("unknown").is_err());
    }

    #[test]
    fn parse_spec_kind_size() {
        let s = BlurSpec::parse("gaussian:9").unwrap();
        assert_eq!(s.kind, BlurKind::Gaussian);
        assert_eq!(s.size, 9);

        let s = BlurSpec::parse("flat:5").unwrap();
        assert_eq!(s.kind, BlurKind::Flat);
        assert_eq!(s.size, 5);

        let s = BlurSpec::parse("none").unwrap();
        assert_eq!(s.kind, BlurKind::None);
        assert_eq!(s.size, 0);

        let s = BlurSpec::parse("none:0").unwrap();
        assert_eq!(s.kind, BlurKind::None);

        // Round-trip through Display.
        assert_eq!(format!("{}", BlurSpec::gaussian(9)), "gaussian:9");
        assert_eq!(format!("{}", BlurSpec::none()), "none");
    }

    #[test]
    fn parse_spec_errors() {
        assert!(BlurSpec::parse("").is_err());
        assert!(BlurSpec::parse("gaussian").is_err()); // size required
        assert!(BlurSpec::parse("gaussian:0").is_err()); // size must be > 0
        assert!(BlurSpec::parse("gaussian:abc").is_err());
        assert!(BlurSpec::parse("notakind:9").is_err());
    }

    #[test]
    fn none_is_identity() {
        let plane = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let out = blur_plane(&plane, BlurSpec::none()).unwrap();
        assert!(arrays_close(&plane, &out, 0.0));
    }

    #[test]
    fn size_one_is_identity() {
        let plane = array![[1.0, 2.0], [3.0, 4.0]];
        for kind in [BlurKind::Flat, BlurKind::Gaussian, BlurKind::Triangle] {
            let out = blur_plane(&plane, BlurSpec { kind, size: 1 }).unwrap();
            assert!(
                arrays_close(&plane, &out, 0.0),
                "kind {:?} with size 1 should be identity",
                kind
            );
        }
    }

    #[test]
    fn flat_3x3_known_value() {
        // 5x5 input, 3x3 box filter. Center pixel value = mean of 9 neighbors.
        let plane = array![
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0, 19.0, 20.0],
            [21.0, 22.0, 23.0, 24.0, 25.0],
        ];
        let out = blur_plane(&plane, BlurSpec::flat(3)).unwrap();
        // Center cell (2,2) = (7+8+9 + 12+13+14 + 17+18+19) / 9 = 117 / 9 = 13.0.
        assert!(approx_eq(out[(2, 2)], 13.0, 1e-5), "got {}", out[(2, 2)]);
    }

    #[test]
    fn triangle_3x3_known_value() {
        // 5x5 input, 3x3 triangle filter.
        // 1-D weights for k=3: half=1, w = [(1+1)-|0-1|, (1+1)-|1-1|, (1+1)-|2-1|]
        //                     = [1, 2, 1] before normalization, sum = 4, so [0.25, 0.5, 0.25].
        // 2-D weights = outer(w,w), sum still 1.
        // Center cell value = sum_{di,dj} w(di)*w(dj) * input(2+di, 2+dj)
        //   shorthand: w = (0.25, 0.5, 0.25); 2D weights:
        //     0.0625 0.125 0.0625
        //     0.125  0.25  0.125
        //     0.0625 0.125 0.0625
        // Apply to 3x3 patch around (2,2):
        //   7  8  9
        //   12 13 14
        //   17 18 19
        // expected = 0.0625*(7+9+17+19) + 0.125*(8+12+14+18) + 0.25*13
        //          = 0.0625*52 + 0.125*52 + 0.25*13
        //          = 3.25 + 6.5 + 3.25 = 13.0
        let plane = array![
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0, 19.0, 20.0],
            [21.0, 22.0, 23.0, 24.0, 25.0],
        ];
        let out = blur_plane(&plane, BlurSpec::triangle(3)).unwrap();
        assert!(approx_eq(out[(2, 2)], 13.0, 1e-5), "got {}", out[(2, 2)]);
    }

    #[test]
    fn gaussian_3x3_preserves_constant() {
        // A constant plane stays constant under any positive-weight blur.
        let plane = Array2::from_elem((7, 7), 3.7f32);
        let out = blur_plane(&plane, BlurSpec::gaussian(3)).unwrap();
        for v in out.iter() {
            assert!(approx_eq(*v, 3.7, 1e-5), "got {v}");
        }
    }

    #[test]
    fn gaussian_lowers_variance() {
        // Random-ish input, gaussian should reduce variance vs the input.
        let plane = array![
            [0.0, 10.0, 0.0, 10.0, 0.0],
            [10.0, 0.0, 10.0, 0.0, 10.0],
            [0.0, 10.0, 0.0, 10.0, 0.0],
            [10.0, 0.0, 10.0, 0.0, 10.0],
            [0.0, 10.0, 0.0, 10.0, 0.0],
        ];
        let out = blur_plane(&plane, BlurSpec::gaussian(5)).unwrap();
        let var_in = variance(&plane);
        let var_out = variance(&out);
        assert!(
            var_out < var_in * 0.5,
            "gaussian should reduce variance significantly: in={var_in}, out={var_out}"
        );
    }

    fn variance(a: &Array2<f32>) -> f32 {
        let n = a.len() as f32;
        let mean = a.iter().sum::<f32>() / n;
        a.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n
    }

    #[test]
    fn median_3_known_value() {
        // 5x5 input. 3x3 median around (2,2) covers values 7..19 minus the corners.
        // Sorted 3x3 patch around (2,2): 7, 8, 9, 12, 13, 14, 17, 18, 19. Median = 13.
        let plane = array![
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0, 19.0, 20.0],
            [21.0, 22.0, 23.0, 24.0, 25.0],
        ];
        let out = blur_plane(&plane, BlurSpec::median(3)).unwrap();
        assert!(approx_eq(out[(2, 2)], 13.0, 1e-5), "got {}", out[(2, 2)]);
    }

    #[test]
    fn median_oversize_errors_for_f32() {
        let plane = Array2::<f32>::zeros((11, 11));
        let res = blur_plane(&plane, BlurSpec::median(7));
        assert!(res.is_err(), "median k=7 should be rejected on f32");
    }

    #[test]
    fn frame_blurs_each_channel() {
        // Build a 3-channel constant frame; each plane should remain constant after blur.
        let mut frame = Array3::<f32>::zeros((5, 5, 3));
        for ch in 0..3 {
            frame
                .slice_mut(ndarray::s![.., .., ch])
                .fill((ch as f32 + 1.0) * 2.0);
        }
        let out = blur_frame(&frame, BlurSpec::gaussian(3)).unwrap();
        for ch in 0..3 {
            let expected = (ch as f32 + 1.0) * 2.0;
            for v in out.slice(ndarray::s![.., .., ch]).iter() {
                assert!(approx_eq(*v, expected, 1e-5), "ch {ch}: got {v}");
            }
        }
    }
}
