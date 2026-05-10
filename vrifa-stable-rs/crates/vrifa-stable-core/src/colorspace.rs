use crate::{cvutil, Result, VrifaError};
use ndarray::Array3;
use opencv::imgproc;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ColorSpace {
    Cielab,
    Rgb,
    Hsv,
    Grayscale,
}

impl ColorSpace {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_uppercase().as_str() {
            "CIELAB" | "LAB" => Ok(Self::Cielab),
            "RGB" => Ok(Self::Rgb),
            "HSV" => Ok(Self::Hsv),
            "GRAYSCALE" | "GRAY" => Ok(Self::Grayscale),
            other => Err(VrifaError::InvalidParameter(format!(
                "unsupported colorspace: {other}"
            ))),
        }
    }

    pub fn canonical_name(self) -> &'static str {
        match self {
            Self::Cielab => "CIELAB",
            Self::Rgb => "RGB",
            Self::Hsv => "HSV",
            Self::Grayscale => "GRAYSCALE",
        }
    }

    pub fn channel_count(self) -> usize {
        match self {
            Self::Grayscale => 1,
            _ => 3,
        }
    }
}

pub fn convert_frame_to_colorspace(
    frame_bgr: &Array3<u8>,
    colorspace: ColorSpace,
) -> Result<Array3<u8>> {
    if frame_bgr.dim().2 != 3 {
        return Err(VrifaError::Shape(format!(
            "BGR frame must have 3 channels, got {}",
            frame_bgr.dim().2
        )));
    }

    let src = cvutil::array3_u8_to_mat(frame_bgr)?;
    let mut dst = opencv::core::Mat::default();
    let code = match colorspace {
        ColorSpace::Cielab => imgproc::COLOR_BGR2Lab,
        ColorSpace::Rgb => imgproc::COLOR_BGR2RGB,
        ColorSpace::Hsv => imgproc::COLOR_BGR2HSV,
        ColorSpace::Grayscale => imgproc::COLOR_BGR2GRAY,
    };
    imgproc::cvt_color_def(&src, &mut dst, code)?;

    if colorspace == ColorSpace::Grayscale {
        let gray = cvutil::mat_to_array2_u8(&dst)?;
        let (height, width) = gray.dim();
        let mut out = Array3::<u8>::zeros((height, width, 1));
        out.slice_mut(ndarray::s![.., .., 0]).assign(&gray);
        Ok(out)
    } else {
        cvutil::mat_to_array3_u8(&dst)
    }
}
