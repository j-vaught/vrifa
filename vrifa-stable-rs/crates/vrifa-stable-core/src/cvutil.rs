use crate::{Result, VrifaError};
use ndarray::{Array2, Array3};
use opencv::core::{self, Mat, Scalar};
use opencv::prelude::*;

fn require_contiguous<T>(slice: Option<&[T]>) -> Result<&[T]> {
    slice.ok_or_else(|| VrifaError::Shape("array is not contiguous in memory order".to_string()))
}

pub fn array3_u8_to_mat(array: &Array3<u8>) -> Result<Mat> {
    let (height, width, channels) = array.dim();
    let typ = core::CV_MAKETYPE(core::CV_8U, channels as i32);
    let mut mat =
        Mat::new_rows_cols_with_default(height as i32, width as i32, typ, Scalar::default())?;
    mat.data_bytes_mut()?
        .copy_from_slice(require_contiguous(array.as_slice_memory_order())?);
    Ok(mat)
}

pub fn array2_u8_to_mat(array: &Array2<u8>) -> Result<Mat> {
    let (height, width) = array.dim();
    let mut mat = Mat::new_rows_cols_with_default(
        height as i32,
        width as i32,
        core::CV_8UC1,
        Scalar::default(),
    )?;
    mat.data_bytes_mut()?
        .copy_from_slice(require_contiguous(array.as_slice_memory_order())?);
    Ok(mat)
}

pub fn array2_f32_to_mat(array: &Array2<f32>) -> Result<Mat> {
    let (height, width) = array.dim();
    let mut mat = Mat::new_rows_cols_with_default(
        height as i32,
        width as i32,
        core::CV_32FC1,
        Scalar::default(),
    )?;
    mat.data_typed_mut::<f32>()?
        .copy_from_slice(require_contiguous(array.as_slice_memory_order())?);
    Ok(mat)
}

pub fn array3_f32_to_mat(array: &Array3<f32>) -> Result<Mat> {
    let (height, width, channels) = array.dim();
    let typ = core::CV_MAKETYPE(core::CV_32F, channels as i32);
    let mut mat =
        Mat::new_rows_cols_with_default(height as i32, width as i32, typ, Scalar::default())?;
    mat.data_typed_mut::<f32>()?
        .copy_from_slice(require_contiguous(array.as_slice_memory_order())?);
    Ok(mat)
}

pub fn array3_f32_channel_to_mat(array: &Array3<f32>, channel: usize) -> Result<Mat> {
    let (_, _, channels) = array.dim();
    if channel >= channels {
        return Err(VrifaError::Shape(format!(
            "requested channel {channel} but frame has {channels} channel(s)"
        )));
    }
    let plane = array.slice(ndarray::s![.., .., channel]).to_owned();
    array2_f32_to_mat(&plane)
}

pub fn mat_to_array3_u8(mat: &Mat) -> Result<Array3<u8>> {
    let rows = mat.rows() as usize;
    let cols = mat.cols() as usize;
    let channels = mat.channels() as usize;
    let expected = rows * cols * channels;
    let bytes = mat.data_bytes()?;
    if bytes.len() < expected {
        return Err(VrifaError::Shape(format!(
            "mat has {} bytes, expected at least {}",
            bytes.len(),
            expected
        )));
    }
    Array3::from_shape_vec((rows, cols, channels), bytes[..expected].to_vec())
        .map_err(|err| VrifaError::Shape(err.to_string()))
}

pub fn mat_to_array2_u8(mat: &Mat) -> Result<Array2<u8>> {
    let rows = mat.rows() as usize;
    let cols = mat.cols() as usize;
    let channels = mat.channels();
    if channels != 1 {
        return Err(VrifaError::Shape(format!(
            "expected single-channel mat, got {} channels",
            channels
        )));
    }
    let expected = rows * cols;
    let bytes = mat.data_bytes()?;
    if bytes.len() < expected {
        return Err(VrifaError::Shape(format!(
            "mat has {} bytes, expected at least {}",
            bytes.len(),
            expected
        )));
    }
    Array2::from_shape_vec((rows, cols), bytes[..expected].to_vec())
        .map_err(|err| VrifaError::Shape(err.to_string()))
}

pub fn mat_to_array2_f32(mat: &Mat) -> Result<Array2<f32>> {
    let rows = mat.rows() as usize;
    let cols = mat.cols() as usize;
    let channels = mat.channels();
    if channels != 1 {
        return Err(VrifaError::Shape(format!(
            "expected single-channel mat, got {} channels",
            channels
        )));
    }
    let expected = rows * cols;
    let values = mat.data_typed::<f32>()?;
    if values.len() < expected {
        return Err(VrifaError::Shape(format!(
            "mat has {} f32 values, expected at least {}",
            values.len(),
            expected
        )));
    }
    Array2::from_shape_vec((rows, cols), values[..expected].to_vec())
        .map_err(|err| VrifaError::Shape(err.to_string()))
}

pub fn mat_to_array3_f32(mat: &Mat) -> Result<Array3<f32>> {
    let rows = mat.rows() as usize;
    let cols = mat.cols() as usize;
    let channels = mat.channels() as usize;
    let expected = rows * cols * channels;
    let values = mat.data_typed::<f32>()?;
    if values.len() < expected {
        return Err(VrifaError::Shape(format!(
            "mat has {} f32 values, expected at least {}",
            values.len(),
            expected
        )));
    }
    Array3::from_shape_vec((rows, cols, channels), values[..expected].to_vec())
        .map_err(|err| VrifaError::Shape(err.to_string()))
}
