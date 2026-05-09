use anyhow::{anyhow, Context, Result};
use ndarray::{Array2, Array3};
use opencv::core::{Mat, Size};
use opencv::imgcodecs;
use opencv::prelude::*;
use opencv::videoio;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::thread::{self, JoinHandle};
use vrifa_core::cvutil;

pub type BgrFrame = Array3<u8>;

#[derive(Clone, Copy, Debug)]
pub struct VideoMetadata {
    pub total_frames: Option<usize>,
    pub fps: f64,
    pub width: usize,
    pub height: usize,
}

pub struct VideoReader {
    capture: videoio::VideoCapture,
    metadata: VideoMetadata,
    next_index: usize,
}

impl VideoReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let capture = videoio::VideoCapture::from_file_def(&path.to_string_lossy())
            .with_context(|| format!("opening video {}", path.display()))?;
        if !capture.is_opened()? {
            return Err(anyhow!("unable to open video: {}", path.display()));
        }
        let total_frames = capture.get(videoio::CAP_PROP_FRAME_COUNT)? as usize;
        let fps = capture.get(videoio::CAP_PROP_FPS)?;
        let width = capture.get(videoio::CAP_PROP_FRAME_WIDTH)? as usize;
        let height = capture.get(videoio::CAP_PROP_FRAME_HEIGHT)? as usize;
        Ok(Self {
            capture,
            metadata: VideoMetadata {
                total_frames: (total_frames > 0).then_some(total_frames),
                fps: if fps > 0.0 { fps } else { 30.0 },
                width,
                height,
            },
            next_index: 1,
        })
    }

    pub fn metadata(&self) -> VideoMetadata {
        self.metadata
    }

    pub fn seek_zero(&mut self) -> Result<()> {
        self.capture.set(videoio::CAP_PROP_POS_FRAMES, 0.0)?;
        self.next_index = 1;
        Ok(())
    }

    pub fn read_next(&mut self) -> Result<Option<(usize, BgrFrame)>> {
        let mut mat = Mat::default();
        if !self.capture.read(&mut mat)? || mat.empty() {
            return Ok(None);
        }
        let index = self.next_index;
        self.next_index += 1;
        Ok(Some((index, cvutil::mat_to_array3_u8(&mat)?)))
    }

    pub fn read_frame_at(&mut self, one_based_index: usize) -> Result<Option<BgrFrame>> {
        if one_based_index == 0 {
            return Ok(None);
        }
        self.capture
            .set(videoio::CAP_PROP_POS_FRAMES, (one_based_index - 1) as f64)?;
        self.next_index = one_based_index;
        self.read_next().map(|maybe| maybe.map(|(_, frame)| frame))
    }

    pub fn read_frame_at_zero_based(
        &mut self,
        zero_based_index: usize,
    ) -> Result<Option<BgrFrame>> {
        self.capture
            .set(videoio::CAP_PROP_POS_FRAMES, zero_based_index as f64)?;
        self.next_index = zero_based_index + 1;
        self.read_next().map(|maybe| maybe.map(|(_, frame)| frame))
    }
}

pub struct VideoWriter {
    writer: videoio::VideoWriter,
    is_color: bool,
}

enum VideoFrame {
    Bgr(BgrFrame),
    Gray(Array2<u8>),
}

enum PngFrame {
    Bgr { path: PathBuf, frame: BgrFrame },
    Gray { path: PathBuf, frame: Array2<u8> },
}

pub struct AsyncVideoWriter {
    sender: SyncSender<VideoFrame>,
    handle: JoinHandle<Result<()>>,
    is_color: bool,
}

pub struct AsyncPngWriter {
    sender: SyncSender<PngFrame>,
    handle: JoinHandle<Result<()>>,
    is_color: bool,
}

impl AsyncVideoWriter {
    pub fn open(
        path: impl AsRef<Path>,
        fps: f64,
        width: usize,
        height: usize,
        is_color: bool,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let (sender, receiver) = sync_channel::<VideoFrame>(4);
        let handle = thread::spawn(move || -> Result<()> {
            let mut writer = VideoWriter::open(path, fps, width, height, is_color)?;
            for frame in receiver {
                match frame {
                    VideoFrame::Bgr(frame) => writer.write_bgr(&frame)?,
                    VideoFrame::Gray(frame) => writer.write_gray(&frame)?,
                }
            }
            writer.release()?;
            Ok(())
        });
        Ok(Self {
            sender,
            handle,
            is_color,
        })
    }

    pub fn write_bgr(&self, frame: BgrFrame) -> Result<()> {
        if !self.is_color {
            return Err(anyhow!("writer was opened as grayscale"));
        }
        self.sender
            .send(VideoFrame::Bgr(frame))
            .map_err(|err| anyhow!("video writer thread stopped: {err}"))
    }

    pub fn write_gray(&self, frame: Array2<u8>) -> Result<()> {
        if self.is_color {
            return Err(anyhow!("writer was opened as color"));
        }
        self.sender
            .send(VideoFrame::Gray(frame))
            .map_err(|err| anyhow!("video writer thread stopped: {err}"))
    }

    pub fn close(self) -> Result<()> {
        drop(self.sender);
        self.handle
            .join()
            .map_err(|_| anyhow!("video writer thread panicked"))?
    }
}

impl AsyncPngWriter {
    pub fn open(is_color: bool) -> Result<Self> {
        let (sender, receiver) = sync_channel::<PngFrame>(8);
        let handle = thread::spawn(move || -> Result<()> {
            for frame in receiver {
                match frame {
                    PngFrame::Bgr { path, frame } => write_bgr_png_impl(&path, &frame)?,
                    PngFrame::Gray { path, frame } => write_gray_png_impl(&path, &frame)?,
                }
            }
            Ok(())
        });
        Ok(Self {
            sender,
            handle,
            is_color,
        })
    }

    pub fn write_bgr(&self, path: impl AsRef<Path>, frame: BgrFrame) -> Result<()> {
        if !self.is_color {
            return Err(anyhow!("writer was opened as grayscale"));
        }
        self.sender
            .send(PngFrame::Bgr {
                path: path.as_ref().to_path_buf(),
                frame,
            })
            .map_err(|err| anyhow!("png writer thread stopped: {err}"))
    }

    pub fn write_gray(&self, path: impl AsRef<Path>, frame: Array2<u8>) -> Result<()> {
        if self.is_color {
            return Err(anyhow!("writer was opened as color"));
        }
        self.sender
            .send(PngFrame::Gray {
                path: path.as_ref().to_path_buf(),
                frame,
            })
            .map_err(|err| anyhow!("png writer thread stopped: {err}"))
    }

    pub fn close(self) -> Result<()> {
        drop(self.sender);
        self.handle
            .join()
            .map_err(|_| anyhow!("png writer thread panicked"))?
    }
}

impl VideoWriter {
    pub fn open(
        path: impl AsRef<Path>,
        fps: f64,
        width: usize,
        height: usize,
        is_color: bool,
    ) -> Result<Self> {
        let path = path.as_ref();
        let fourcc = videoio::VideoWriter::fourcc('m', 'p', '4', 'v')?;
        let writer = videoio::VideoWriter::new(
            &path.to_string_lossy(),
            fourcc,
            fps,
            Size::new(width as i32, height as i32),
            is_color,
        )
        .with_context(|| format!("opening video writer {}", path.display()))?;
        if !writer.is_opened()? {
            return Err(anyhow!("failed to open video writer: {}", path.display()));
        }
        Ok(Self { writer, is_color })
    }

    pub fn write_bgr(&mut self, frame: &BgrFrame) -> Result<()> {
        if !self.is_color {
            return Err(anyhow!("writer was opened as grayscale"));
        }
        let mat = cvutil::array3_u8_to_mat(frame)?;
        self.writer.write(&mat)?;
        Ok(())
    }

    pub fn write_gray(&mut self, frame: &Array2<u8>) -> Result<()> {
        if self.is_color {
            return Err(anyhow!("writer was opened as color"));
        }
        let mat = cvutil::array2_u8_to_mat(frame)?;
        self.writer.write(&mat)?;
        Ok(())
    }

    pub fn release(&mut self) -> Result<()> {
        self.writer.release()?;
        Ok(())
    }
}

pub fn write_bgr_png(path: impl AsRef<Path>, frame: &BgrFrame) -> Result<()> {
    write_bgr_png_impl(path.as_ref(), frame)
}

pub fn write_gray_png(path: impl AsRef<Path>, frame: &Array2<u8>) -> Result<()> {
    write_gray_png_impl(path.as_ref(), frame)
}

fn write_bgr_png_impl(path: &Path, frame: &BgrFrame) -> Result<()> {
    let (_, _, channels) = frame.dim();
    if channels != 3 {
        return Err(anyhow!(
            "BGR PNG frame must have 3 channels, got {channels}"
        ));
    }
    // Match Python's cv2.imwrite output to keep parity artifacts deterministic.
    let mat = cvutil::array3_u8_to_mat(frame)?;
    imgcodecs::imwrite_def(&path.to_string_lossy(), &mat)
        .with_context(|| format!("writing PNG {}", path.display()))?;
    Ok(())
}

fn write_gray_png_impl(path: &Path, frame: &Array2<u8>) -> Result<()> {
    // Match Python's cv2.imwrite output to keep parity artifacts deterministic.
    let mat = cvutil::array2_u8_to_mat(frame)?;
    imgcodecs::imwrite_def(&path.to_string_lossy(), &mat)
        .with_context(|| format!("writing PNG {}", path.display()))?;
    Ok(())
}
