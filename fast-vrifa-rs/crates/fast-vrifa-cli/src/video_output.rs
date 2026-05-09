use anyhow::{anyhow, bail, Context, Result};
use ndarray::{Array2, Array3};
use std::env;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use vrifa_io::AsyncVideoWriter;

enum VideoFrame {
    Bgr(Arc<Array3<u8>>),
    Gray(Arc<Array2<u8>>),
}

pub enum VideoOutputWriter {
    OpenCv(AsyncVideoWriter),
    Ffmpeg(AsyncFfmpegWriter),
}

pub struct AsyncFfmpegWriter {
    sender: SyncSender<VideoFrame>,
    handle: JoinHandle<Result<()>>,
    is_color: bool,
}

impl VideoOutputWriter {
    pub fn open(
        path: impl AsRef<Path>,
        fps: f64,
        width: usize,
        height: usize,
        is_color: bool,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if is_color {
            if let Ok(writer) = AsyncFfmpegWriter::open(path.clone(), fps, width, height, true) {
                return Ok(Self::Ffmpeg(writer));
            }
        }
        Ok(Self::OpenCv(AsyncVideoWriter::open(
            path, fps, width, height, is_color,
        )?))
    }

    pub fn write_bgr(&self, frame: Arc<Array3<u8>>) -> Result<()> {
        match self {
            Self::OpenCv(writer) => writer.write_bgr(frame),
            Self::Ffmpeg(writer) => writer.write_bgr(frame),
        }
    }

    pub fn write_gray(&self, frame: Arc<Array2<u8>>) -> Result<()> {
        match self {
            Self::OpenCv(writer) => writer.write_gray(frame),
            Self::Ffmpeg(writer) => writer.write_gray(frame),
        }
    }

    pub fn close(self) -> Result<()> {
        match self {
            Self::OpenCv(writer) => writer.close(),
            Self::Ffmpeg(writer) => writer.close(),
        }
    }
}

impl AsyncFfmpegWriter {
    pub fn open(
        path: PathBuf,
        fps: f64,
        width: usize,
        height: usize,
        is_color: bool,
    ) -> Result<Self> {
        let encoder = preferred_encoder();
        let child = spawn_ffmpeg(&path, fps, width, height, is_color, &encoder)
            .or_else(|_| spawn_ffmpeg(&path, fps, width, height, is_color, "libx264"))
            .with_context(|| format!("starting ffmpeg video writer for {}", path.display()))?;
        let (sender, receiver) = sync_channel::<VideoFrame>(16);
        let handle = thread::spawn(move || -> Result<()> {
            let mut child = child;
            let mut stdin = child
                .stdin
                .take()
                .ok_or_else(|| anyhow!("ffmpeg stdin pipe was not available"))?;
            for frame in receiver {
                match frame {
                    VideoFrame::Bgr(frame) => {
                        if !is_color {
                            bail!("attempted to send color data to grayscale ffmpeg writer");
                        }
                        let bytes = frame
                            .as_slice_memory_order()
                            .ok_or_else(|| anyhow!("BGR frame must be contiguous"))?;
                        stdin
                            .write_all(bytes)
                            .context("writing BGR frame to ffmpeg")?;
                    }
                    VideoFrame::Gray(frame) => {
                        if is_color {
                            bail!("attempted to send grayscale data to color ffmpeg writer");
                        }
                        let bytes = frame
                            .as_slice_memory_order()
                            .ok_or_else(|| anyhow!("gray frame must be contiguous"))?;
                        stdin
                            .write_all(bytes)
                            .context("writing grayscale frame to ffmpeg")?;
                    }
                }
            }
            drop(stdin);
            let output = child
                .wait_with_output()
                .context("waiting for ffmpeg to exit")?;
            if !output.status.success() {
                bail!(
                    "ffmpeg exited with {}: {}",
                    output.status,
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            Ok(())
        });
        Ok(Self {
            sender,
            handle,
            is_color,
        })
    }

    pub fn write_bgr(&self, frame: Arc<Array3<u8>>) -> Result<()> {
        if !self.is_color {
            bail!("writer was opened as grayscale");
        }
        self.sender
            .send(VideoFrame::Bgr(frame))
            .map_err(|err| anyhow!("ffmpeg writer thread stopped: {err}"))
    }

    pub fn write_gray(&self, frame: Arc<Array2<u8>>) -> Result<()> {
        if self.is_color {
            bail!("writer was opened as color");
        }
        self.sender
            .send(VideoFrame::Gray(frame))
            .map_err(|err| anyhow!("ffmpeg writer thread stopped: {err}"))
    }

    pub fn close(self) -> Result<()> {
        drop(self.sender);
        self.handle
            .join()
            .map_err(|_| anyhow!("ffmpeg writer thread panicked"))?
    }
}

fn preferred_encoder() -> String {
    env::var("FAST_VRIFA_FFMPEG_ENCODER").unwrap_or_else(|_| "h264_nvenc".to_string())
}

fn spawn_ffmpeg(
    path: &Path,
    fps: f64,
    width: usize,
    height: usize,
    is_color: bool,
    encoder: &str,
) -> Result<Child> {
    let pix_fmt = if is_color { "bgr24" } else { "gray" };
    let size = format!("{width}x{height}");
    let fps = if fps > 0.0 { fps } else { 30.0 };
    let fps_text = format!("{fps:.6}");
    let mut args = vec![
        "-y".to_string(),
        "-f".to_string(),
        "rawvideo".to_string(),
        "-pix_fmt".to_string(),
        pix_fmt.to_string(),
        "-s:v".to_string(),
        size,
        "-r".to_string(),
        fps_text,
        "-i".to_string(),
        "-".to_string(),
        "-an".to_string(),
        "-c:v".to_string(),
        encoder.to_string(),
    ];
    if encoder == "h264_nvenc" {
        args.extend([
            "-preset".to_string(),
            "p1".to_string(),
            "-cq".to_string(),
            "19".to_string(),
            "-b:v".to_string(),
            "0".to_string(),
            "-pix_fmt".to_string(),
            "yuv420p".to_string(),
        ]);
    } else if encoder == "libx264" {
        args.extend([
            "-preset".to_string(),
            "ultrafast".to_string(),
            "-crf".to_string(),
            "18".to_string(),
            "-pix_fmt".to_string(),
            "yuv420p".to_string(),
        ]);
    }
    args.push(path.to_string_lossy().into_owned());
    Command::new("ffmpeg")
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("spawning ffmpeg for {}", path.display()))
}
