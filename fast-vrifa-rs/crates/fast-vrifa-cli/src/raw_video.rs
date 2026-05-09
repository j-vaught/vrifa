use anyhow::{anyhow, bail, Context, Result};
use memmap2::MmapMut;
use ndarray::{Array2, Array3};
use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

pub type BgrFrame = Array3<u8>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RawPixelFormat {
    Gray8,
    Bgr24,
}

impl RawPixelFormat {
    fn frame_bytes(self, width: usize, height: usize) -> Result<usize> {
        let channels = match self {
            Self::Gray8 => 1usize,
            Self::Bgr24 => 3usize,
        };
        width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(channels))
            .ok_or_else(|| anyhow!("raw frame byte size overflow"))
    }

    fn ffmpeg_input_pix_fmt(self) -> &'static str {
        match self {
            Self::Gray8 => "gray",
            Self::Bgr24 => "bgr24",
        }
    }
}

enum RawVideoFrame {
    Bgr(Arc<BgrFrame>),
    Gray(Arc<Array2<u8>>),
}

pub struct RawVideoArtifact {
    pub path: PathBuf,
    pub format: RawPixelFormat,
    pub width: usize,
    pub height: usize,
    pub fps: f64,
    pub frames_written: usize,
}

pub struct RawGrayFrameReader {
    reader: BufReader<File>,
    width: usize,
    height: usize,
    frame_bytes: usize,
}

pub struct AsyncRawVideoWriter {
    sender: SyncSender<RawVideoFrame>,
    handle: JoinHandle<Result<usize>>,
    path: PathBuf,
    format: RawPixelFormat,
    width: usize,
    height: usize,
    fps: f64,
}

impl AsyncRawVideoWriter {
    pub fn open(
        path: impl AsRef<Path>,
        fps: f64,
        width: usize,
        height: usize,
        format: RawPixelFormat,
        expected_frames: Option<usize>,
        queue_capacity: usize,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let (sender, receiver) = sync_channel::<RawVideoFrame>(queue_capacity.max(1));
        let writer_path = path.clone();
        let handle = thread::spawn(move || -> Result<usize> {
            let mut sink =
                RawVideoSinkState::open(&writer_path, width, height, format, expected_frames)?;
            for frame in receiver {
                sink.push(frame)?;
            }
            sink.finish()
        });
        Ok(Self {
            sender,
            handle,
            path,
            format,
            width,
            height,
            fps,
        })
    }

    pub fn write_bgr(&self, frame: Arc<BgrFrame>) -> Result<()> {
        if !matches!(self.format, RawPixelFormat::Bgr24) {
            bail!("raw writer was opened as grayscale");
        }
        self.sender
            .send(RawVideoFrame::Bgr(frame))
            .map_err(|err| anyhow!("raw video writer thread stopped: {err}"))
    }

    pub fn write_gray(&self, frame: Arc<Array2<u8>>) -> Result<()> {
        if !matches!(self.format, RawPixelFormat::Gray8) {
            bail!("raw writer was opened as color");
        }
        self.sender
            .send(RawVideoFrame::Gray(frame))
            .map_err(|err| anyhow!("raw video writer thread stopped: {err}"))
    }

    pub fn close(self) -> Result<RawVideoArtifact> {
        drop(self.sender);
        let frames_written = self
            .handle
            .join()
            .map_err(|_| anyhow!("raw video writer thread panicked"))??;
        Ok(RawVideoArtifact {
            path: self.path,
            format: self.format,
            width: self.width,
            height: self.height,
            fps: self.fps,
            frames_written,
        })
    }
}

enum RawVideoSink {
    Mapped { file: File, map: MmapMut },
    Buffered { writer: BufWriter<File> },
}

impl RawVideoSink {
    fn open(
        path: &Path,
        width: usize,
        height: usize,
        format: RawPixelFormat,
        expected_frames: Option<usize>,
    ) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(path)
            .with_context(|| format!("opening raw stream {}", path.display()))?;
        let frame_bytes = format.frame_bytes(width, height)?;
        if let Some(expected_frames) = expected_frames {
            let total_len = frame_bytes
                .checked_mul(expected_frames)
                .ok_or_else(|| anyhow!("raw stream length overflow"))?;
            file.set_len(total_len as u64)
                .with_context(|| format!("preallocating {}", path.display()))?;
            let map = unsafe { MmapMut::map_mut(&file) }
                .with_context(|| format!("mapping {}", path.display()))?;
            Ok(Self::Mapped { file, map })
        } else {
            Ok(Self::Buffered {
                writer: BufWriter::new(file),
            })
        }
    }
}

struct RawVideoSinkState {
    sink: RawVideoSink,
    frame_bytes: usize,
    frames_written: usize,
}

impl RawVideoSinkState {
    fn open(
        path: &Path,
        width: usize,
        height: usize,
        format: RawPixelFormat,
        expected_frames: Option<usize>,
    ) -> Result<Self> {
        Ok(Self {
            sink: RawVideoSink::open(path, width, height, format, expected_frames)?,
            frame_bytes: format.frame_bytes(width, height)?,
            frames_written: 0,
        })
    }

    fn push(&mut self, frame: RawVideoFrame) -> Result<()> {
        match frame {
            RawVideoFrame::Bgr(frame) => {
                let bytes = frame
                    .as_slice_memory_order()
                    .ok_or_else(|| anyhow!("color raw frame must be contiguous"))?;
                self.push_bytes(bytes)?;
            }
            RawVideoFrame::Gray(frame) => {
                let bytes = frame
                    .as_slice_memory_order()
                    .ok_or_else(|| anyhow!("gray raw frame must be contiguous"))?;
                self.push_bytes(bytes)?;
            }
        }
        Ok(())
    }

    fn push_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() != self.frame_bytes {
            bail!(
                "raw frame byte mismatch: expected {}, got {}",
                self.frame_bytes,
                bytes.len()
            );
        }
        match &mut self.sink {
            RawVideoSink::Mapped { map, .. } => {
                let offset = self
                    .frames_written
                    .checked_mul(self.frame_bytes)
                    .ok_or_else(|| anyhow!("raw stream offset overflow"))?;
                let end = offset
                    .checked_add(bytes.len())
                    .ok_or_else(|| anyhow!("raw stream end offset overflow"))?;
                if end > map.len() {
                    bail!("raw stream exceeded preallocated length");
                }
                map[offset..end].copy_from_slice(bytes);
            }
            RawVideoSink::Buffered { writer } => {
                writer
                    .write_all(bytes)
                    .context("appending raw frame bytes")?;
            }
        }
        self.frames_written += 1;
        Ok(())
    }

    fn finish(self) -> Result<usize> {
        let frames_written = self.frames_written;
        let actual_len = self
            .frame_bytes
            .checked_mul(frames_written)
            .ok_or_else(|| anyhow!("final raw stream length overflow"))?;
        match self.sink {
            RawVideoSink::Mapped { file, map } => {
                drop(map);
                file.set_len(actual_len as u64)
                    .context("truncating raw stream to written length")?;
            }
            RawVideoSink::Buffered { mut writer } => {
                writer.flush().context("flushing raw stream file")?;
                drop(
                    writer
                        .into_inner()
                        .map_err(|err| anyhow!("finalizing raw stream file: {err}"))?,
                );
            }
        }
        Ok(frames_written)
    }
}

pub fn finalize_raw_stream_to_mp4(
    ffmpeg_bin: &OsStr,
    artifact: &RawVideoArtifact,
    output_path: impl AsRef<Path>,
) -> Result<()> {
    if artifact.frames_written == 0 {
        return Ok(());
    }
    let output_path = output_path.as_ref();
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let mut command = Command::new(ffmpeg_bin);
    command
        .arg("-y")
        .arg("-f")
        .arg("rawvideo")
        .arg("-pix_fmt")
        .arg(artifact.format.ffmpeg_input_pix_fmt())
        .arg("-s")
        .arg(format!("{}x{}", artifact.width, artifact.height))
        .arg("-r")
        .arg(format!("{:.6}", artifact.fps))
        .arg("-i")
        .arg(&artifact.path)
        .arg("-c:v")
        .arg("libx264")
        .arg("-crf")
        .arg("18");
    if matches!(artifact.format, RawPixelFormat::Bgr24) {
        command.arg("-pix_fmt").arg("yuv420p");
    } else {
        command.arg("-vf").arg("format=gray");
    }
    let status = command
        .arg(output_path)
        .status()
        .with_context(|| format!("launching ffmpeg for {}", output_path.display()))?;
    if !status.success() {
        bail!("ffmpeg failed while encoding {}", output_path.display());
    }
    Ok(())
}

impl RawGrayFrameReader {
    pub fn open(artifact: &RawVideoArtifact) -> Result<Self> {
        if !matches!(artifact.format, RawPixelFormat::Gray8) {
            bail!("raw gray-frame reader expects a Gray8 artifact");
        }
        let file = File::open(&artifact.path)
            .with_context(|| format!("opening {}", artifact.path.display()))?;
        Ok(Self {
            reader: BufReader::new(file),
            width: artifact.width,
            height: artifact.height,
            frame_bytes: artifact.width.checked_mul(artifact.height).ok_or_else(|| {
                anyhow!(
                    "raw gray frame byte size overflow for {}",
                    artifact.path.display()
                )
            })?,
        })
    }

    pub fn read_next(&mut self) -> Result<Option<Array2<u8>>> {
        let mut bytes = vec![0u8; self.frame_bytes];
        match self.reader.read_exact(&mut bytes) {
            Ok(()) => Array2::from_shape_vec((self.height, self.width), bytes)
                .context("reshaping raw gray frame")
                .map(Some),
            Err(err) if err.kind() == ErrorKind::UnexpectedEof => Ok(None),
            Err(err) => Err(err).context("reading raw gray frame"),
        }
    }
}
