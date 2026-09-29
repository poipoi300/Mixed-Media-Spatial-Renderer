//! Long-running ffmpeg decodes streamed off its stdout.
//!
//! Playback decodes a video's sound (and picture) as one ffmpeg process that
//! runs from a start time onward, restarted only when playback jumps. A
//! dedicated OS thread per stream — not the shared compute pool, which a
//! stream would occupy for its whole lifetime — reads fixed-size records off
//! the pipe into a bounded channel. The bound is the read-ahead: once it
//! fills, the thread blocks, ffmpeg blocks on the full pipe, and a paused
//! stream costs no CPU. Dropping the receiver ends the stream: the thread's
//! next send fails and it kills ffmpeg.

use std::{
    ffi::OsString,
    io::{ErrorKind, Read},
    process::{Child, ChildStderr, Command, Stdio},
    thread::{self, JoinHandle},
};

use anyhow::{Context, Result};
use crossbeam_channel::{bounded, Receiver, Sender};

/// Bytes of ffmpeg's stderr kept for the end report. Everything past it is
/// still read, so ffmpeg never blocks on a full stderr pipe.
const STDERR_KEPT_BYTES: usize = 16 * 1024;

/// Why a stream stopped delivering records. Always the last item it sends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEnd {
    /// ffmpeg reached the end of the input.
    Finished,
    /// The input has no stream of the kind the decode selects (a video
    /// without sound, for example).
    MissingStream,
    Failed(String),
}

#[derive(Debug)]
pub enum StreamItem<T> {
    Record(T),
    End(StreamEnd),
}

/// Starts `ffmpeg <arguments>` and a thread that splits its stdout into
/// `record_bytes`-sized records, converts each with `into_record`, and sends
/// them through a channel holding at most `read_ahead_records`. A trailing
/// partial record is dropped. Fails only when ffmpeg cannot be started.
pub fn spawn_ffmpeg_stream<T, F>(
    thread_name: String,
    arguments: Vec<OsString>,
    record_bytes: usize,
    read_ahead_records: usize,
    into_record: F,
) -> Result<Receiver<StreamItem<T>>>
where
    T: Send + 'static,
    F: Fn(Vec<u8>) -> T + Send + 'static,
{
    let mut child = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-nostdin"])
        .args(arguments)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to start ffmpeg from PATH")?;
    let (sender, receiver) = bounded(read_ahead_records.max(1));
    thread::Builder::new()
        .name(thread_name)
        .spawn(move || pump_records(&mut child, &sender, record_bytes, into_record))
        .context("failed to start an ffmpeg reader thread")?;
    Ok(receiver)
}

fn pump_records<T, F>(
    child: &mut Child,
    sender: &Sender<StreamItem<T>>,
    record_bytes: usize,
    into_record: F,
) where
    F: Fn(Vec<u8>) -> T,
{
    let (Some(mut stdout), Some(stderr)) = (child.stdout.take(), child.stderr.take()) else {
        // Both pipes were requested at spawn, so they are always present.
        stop_child(child);
        return;
    };
    let stderr_reader = thread::spawn(move || read_stderr(stderr));
    loop {
        let mut record = vec![0u8; record_bytes];
        match stdout.read_exact(&mut record) {
            Ok(()) => {}
            Err(error) if error.kind() == ErrorKind::UnexpectedEof => break,
            Err(error) => {
                stop_child(child);
                let _ = sender.send(StreamItem::End(StreamEnd::Failed(error.to_string())));
                return;
            }
        }
        if sender
            .send(StreamItem::Record(into_record(record)))
            .is_err()
        {
            // Nobody listens any more.
            stop_child(child);
            return;
        }
    }
    let end = stream_end(child, stderr_reader);
    let _ = sender.send(StreamItem::End(end));
}

/// Reads stderr to its end, keeping the start of it. Runs on its own thread:
/// Windows pipes buffer only a few KiB, so an ffmpeg logging while the
/// record loop waits on stdout would otherwise stall.
fn read_stderr(mut stderr: ChildStderr) -> String {
    let mut kept = Vec::new();
    let mut buffer = [0u8; 4096];
    loop {
        match stderr.read(&mut buffer) {
            Ok(0) | Err(_) => break,
            Ok(read) => {
                let room = STDERR_KEPT_BYTES.saturating_sub(kept.len());
                kept.extend_from_slice(&buffer[..read.min(room)]);
            }
        }
    }
    String::from_utf8_lossy(&kept).into_owned()
}

/// Classifies how ffmpeg exited once its stdout closed.
fn stream_end(child: &mut Child, stderr_reader: JoinHandle<String>) -> StreamEnd {
    let status = child.wait();
    let message = stderr_reader.join().unwrap_or_default();
    match status {
        Ok(status) if status.success() => StreamEnd::Finished,
        // ffmpeg reports this when the input has nothing to map into the
        // output, such as a sound decode of a silent video.
        _ if message.contains("does not contain any stream")
            || message.contains("matches no streams") =>
        {
            StreamEnd::MissingStream
        }
        Ok(status) => StreamEnd::Failed(format!("ffmpeg exited with {status}: {message}")),
        Err(error) => StreamEnd::Failed(error.to_string()),
    }
}

fn stop_child(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}
