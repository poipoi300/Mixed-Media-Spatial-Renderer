//! Shared plumbing for "decode off the main thread, collect on it".
//!
//! Every media pipeline in the viewer (poster/billboard images, video chunks,
//! audio tracks) has the same shape: a system hands work to a background
//! worker, the worker pushes its result into a queue, and a later system
//! drains that queue within a frame budget. Both halves of that pattern live
//! here so the pipelines share one implementation instead of three.
//!
//! Work runs on Bevy's [`AsyncComputeTaskPool`], not a freshly spawned OS
//! thread. Decodes are frequent and short (one per billboard that enters the
//! cache), so paying a thread create/destroy per decode was pure overhead;
//! the pool keeps a fixed set of threads warm and hands the closure to an
//! idle one. Concurrency itself is still bounded upstream by
//! [`crate::decode_budget::DecodeBudget`] — the pool is the execution
//! mechanism, not the limiter.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use bevy::tasks::AsyncComputeTaskPool;

/// Runs `work` on the async compute pool, detached: the caller never joins it
/// and the result is expected to arrive through a [`CompletedWork`] queue the
/// closure captures.
pub fn spawn_background<F>(work: F)
where
    F: FnOnce() + Send + 'static,
{
    AsyncComputeTaskPool::get()
        .spawn(async move { work() })
        .detach();
}

/// One-way "stop, nobody wants this any more" flag shared between the
/// scheduler and a detached worker.
///
/// Work is dispatched onto a pool and never joined, so the only way to stop
/// an obsolete decode is to have the worker notice. Cancellation is therefore
/// cooperative: the worker polls the token at the points where the remaining
/// work is expensive (before reading the file, before resizing, between
/// tiles). A cancelled decode still delivers a result — an
/// [`Err`] the receiving system discards — so in-flight bookkeeping needs no
/// separate "it vanished" path.
#[derive(Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    /// Asks the worker to abandon its remaining work. Idempotent, and safe to
    /// call after the work already finished.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// Results produced by background workers and consumed by one main-thread
/// system. Cloning shares the same queue, so a worker only needs its own
/// handle.
///
/// Ordering is FIFO, with [`CompletedWork::push_front`] available for the
/// receiving system to return an item it accepted but could not afford to
/// process this frame.
pub struct CompletedWork<T>(Arc<Mutex<VecDeque<T>>>);

impl<T> CompletedWork<T> {
    fn locked(&self) -> std::sync::MutexGuard<'_, VecDeque<T>> {
        // A poisoned queue means a worker panicked mid-push and the contents
        // may be torn; there is no meaningful recovery inside a frame.
        self.0
            .lock()
            .expect("background work queue poisoned by a panicking worker")
    }

    /// Appends a finished result. Called from a background worker.
    pub fn push(&self, result: T) {
        self.locked().push_back(result);
    }

    /// Takes the oldest result, if any.
    pub fn pop_front(&self) -> Option<T> {
        self.locked().pop_front()
    }

    /// Returns a result to the head of the queue so it is retried first next
    /// frame, preserving arrival order.
    pub fn push_front(&self, result: T) {
        self.locked().push_front(result);
    }

    /// Takes every queued result at once.
    pub fn drain(&self) -> Vec<T> {
        self.locked().drain(..).collect()
    }

    pub fn len(&self) -> usize {
        self.locked().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// Derived impls would demand `T: Clone` / `T: Default`; the handle is shared
// and the queue starts empty regardless of what it carries.
impl<T> Clone for CompletedWork<T> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<T> Default for CompletedWork<T> {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(VecDeque::new())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clones_share_one_queue() {
        let queue: CompletedWork<u32> = CompletedWork::default();
        let worker = queue.clone();
        worker.push(1);
        worker.push(2);

        assert_eq!(queue.len(), 2);
        assert_eq!(queue.pop_front(), Some(1));
        assert_eq!(queue.pop_front(), Some(2));
        assert!(queue.is_empty());
    }

    #[test]
    fn cancel_is_visible_through_every_clone() {
        let token = CancelToken::new();
        let worker = token.clone();
        assert!(!worker.is_cancelled());

        token.cancel();

        assert!(worker.is_cancelled());
        // Idempotent: a second cancel (e.g. eviction after a supersede) is
        // not an error.
        token.cancel();
        assert!(worker.is_cancelled());
    }

    #[test]
    fn push_front_retries_before_newer_results() {
        let queue: CompletedWork<u32> = CompletedWork::default();
        queue.push(1);
        queue.push(2);

        let deferred = queue.pop_front().expect("queued result");
        queue.push_front(deferred);

        assert_eq!(queue.drain(), vec![1, 2]);
        assert!(queue.is_empty());
    }
}
