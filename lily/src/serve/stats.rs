//! Request statistics for the optional TUI. Pure data, no terminal types, so
//! the aggregation is unit-testable without a terminal.

use std::collections::VecDeque;
use std::time::Duration;

/// Requests shown in the recent-requests table.
pub(super) const RECENT_CAP: usize = 20;
/// Completions that feed the rolling tokens-per-second figure.
pub(super) const ROLLING_WINDOW: usize = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TokenCounts {
    pub prompt: u64,
    pub cached: u64,
    pub completion: u64,
}

#[derive(Clone, Debug)]
pub(super) struct RequestRecord {
    pub method: String,
    pub path: String,
    pub status: u16,
    pub elapsed: Duration,
    /// Present for successful chat completions only.
    pub tokens: Option<TokenCounts>,
    pub error: Option<String>,
}

/// What the serve loop reports to the TUI thread.
pub(super) enum Event {
    Started,
    Finished(RequestRecord),
}

#[derive(Default)]
pub(super) struct Stats {
    pub total: u64,
    pub ok: u64,
    pub client_errors: u64,
    pub server_errors: u64,
    pub in_flight: bool,
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    pub completion_tokens: u64,
    pub recent: VecDeque<RequestRecord>,
}

impl Stats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, event: Event) {
        match event {
            Event::Started => self.in_flight = true,
            Event::Finished(record) => {
                self.in_flight = false;
                self.total += 1;
                match record.status {
                    200..=299 => self.ok += 1,
                    400..=499 => self.client_errors += 1,
                    _ => self.server_errors += 1,
                }
                if let Some(tokens) = record.tokens {
                    self.prompt_tokens += tokens.prompt;
                    self.cached_tokens += tokens.cached;
                    self.completion_tokens += tokens.completion;
                }
                if self.recent.len() == RECENT_CAP {
                    self.recent.pop_front();
                }
                self.recent.push_back(record);
            }
        }
    }

    /// Cached prompt tokens over all prompt tokens, across completions.
    pub fn cache_hit_ratio(&self) -> Option<f64> {
        (self.prompt_tokens > 0)
            .then(|| self.cached_tokens as f64 / self.prompt_tokens as f64)
    }

    /// Completion tokens per second of request wall time over the last
    /// [`ROLLING_WINDOW`] completions. Includes prefill, so it is a request
    /// rate rather than a decode rate.
    pub fn rolling_tok_per_s(&self) -> Option<f64> {
        let (tokens, secs) = self
            .recent
            .iter()
            .rev()
            .filter_map(|r| r.tokens.map(|t| (t.completion, r.elapsed.as_secs_f64())))
            .take(ROLLING_WINDOW)
            .fold((0u64, 0f64), |(t, s), (ct, cs)| (t + ct, s + cs));
        (secs > 0.0).then(|| tokens as f64 / secs)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/serve/stats.rs"]
mod tests;
