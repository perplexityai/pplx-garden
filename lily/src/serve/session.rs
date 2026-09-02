use anyhow::{Result, ensure};

use crate::metal::MetalContext;
use crate::model::{DecodeState, Qwen3_5Model};

pub(super) struct Session {
    pub tokens: Vec<u32>,
    pub state: DecodeState,
    pub cache_key: Option<String>,
    last_used: u64,
}

pub(super) struct SessionStore {
    entries: Vec<Session>,
    capacity: usize,
    max_seq: usize,
    clock: u64,
}

impl SessionStore {
    pub fn new(max_seq: usize, capacity: usize) -> Self {
        Self { entries: Vec::new(), capacity, max_seq, clock: 0 }
    }

    /// Check out the longest strict token prefix. A matching explicit key wins
    /// over a longer keyless match, but never bypasses token equality.
    pub fn acquire(
        &mut self,
        ctx: &MetalContext,
        model: &Qwen3_5Model,
        prompt: &[u32],
        cache_key: Option<&str>,
    ) -> Result<(Session, usize)> {
        let best = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| strict_prefix(&entry.tokens, prompt))
            .max_by_key(|(_, entry)| {
                let key_match = cache_key
                    .is_some_and(|key| entry.cache_key.as_deref() == Some(key));
                (key_match, entry.tokens.len())
            })
            .map(|(index, _)| index);

        if let Some(index) = best {
            let session = self.entries.swap_remove(index);
            let resume_at = session.tokens.len();
            ensure!(session.state.pos == resume_at, "invalid cached decode state");
            return Ok((session, resume_at));
        }

        let state = if self.entries.len() >= self.capacity {
            let oldest = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(index, _)| index)
                .unwrap_or(0);
            let mut session = self.entries.swap_remove(oldest);
            session.state.reset();
            session.state
        } else {
            model.new_state(ctx, self.max_seq)?
        };
        Ok((Session { tokens: Vec::new(), state, cache_key: None, last_used: 0 }, 0))
    }

    pub fn release(&mut self, mut session: Session, cache_key: Option<&str>) {
        if self.capacity == 0 || session.state.pos != session.tokens.len() {
            return;
        }
        self.clock = self.clock.wrapping_add(1);
        session.last_used = self.clock;
        session.cache_key = cache_key.map(str::to_owned);
        self.entries.push(session);
    }
}

fn strict_prefix(prefix: &[u32], full: &[u32]) -> bool {
    prefix.len() < full.len() && full.starts_with(prefix)
}

#[cfg(test)]
#[path = "../../tests/unit/serve/session.rs"]
mod tests;
