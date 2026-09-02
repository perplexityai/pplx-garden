//! Checkpoint tokenization and the single greedy decode loop.

use std::path::Path;

use anyhow::{Result, ensure};

use crate::chat::Conversation;
use crate::metal::MetalContext;
use crate::model::{DecodeState, Qwen3_5Model, Scratch};
pub use crate::tokenizer::Thinking;
use crate::tokenizer::Tokenizer;

pub struct Generation {
    pub tokens: Vec<u32>,
    pub text: String,
    /// Whether the last returned token is one of the configured stop tokens.
    pub stopped: bool,
    /// Number of prompt-suffix and generated tokens actually fed into state.
    pub fed: usize,
}

fn ends_with_stop_token(tokens: &[u32], stop_tokens: &[u32]) -> bool {
    tokens.last().is_some_and(|token| stop_tokens.contains(token))
}

pub struct Generator {
    tokenizer: Tokenizer,
    stop_tokens: Vec<u32>,
}

impl Generator {
    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_model_dir(dir)?;
        let stop_tokens = tokenizer.stop_tokens().to_vec();
        Ok(Self { tokenizer, stop_tokens })
    }

    pub fn add_stop_tokens(&mut self, ids: &[u32]) {
        for &id in ids {
            if !self.stop_tokens.contains(&id) {
                self.stop_tokens.push(id);
            }
        }
    }

    pub fn encode_chat(
        &self,
        messages: &Conversation,
        thinking: Thinking,
    ) -> Result<Vec<u32>> {
        ensure!(!messages.is_empty(), "empty conversation");
        self.tokenizer.encode(&self.tokenizer.render_chat(messages, thinking)?)
    }

    pub fn decode_text(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer.decode(tokens, true)
    }

    pub fn generate(
        &self,
        ctx: &MetalContext,
        model: &Qwen3_5Model,
        state: &mut DecodeState,
        scratch: &mut Scratch,
        prompt_ids: &[u32],
        max_tokens: usize,
    ) -> Result<Generation> {
        ensure!(!prompt_ids.is_empty(), "empty prompt");
        let pos_before = state.pos;
        model.prefill(ctx, state, scratch, prompt_ids)?;
        let tokens = self.decode_pipelined(ctx, model, state, scratch, max_tokens)?;
        let stopped = ends_with_stop_token(&tokens, &self.stop_tokens);
        let text = self.decode_text(&tokens)?;
        let fed = state
            .pos
            .checked_sub(pos_before)
            .ok_or_else(|| anyhow::anyhow!("decode state moved backwards"))?;
        Ok(Generation { tokens, text, stopped, fed })
    }

    fn decode_pipelined(
        &self,
        ctx: &MetalContext,
        model: &Qwen3_5Model,
        state: &mut DecodeState,
        scratch: &Scratch,
        max_tokens: usize,
    ) -> Result<Vec<u32>> {
        let read_slot = |slot: usize| -> Result<u32> {
            Ok(scratch.next_token.view(slot, &[1])?.to_u32()?[0])
        };
        let mut tokens = Vec::new();
        if max_tokens == 0 {
            return Ok(tokens);
        }

        let mut slot = 0usize;
        let first = read_slot(slot)?;
        tokens.push(first);
        if self.stop_tokens.contains(&first) || max_tokens == 1 {
            return Ok(tokens);
        }

        let mut pending =
            model.submit_decode_step(ctx, state, scratch, slot, 1 - slot)?;
        for index in 1..max_tokens {
            let next = if index + 1 < max_tokens {
                Some(model.submit_decode_step(ctx, state, scratch, 1 - slot, slot)?)
            } else {
                None
            };
            pending.wait()?;
            slot = 1 - slot;
            let chosen = read_slot(slot)?;
            tokens.push(chosen);
            if self.stop_tokens.contains(&chosen) {
                if let Some(overshoot) = next {
                    overshoot.wait()?;
                }
                break;
            }
            match next {
                Some(pass) => pending = pass,
                None => break,
            }
        }
        Ok(tokens)
    }
}

#[cfg(test)]
#[path = "../tests/unit/generate.rs"]
mod tests;
