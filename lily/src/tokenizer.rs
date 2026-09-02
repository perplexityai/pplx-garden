//! Tokenization and chat-template rendering, read from the checkpoint.
//!
//! Both halves come out of the model directory rather than being compiled in:
//! `tokenizer.json` drives encode/decode through the `tokenizers` crate, and
//! `chat_template.jinja` is rendered with minijinja. That keeps lily's prompt
//! byte-identical to what any other server applying the same checkpoint's
//! template produces, which is the only way a golden recorded elsewhere can
//! be replayed here.
//!
//! `tests/test_tokenizer.rs` pins rendering against the supported 35B
//! checkpoint and verifies that direct-answer histories remain prefix-cacheable.

use std::path::Path;

use anyhow::{Context as _, Result, bail};
use minijinja::value::Value;
use minijinja::{Environment, Error as JinjaError, ErrorKind as JinjaErrorKind};

use crate::chat::{Conversation, Role};

/// Whether the generation prompt opens a reasoning block for the model to
/// continue inside, or closes an empty one so the model answers directly.
///
/// This is what OpenAI-style `chat_template_kwargs: {"enable_thinking": …}`
/// selects. It is a prompt-shape decision, not a sampling one: nothing
/// downstream of the tokenizer needs to know.
///
/// The generation prompt is byte-identical to the checkpoint's own template
/// under both settings — `<think>\n` on, `<think>\n\n</think>\n\n` off.
/// History rendering deliberately differs under [`Thinking::Disabled`]; see
/// [`Tokenizer::render_chat`].
///
/// The API server chooses [`Thinking::Disabled`] so it returns direct answers
/// and can reuse exact multi-turn token prefixes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Thinking {
    #[default]
    Enabled,
    Disabled,
}

/// The closed empty reasoning block. Under [`Thinking::Disabled`] the
/// checkpoint's template emits exactly this after the assistant header, and
/// lily additionally writes it into older assistant turns — see
/// [`Tokenizer::render_chat`].
const EMPTY_REASONING_BLOCK: &str = "<think>\n\n</think>\n\n";

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    env: Environment<'static>,
    /// Ids that end a turn: the tokenizer's `eos_token`, plus whatever the
    /// caller adds from the checkpoint config.
    stop_tokens: Vec<u32>,
}

impl Tokenizer {
    /// Loads `tokenizer.json` and the chat template from a checkpoint
    /// directory.
    ///
    /// The template is looked for in `chat_template.jinja` first and in
    /// `tokenizer_config.json`'s `chat_template` second — newer HF exports
    /// use the standalone file, older ones inline it.
    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        let tokenizer_path = dir.join("tokenizer.json");
        let inner = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| {
                format!("loading tokenizer from {}", tokenizer_path.display())
            })?;

        let config: serde_json::Value =
            match std::fs::read(dir.join("tokenizer_config.json")) {
                Ok(bytes) => serde_json::from_slice(&bytes)
                    .context("parsing tokenizer_config.json")?,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    serde_json::Value::Null
                }
                Err(e) => return Err(e).context("reading tokenizer_config.json"),
            };

        let template = Self::load_template(dir, &config)?;
        let mut env = Environment::new();
        // HF templates are written against Python's `str`; minijinja only
        // ships the Jinja builtins, so string methods arrive through this
        // shim. Without it the Qwen template fails on `.startswith(...)`.
        env.set_unknown_method_callback(
            minijinja_contrib::pycompat::unknown_method_callback,
        );
        // Templates signal unusable input (no user turn, a system message out
        // of position) by calling this; surface it as a render error so the
        // request is refused rather than served a malformed prompt.
        env.add_function(
            "raise_exception",
            |message: String| -> Result<Value, JinjaError> {
                Err(JinjaError::new(JinjaErrorKind::InvalidOperation, message))
            },
        );
        env.add_template_owned("chat", template)
            .context("compiling the checkpoint's chat template")?;

        let mut stop_tokens = Vec::new();
        if let Some(eos) = config.get("eos_token").and_then(token_text)
            && let Some(id) = inner.token_to_id(&eos)
        {
            stop_tokens.push(id);
        }

        Ok(Self { inner, env, stop_tokens })
    }

    fn load_template(dir: &Path, config: &serde_json::Value) -> Result<String> {
        let path = dir.join("chat_template.jinja");
        match std::fs::read_to_string(&path) {
            Ok(text) => return Ok(text),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(e).with_context(|| format!("reading {}", path.display()));
            }
        }
        match config.get("chat_template").and_then(|v| v.as_str()) {
            Some(text) => Ok(text.to_string()),
            None => bail!(
                "no chat template in {}: expected chat_template.jinja or a \
                 `chat_template` field in tokenizer_config.json",
                dir.display()
            ),
        }
    }

    /// Turn-ending ids known to the tokenizer. The checkpoint config may name
    /// more; [`Self::add_stop_tokens`] folds those in.
    pub fn stop_tokens(&self) -> &[u32] {
        &self.stop_tokens
    }

    pub fn add_stop_tokens(&mut self, ids: &[u32]) {
        for &id in ids {
            if !self.stop_tokens.contains(&id) {
                self.stop_tokens.push(id);
            }
        }
    }

    /// Raw encode. Special tokens are never added implicitly: the prompt must
    /// end exactly where the model should continue, and the chat template
    /// already writes every marker it wants.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("encoding text")?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("decoding tokens")
    }

    /// Renders a conversation through the checkpoint's chat template, ending
    /// at the assistant header so the model continues the next turn.
    ///
    /// Under [`Thinking::Disabled`] this departs from the template in one
    /// place, deliberately. The template wraps an assistant turn in
    /// `<think>…</think>` only when the turn comes *after* the last user
    /// message; older turns render as bare content. That makes a nothink
    /// reply re-render differently once another user turn arrives — and
    /// because the session cache needs the cached sequence to be an exact
    /// prefix of the new prompt, one diverging byte costs the whole prefill.
    /// So lily writes the closed empty block into every assistant turn. The
    /// rendering is otherwise the template's, byte for byte.
    pub fn render_chat(
        &self,
        messages: &Conversation,
        thinking: Thinking,
    ) -> Result<String> {
        if messages.is_empty() {
            bail!("empty conversation");
        }
        let messages = self.template_messages(messages, thinking)?;

        let template = self.env.get_template("chat")?;
        let rendered = template.render(minijinja::context! {
            messages => Value::from_serialize(&messages),
            tools => Value::from_serialize(Vec::<serde_json::Value>::new()),
            add_generation_prompt => true,
            enable_thinking => thinking == Thinking::Enabled,
        })?;
        Ok(rendered)
    }

    /// The message list as the template should see it: content flattened to
    /// text, and — under nothink — every assistant turn carrying the closed
    /// empty block that the template would otherwise drop for older turns.
    fn template_messages(
        &self,
        messages: &Conversation,
        thinking: Thinking,
    ) -> Result<Vec<serde_json::Value>> {
        let last_query = last_query_index(messages);
        let mut out = Vec::with_capacity(messages.len());
        for (index, message) in messages.iter().enumerate() {
            let text = message.content.clone();
            let mut value = serde_json::json!({
                "role": message.role,
                "content": text,
            });
            // The template's own wrapper covers assistant turns after the
            // last user message; only the earlier ones need the block written
            // in by hand. Setting `reasoning_content` to a string also stops
            // the template splitting `</think>` back out of the content.
            let needs_block = thinking == Thinking::Disabled
                && message.role == Role::Assistant
                && last_query.is_some_and(|last| index <= last);
            if needs_block {
                let body = if text.starts_with(EMPTY_REASONING_BLOCK) {
                    text
                } else {
                    format!("{EMPTY_REASONING_BLOCK}{text}")
                };
                value["content"] = serde_json::Value::String(body);
                value["reasoning_content"] = serde_json::Value::String(String::new());
            }
            out.push(value);
        }
        Ok(out)
    }
}

/// `eos_token` is either a bare string or an `AddedToken` object; both
/// shapes appear in checkpoints from the same family.
fn token_text(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(map) => {
            map.get("content")?.as_str().map(str::to_string)
        }
        _ => None,
    }
}

/// The last user turn, matching the only query role the minimal API accepts.
fn last_query_index(messages: &Conversation) -> Option<usize> {
    messages.iter().rposition(|message| message.role == Role::User)
}

#[cfg(test)]
#[path = "../tests/unit/tokenizer.rs"]
mod tests;
