//! Qwen3.6-35B-A3B chat-template checks against the checkpoint tokenizer.

use anyhow::{Context, Result, ensure};
use lily::chat::{Conversation, Message};
use lily::generate::{Generator, Thinking};
use serde::Deserialize;

#[derive(Deserialize)]
struct Golden {
    prompt: String,
    prompt_token_ids: Vec<u32>,
}

fn generator() -> Result<Generator> {
    let dir = std::env::var("LILY_MODEL_DIR_35B")
        .context("set LILY_MODEL_DIR_35B to the Qwen3.6-35B-A3B checkpoint")?;
    Generator::from_model_dir(dir.as_ref())
}

#[test]
#[ignore = "requires LILY_MODEL_DIR_35B"]
fn chat_prompt_matches_35b_golden() -> Result<()> {
    let generator = generator()?;
    let golden: Golden =
        serde_json::from_slice(include_bytes!("goldens/golden_35b_chat.json"))?;
    let conversation: Conversation = vec![Message::new_user(&golden.prompt)];
    let actual = generator.encode_chat(&conversation, Thinking::Enabled)?;
    ensure!(actual == golden.prompt_token_ids, "35B chat prompt tokens changed");
    Ok(())
}

#[test]
#[ignore = "requires LILY_MODEL_DIR_35B"]
fn chat_history_is_deterministic_and_cacheable() -> Result<()> {
    let generator = generator()?;
    let first: Conversation = vec![Message::new_user("Name a prime number.")];
    let followed: Conversation = vec![
        Message::new_user("Name a prime number."),
        Message::new_assistant("2"),
        Message::new_user("And the next one?"),
    ];
    let turn1 = generator.encode_chat(&first, Thinking::Disabled)?;
    let turn2 = generator.encode_chat(&followed, Thinking::Disabled)?;
    ensure!(
        turn2.starts_with(&turn1),
        "the follow-up prompt must extend the cached first-turn prompt"
    );
    ensure!(
        generator.encode_chat(&followed, Thinking::Disabled)? == turn2,
        "chat encoding must be deterministic"
    );
    Ok(())
}
