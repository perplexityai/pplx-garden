//! Optional end-to-end greedy golden for the only supported checkpoint.

use anyhow::{Context, Result};
use lily::generate::Generator;
use lily::metal::MetalContext;
use lily::model::Qwen3_5Model;
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct Golden {
    prompt_token_ids: Vec<u32>,
    steps: Vec<Step>,
}

#[derive(Deserialize)]
struct Step {
    chosen: u32,
}

#[test]
#[ignore = "requires LILY_MODEL_DIR_35B"]
fn qwen36_35b_greedy_matches_golden() -> Result<()> {
    let dir = std::env::var("LILY_MODEL_DIR_35B")
        .context("set LILY_MODEL_DIR_35B to the Qwen3.6-35B-A3B checkpoint")?;
    let golden: Golden =
        serde_json::from_slice(include_bytes!("goldens/golden_35b_chat.json"))?;
    let expected: Vec<u32> = golden.steps.iter().map(|s| s.chosen).collect();

    let ctx = MetalContext::new()?;
    let model = Qwen3_5Model::load(&ctx, Path::new(&dir))?;
    let mut generator = Generator::from_model_dir(Path::new(&dir))?;
    generator.add_stop_tokens(&model.config.eos_token_id.as_vec());
    let capacity = golden.prompt_token_ids.len() + expected.len() + 1;
    let mut state = model.new_state(&ctx, capacity)?;
    let mut scratch = model.new_scratch_with_capacity(&ctx, capacity)?;
    let actual = generator.generate(
        &ctx,
        &model,
        &mut state,
        &mut scratch,
        &golden.prompt_token_ids,
        expected.len(),
    )?;
    assert_eq!(actual.tokens, expected);
    Ok(())
}
