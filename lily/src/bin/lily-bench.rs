use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, ensure};
use clap::Parser;
use lily::kernels::attention::MAX_SEQ;
use lily::metal::MetalContext;
use lily::model::Qwen3_5Model;

#[derive(Parser)]
#[command(name = "lily-bench", about = "In-process production generation benchmark")]
struct Cli {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    prompt_len: usize,
    #[arg(long, default_value_t = 64)]
    decode_steps: usize,
    #[arg(long, default_value_t = false)]
    gpu_timing: bool,
    #[arg(long)]
    json_out: PathBuf,
}

fn fnv1a(tokens: &[u32]) -> u64 {
    tokens.iter().fold(0xcbf29ce484222325u64, |hash, token| {
        (hash ^ u64::from(*token)).wrapping_mul(0x100000001b3)
    })
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    ensure!(cli.prompt_len > 0, "--prompt-len must be positive");
    ensure!(cli.decode_steps > 0, "--decode-steps must be positive");
    let max_seq = cli
        .prompt_len
        .checked_add(cli.decode_steps)
        .and_then(|tokens| tokens.checked_add(1))
        .ok_or_else(|| anyhow::anyhow!("benchmark token budget overflow"))?;
    ensure!(max_seq <= MAX_SEQ, "benchmark exceeds model window {MAX_SEQ}");

    let ctx = MetalContext::new()?;
    let model = Qwen3_5Model::load(&ctx, &cli.model)?;
    let vocab = u32::try_from(model.config.vocab_size)?;
    let token = |index: usize| (index as u32).wrapping_mul(2_654_435_761) % vocab;
    let prompt: Vec<u32> = (0..cli.prompt_len).map(token).collect();
    let mut scratch = model.new_scratch_with_capacity(&ctx, max_seq)?;

    // Warm the exact prompt shape and production cadence: one delivered decode
    // step plus its async lookahead. This grows scratch and compiles every
    // shape-specialized Metal pipeline before the measured state is created.
    {
        let mut warm_state = model.new_state(&ctx, cli.prompt_len + 2)?;
        model.prefill(&ctx, &mut warm_state, &mut scratch, &prompt)?;
        let first = model.submit_decode_step(&ctx, &mut warm_state, &scratch, 0, 1)?;
        let lookahead =
            model.submit_decode_step(&ctx, &mut warm_state, &scratch, 1, 0)?;
        first.wait()?;
        lookahead.wait()?;
    }

    let mut state = model.new_state(&ctx, max_seq)?;
    let prefill_start = Instant::now();
    model.prefill(&ctx, &mut state, &mut scratch, &prompt)?;

    // Depth-2 production decode: submit the next command buffer before
    // waiting for the previous one, with generated ids remaining on the GPU.
    // Submit the first decode before token delivery and cadence timing.
    let mut pending = model.submit_decode_step(&ctx, &mut state, &scratch, 0, 1)?;
    let first_token_id = scratch.next_token.view(0, &[1])?.to_u32()?[0];
    let prefill_secs = prefill_start.elapsed().as_secs_f64();
    let decode_start = Instant::now();
    let mut previous_delivery = decode_start;
    let mut decode_intervals = Vec::with_capacity(cli.decode_steps);
    let mut completed_passes =
        if cli.gpu_timing { Vec::with_capacity(cli.decode_steps) } else { Vec::new() };
    let mut token_ids = Vec::with_capacity(cli.decode_steps);
    for index in 1..cli.decode_steps {
        let next = model.submit_decode_step(
            &ctx,
            &mut state,
            &scratch,
            index % 2,
            (index + 1) % 2,
        )?;
        let completed = if cli.gpu_timing {
            Some(pending.wait_retain()?)
        } else {
            pending.wait()?;
            None
        };
        token_ids.push(scratch.next_token.view(index % 2, &[1])?.to_u32()?[0]);
        let delivered = Instant::now();
        decode_intervals
            .push(delivered.duration_since(previous_delivery).as_secs_f64());
        if let Some(completed) = completed {
            completed_passes.push(completed);
        }
        previous_delivery = delivered;
        pending = next;
    }
    // Submit the final lookahead inside the cadence timer, then drain it outside.
    let lookahead = model.submit_decode_step(
        &ctx,
        &mut state,
        &scratch,
        cli.decode_steps % 2,
        (cli.decode_steps + 1) % 2,
    )?;
    let completed = if cli.gpu_timing {
        Some(pending.wait_retain()?)
    } else {
        pending.wait()?;
        None
    };
    token_ids.push(scratch.next_token.view(cli.decode_steps % 2, &[1])?.to_u32()?[0]);
    let delivered = Instant::now();
    decode_intervals.push(delivered.duration_since(previous_delivery).as_secs_f64());
    if let Some(completed) = completed {
        completed_passes.push(completed);
    }
    let decode_secs = delivered.duration_since(decode_start).as_secs_f64();
    let lookahead_completed = if cli.gpu_timing {
        Some(lookahead.wait_retain()?)
    } else {
        lookahead.wait()?;
        None
    };

    // Diagnostic timestamp queries are intentionally outside the production
    // cadence timer. The default path retains no completed command buffers.
    let mut gpu_passes = Vec::with_capacity(completed_passes.len());
    for completed in completed_passes {
        let gpu = completed.timing()?;
        gpu_passes.push(serde_json::json!({
            "start_secs": gpu.gpu_start_secs,
            "end_secs": gpu.gpu_end_secs,
            "wall_secs": gpu.gpu_end_secs - gpu.gpu_start_secs,
        }));
    }
    let lookahead_gpu =
        lookahead_completed.map(|completed| completed.timing()).transpose()?;
    let token_digest = fnv1a(&token_ids);

    let report = serde_json::json!({
        "schema_version": 1,
        "meta": {
            "engine": "lily",
            "source_id": option_env!("LILY_BENCH_SOURCE_ID").unwrap_or("unknown"),
            "crate_version": env!("CARGO_PKG_VERSION"),
            "harness": "src/bin/lily-bench.rs",
        },
        "workload": {
            "prompt_len": cli.prompt_len,
            "prompt_kind": "u32_golden_ratio_hash_mod_vocab",
            "decode_steps": cli.decode_steps,
            "decode_mode": "production_depth2_concurrent",
            "gpu_timing_diagnostic": cli.gpu_timing,
        },
        "results": {
            "prefill": {
                "wall_secs": prefill_secs,
                "tok_s": cli.prompt_len as f64 / prefill_secs,
                "first_token_id": first_token_id,
            },
            "decode": {
                "wall_secs": decode_secs,
                "tok_s": cli.decode_steps as f64 / decode_secs,
                "step_wall_secs": decode_intervals,
                "gpu_passes": gpu_passes,
                "lookahead_gpu_pass": lookahead_gpu.map(|gpu| serde_json::json!({
                    "start_secs": gpu.gpu_start_secs,
                    "end_secs": gpu.gpu_end_secs,
                    "wall_secs": gpu.gpu_end_secs - gpu.gpu_start_secs,
                })),
                "token_digest": format!("{token_digest:016x}"),
                "token_ids": token_ids,
            },
        },
    });
    std::fs::write(&cli.json_out, serde_json::to_vec_pretty(&report)?)?;
    eprintln!(
        "prefill: {} tok in {:.6}s ({:.1} tok/s) | decode: {} steps in {:.6}s ({:.1} tok/s) | digest={token_digest:016x}",
        cli.prompt_len,
        prefill_secs,
        cli.prompt_len as f64 / prefill_secs,
        cli.decode_steps,
        decode_secs,
        cli.decode_steps as f64 / decode_secs,
    );
    Ok(())
}
