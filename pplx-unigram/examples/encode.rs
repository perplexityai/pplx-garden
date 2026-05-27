//! Encode a string with the Unigram engine.
//!
//! Usage:
//!     cargo run --release --example encode -- <path/to/tokenizer.json> <text>
//!
//! XLM-RoBERTa tokenizer.json can be downloaded from:
//! https://huggingface.co/FacebookAI/xlm-roberta-base/blob/main/tokenizer.json

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use pplx_unigram::{EncodeState, Engine};

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let (Some(path), Some(text)) = (args.next(), args.next()) else {
        eprintln!("usage: encode <tokenizer.json> <text>");
        return ExitCode::from(2);
    };
    if text.is_empty() {
        eprintln!("error: text argument must not be empty");
        return ExitCode::from(2);
    }
    let path = PathBuf::from(path);

    let load_start = Instant::now();
    let engine = match Engine::from_hf_json_path(&path) {
        Ok(engine) => engine,
        Err(err) => {
            eprintln!("error loading {}: {err}", path.display());
            return ExitCode::from(1);
        }
    };
    println!(
        "loaded {} ({} tokens) in {:?}",
        path.display(),
        engine.vocab_size(),
        load_start.elapsed()
    );

    // Warm the scratch buffers so per-encode allocations don't show up.
    let mut state = EncodeState::new();
    if let Err(err) = engine.encode(&text, &mut state) {
        eprintln!("encode failed: {err}");
        return ExitCode::from(1);
    }

    // Time 1000 steady-state encodes.
    let encode_start = Instant::now();
    for _ in 0..1000 {
        if let Err(err) = engine.encode(&text, &mut state) {
            eprintln!("encode failed: {err}");
            return ExitCode::from(1);
        }
    }
    let per_encode = encode_start.elapsed() / 1000;

    let preview: String =
        if text.len() > 80 { format!("{}…", &text[..80]) } else { text.clone() };
    println!("input  : {:?}", preview);
    println!("tokens : {:?}", &state.tokens[..state.tokens.len().min(16)]);
    println!("count  : {}", state.tokens.len());
    println!("encode : {:?} (avg over 1000 iters)", per_encode);
    ExitCode::SUCCESS
}
