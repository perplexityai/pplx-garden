//! Encode strings with the Unigram engine.
//!
//! Single-shot:
//!     cargo run --release --example encode -- <tokenizer.json> <text>
//!
//! Batch from stdin (one input per non-empty line):
//!     echo -e "hello\nworld" | cargo run --release --example encode -- <tokenizer.json> --stdin
//!
//! Batch from a file (one input per non-empty line):
//!     cargo run --release --example encode -- <tokenizer.json> --file inputs.txt
//!
//! JSON-lines output (works with any of the above):
//!     cargo run --release --example encode -- <tokenizer.json> --stdin --json
//!     ... emits one `{"input": "...", "tokens": [...], "count": N}` per line.
//!
//! XLM-RoBERTa tokenizer.json can be downloaded from:
//! https://huggingface.co/FacebookAI/xlm-roberta-base/blob/main/tokenizer.json

use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use pplx_unigram::{EncodeState, Engine};

enum Mode {
    Single(String),
    Stdin,
    File(PathBuf),
}

struct Args {
    tokenizer: PathBuf,
    mode: Mode,
    json: bool,
}

const USAGE: &str = "\
usage:
    encode <tokenizer.json> <text>             single-shot (text from argv)
    encode <tokenizer.json> --stdin            batch from stdin (one input per line)
    encode <tokenizer.json> --file <path>      batch from file (one input per line)

flags:
    --json                                     emit one JSON object per input
                                               ({\"input\", \"tokens\", \"count\"})
    -h, --help                                 print this message

empty lines are skipped in batch modes.
";

fn parse_args() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let tokenizer_arg = args.next().ok_or_else(|| "missing <tokenizer.json>".to_string())?;
    if matches!(tokenizer_arg.as_str(), "-h" | "--help") {
        // Caller prints usage on Err; surface as a no-op error.
        return Err(String::new());
    }
    let tokenizer = PathBuf::from(tokenizer_arg);

    let mut text: Option<String> = None;
    let mut stdin = false;
    let mut file: Option<PathBuf> = None;
    let mut json = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--stdin" => stdin = true,
            "--json" => json = true,
            "--file" => {
                let path = args
                    .next()
                    .ok_or_else(|| "--file requires a PATH argument".to_string())?;
                file = Some(PathBuf::from(path));
            }
            "-h" | "--help" => return Err(String::new()),
            other if text.is_none() && !other.starts_with("--") => {
                text = Some(other.to_string());
            }
            other => return Err(format!("unexpected argument: {other}")),
        }
    }

    let mode = match (text, stdin, file) {
        (Some(t), false, None) => Mode::Single(t),
        (None, true, None) => Mode::Stdin,
        (None, false, Some(p)) => Mode::File(p),
        (None, false, None) => {
            return Err("specify one of <text>, --stdin, or --file <path>".to_string());
        }
        _ => {
            return Err(
                "<text>, --stdin, and --file are mutually exclusive".to_string(),
            );
        }
    };

    Ok(Args { tokenizer, mode, json })
}

fn escape_json_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn emit_json_line<W: Write>(
    out: &mut W,
    input: &str,
    tokens: &[u32],
) -> io::Result<()> {
    let mut buf = String::with_capacity(64 + tokens.len() * 8);
    buf.push_str("{\"input\":");
    escape_json_string(input, &mut buf);
    buf.push_str(",\"tokens\":[");
    for (i, t) in tokens.iter().enumerate() {
        if i > 0 {
            buf.push(',');
        }
        buf.push_str(&t.to_string());
    }
    buf.push_str("],\"count\":");
    buf.push_str(&tokens.len().to_string());
    buf.push('}');
    writeln!(out, "{buf}")
}

fn encode_one<W: Write>(
    engine: &Engine,
    state: &mut EncodeState,
    out: &mut W,
    text: &str,
    json: bool,
) -> Result<(), String> {
    engine.encode(text, state).map_err(|e| format!("encode failed: {e}"))?;
    if json {
        emit_json_line(out, text, &state.tokens)
            .map_err(|e| format!("write failed: {e}"))?;
    } else {
        writeln!(out, "{:?}", state.tokens).map_err(|e| format!("write failed: {e}"))?;
    }
    Ok(())
}

fn run_batch<R: BufRead, W: Write>(
    engine: &Engine,
    reader: R,
    out: &mut W,
    json: bool,
) -> Result<usize, String> {
    let mut state = EncodeState::new();
    let mut count = 0;
    for line in reader.lines() {
        let line = line.map_err(|e| format!("read failed: {e}"))?;
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            continue;
        }
        encode_one(engine, &mut state, out, line, json)?;
        count += 1;
    }
    Ok(count)
}

fn run_single<W: Write>(
    engine: &Engine,
    out: &mut W,
    text: &str,
    json: bool,
) -> Result<(), String> {
    if text.is_empty() {
        return Err("text argument must not be empty".to_string());
    }

    // Warm the scratch buffers so per-encode allocations don't show up in timings.
    let mut state = EncodeState::new();
    engine.encode(text, &mut state).map_err(|e| format!("encode failed: {e}"))?;

    if json {
        emit_json_line(out, text, &state.tokens)
            .map_err(|e| format!("write failed: {e}"))?;
        return Ok(());
    }

    // Time 1000 steady-state encodes (text mode only, JSON mode is for piping).
    let encode_start = Instant::now();
    for _ in 0..1000 {
        engine.encode(text, &mut state).map_err(|e| format!("encode failed: {e}"))?;
    }
    let per_encode = encode_start.elapsed() / 1000;

    let preview: String =
        if text.len() > 80 { format!("{}…", &text[..80]) } else { text.to_string() };
    writeln!(out, "input  : {:?}", preview).map_err(|e| format!("write failed: {e}"))?;
    writeln!(
        out,
        "tokens : {:?}",
        &state.tokens[..state.tokens.len().min(16)]
    )
    .map_err(|e| format!("write failed: {e}"))?;
    writeln!(out, "count  : {}", state.tokens.len())
        .map_err(|e| format!("write failed: {e}"))?;
    writeln!(out, "encode : {:?} (avg over 1000 iters)", per_encode)
        .map_err(|e| format!("write failed: {e}"))?;
    Ok(())
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(msg) => {
            if !msg.is_empty() {
                eprintln!("error: {msg}");
            }
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };

    let load_start = Instant::now();
    let engine = match Engine::from_hf_json_path(&args.tokenizer) {
        Ok(engine) => engine,
        Err(err) => {
            eprintln!("error loading {}: {err}", args.tokenizer.display());
            return ExitCode::from(1);
        }
    };
    if !args.json {
        eprintln!(
            "loaded {} ({} tokens) in {:?}",
            args.tokenizer.display(),
            engine.vocab_size(),
            load_start.elapsed()
        );
    }

    let stdout = io::stdout();
    let mut out = stdout.lock();

    let result = match &args.mode {
        Mode::Single(text) => run_single(&engine, &mut out, text, args.json),
        Mode::Stdin => {
            let stdin = io::stdin();
            run_batch(&engine, stdin.lock(), &mut out, args.json).map(|_| ())
        }
        Mode::File(path) => match File::open(path) {
            Ok(file) => run_batch(&engine, BufReader::new(file), &mut out, args.json)
                .map(|_| ()),
            Err(err) => Err(format!("error opening {}: {err}", path.display())),
        },
    };

    if let Err(msg) = result {
        eprintln!("{msg}");
        return ExitCode::from(1);
    }

    ExitCode::SUCCESS
}
