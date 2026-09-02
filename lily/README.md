# lily

A small Metal inference server for one checkpoint: Qwen3.6-35B-A3B converted
to MLX affine 4-bit weights. Lily exposes a minimal subset of the OpenAI chat
completions API and always decodes greedily.

Performance reports include the measurement contract and reproduction steps:

- [2026-09-01: MLX 0.31.2](docs/2026-09-01-performance-mlx-0.31.2.md)
- [2026-09-02: MLX 0.32.2](docs/2026-09-02-performance-mlx-0.32.2.md)

The Metal kernels compile from source at runtime; there is no offline shader
build step.

## Requirements

- Apple GPU family 10 or later (M5 and newer)
- macOS 26 or later for Metal tensor operations
- Rust 1.92, pinned by `rust-toolchain.toml`
- A local Qwen3.6-35B-A3B MLX affine 4-bit checkpoint with group size 64

Lily validates the exact 35B-A3B architecture and quantization layout at load
time. Dense Qwen checkpoints, smaller Qwen checkpoints, BF16 checkpoints,
GGUF, AWQ, GPTQ, int8 and fp8 are not supported.

The release benchmark and tests use the immutable
`mlx-community/Qwen3.6-35B-A3B-4bit` revision
`38740b847e4cb78f352aba30aa41c76e08e6eb46`. Download that exact checkpoint
with the Hugging Face CLI:

```sh
hf download mlx-community/Qwen3.6-35B-A3B-4bit \
  --revision 38740b847e4cb78f352aba30aa41c76e08e6eb46 \
  --local-dir /path/to/Qwen3.6-35B-A3B-4bit
```

## Run

```sh
cargo build --release --locked

./target/release/lily \
  --model /path/to/Qwen3.6-35B-A3B-4bit \
  --bind 127.0.0.1:8000 \
  --max-seq 4096
```

The server provides:

- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /health`

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.6-35B-A3B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64,
    "prompt_cache_key": "conversation-1"
  }'
```

The request surface is intentionally strict. It accepts text-only
system/user/assistant messages, `max_tokens`, `stream: false`, and the optional
`prompt_cache_key`. The final message must have role `user`, and the server
always renders the checkpoint template with thinking disabled. Sampling
parameters, streaming responses, tools, response formats, multimodal content,
and speculative decoding are rejected.

`--max-seq` is the combined prompt-plus-completion capacity. It cannot exceed
the kernel limit of 262,144 tokens and is clamped to the checkpoint's declared
`max_position_embeddings` when that value is smaller.

## Session prefix cache

The server keeps a fixed two-entry LRU cache of decode states. A state is reused
only when its token sequence is a strict prefix of the new prompt. A matching
`prompt_cache_key` prefers the corresponding entry but never bypasses token
equality. The response reports reused tokens in
`usage.prompt_tokens_details.cached_tokens`.

## Tests

```sh
cargo test --locked
```

This runs the CPU-reference kernel tests, shader compilation test, API surface
tests, and prefix-cache tests. Tests that need the 35B checkpoint are explicitly
ignored by default:

```sh
LILY_MODEL_DIR_35B=/path/to/Qwen3.6-35B-A3B-4bit \
  cargo test --test test_tokenizer -- --ignored --test-threads=1

LILY_MODEL_DIR_35B=/path/to/Qwen3.6-35B-A3B-4bit \
  cargo test --test test_e2e_35b -- --ignored --test-threads=1
```

## Source layout

```text
src/config.rs       strict 35B-A3B checkpoint validation
src/weights.rs      MLX affine Q4 weight loading
src/model.rs        prefill and single-token greedy model graph
src/generate.rs     tokenizer-backed greedy decode loop
src/serve.rs        minimal OpenAI-compatible HTTP server
src/serve/session.rs strict token-prefix session cache
src/kernels/        Rust dispatch and Metal shader sources
tests/              kernel, API, tokenizer, shader, and 35B golden tests
benchmarks/         Lily/MLX harnesses and the fail-closed matrix runner
```

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
