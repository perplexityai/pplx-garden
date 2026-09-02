# 2026-09-01: Lily vs MLX 0.31.2

This report compares Lily with MLX on Qwen3.6-35B-A3B. It measures in-process
production generation throughput, not identical isolated model graphs and not
HTTP end-to-end latency. Model loading, tokenization, chat-template rendering,
HTTP, and detokenization are outside the timed regions.

## Results

Unit: tokens/second. Each value is the median of two recorded rounds. Ratios
are Lily divided by MLX.

| Context | Lily decode | MLX decode | Lily / MLX | Lily prefill | MLX prefill | Lily / MLX |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 194.2 | 146.9 | 1.322x | 2875.7 | 1923.7 | 1.495x |
| 512 | 193.7 | 146.5 | 1.322x | 3872.0 | 2693.7 | 1.437x |
| 1,024 | 193.8 | 144.9 | 1.338x | 4738.9 | 3903.1 | 1.214x |
| 2,048 | 190.2 | 143.6 | 1.325x | 5275.8 | 4688.0 | 1.125x |
| 4,096 | 182.6 | 141.2 | 1.293x | 5728.8 | 4690.1 | 1.221x |
| 8,192 | 179.9 | 133.9 | 1.343x | 5456.0 | 4518.7 | 1.207x |
| 16,384 | 166.3 | 124.5 | 1.336x | 4903.2 | 4104.9 | 1.194x |
| 32,768 | 150.5 | 116.8 | 1.288x | 3884.0 | 3355.0 | 1.158x |
| 65,536 | 126.0 | 96.6 | 1.304x | 2830.4 | 2428.7 | 1.165x |
| 131,072 | 93.7 | 74.5 | 1.257x | 1781.7 | 1525.9 | 1.168x |

Production job `4138` completed all 40 arms. Every engine/context pair
repeated its 64-token digest exactly, and the largest round-to-round throughput
difference was 2.890%.

## Measurement contract

Both engines use batch size 1, greedy argmax, the same deterministic synthetic
prompt token IDs, a fresh process for every arm, an exact-shape warmup, one
measured prefill, and exactly 64 measured decode steps.

The token at prompt position `i` is:

```text
((i * 2654435761) mod 2^32) mod vocab_size
```

Prefill is TTFT-style prompt throughput: prompt tokens divided by time until
the first generated token is delivered and the first decode is submitted. For
MLX this is the literal first yield of mlx-lm 0.31.3 `generate_step`, matching
its production prompt-TPS boundary. It includes 2,048-token prompt chunks,
cache materialization, the single-token tail, full-vocabulary
log-sum-exp/logprobs, greedy argmax, first-token host extraction, and async
submission of the following token. Warmup lookahead is synchronized before
the measured pass.

Lily mirrors that boundary: it runs `Qwen3_5Model::prefill`, submits the first
production decode command buffer, reads the first greedy token, and then ends
the prefill timer. Its decode timer therefore starts with the first step already
submitted. The depth-2 path submits the following command buffer before waiting
for the current one, and generated IDs remain on the GPU between steps.

MLX decode is the next 64 literal inter-yield intervals from `generate_step`;
Lily decode ends when its 64th generated token is available. Both represent
post-fill token completion cadence rather than isolated single-kernel latency.
Both schedule one-token lookahead before every delivery and explicitly drain
the final unused lookahead outside the timer.

The work inside a token is not identical. Stock mlx-lm computes and returns a
full-vocabulary logprob vector even though the harness discards it; Lily's
minimal production API computes greedy token selection without exposing
logprobs. The claim is therefore production-engine throughput, not that Lily
executes the same graph more efficiently.

The two engines follow their own production greedy trajectories. Each engine's
64-token FNV-1a digest must repeat exactly across its two rounds. A cross-engine
digest comparison is informational. Different tokens can change MoE routes and
memory access, so the ratios describe production free-run throughput, not a
controlled same-token workload or a numerical-equivalence test.

## Hardware and software

- Host: `m5max-top`, Apple M5 Max, 40-core GPU, 128 GB unified memory
- OS: macOS 27.0
- Testbed job: `4138`, exclusive `bench` class
- Rust: 1.92.0, release profile, `--locked`
- Lily binary SHA-256:
  `97734ee7ffd728901acc04d9dd53ae1cfdd120bc93458d97eb28e45bd9f1549a`
- Source-tree SHA-256:
  `347eacfa600e7356b2c16a51e6d902decdc59b21af2cd1e499e7d301bdb65e64`
- `Cargo.lock` SHA-256:
  `ac89db7238fd2114563e065cf86001dbee1f0113cda6eb7428f9013ae41a86dd`
- Matrix runner SHA-256:
  `b3f9af845fe9b5656d0813fa23bc2bc36197443f1a14381ce924f31daf0c5e06`
- MLX harness SHA-256:
  `c4f08568e439e6cad703dd49d68ab95731aa901c2654239b3951628984d1a3df`
- Python requirements SHA-256:
  `7aa22ce31516e006cfe1c5d5752a4f596eb7cb8a7c2ba8dc8f9187b3e73c7bdc`
- MLX environment: Python 3.12.14, [mlx 0.31.2][mlx-0312],
  [mlx-lm 0.31.3][mlx-lm-0313], Transformers 5.14.1
- Checkpoint: `mlx-community/Qwen3.6-35B-A3B-4bit`, revision
  `38740b847e4cb78f352aba30aa41c76e08e6eb46`, MLX affine Q4/group-64
  with the checkpoint-declared affine Q8 router-gate exceptions

## Reproduce

Clone the repository and enter Lily:

```sh
git clone https://github.com/perplexityai/pplx-garden.git
cd pplx-garden/lily
```

Download the immutable checkpoint and create the pinned MLX environment:

```sh
hf download mlx-community/Qwen3.6-35B-A3B-4bit \
  --revision 38740b847e4cb78f352aba30aa41c76e08e6eb46 \
  --local-dir /path/to/Qwen3.6-35B-A3B-4bit

python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r benchmarks/requirements.txt
```

Build the benchmark with the source identity embedded in the binary:

```sh
export LILY_BENCH_SOURCE_ID=$(git rev-parse HEAD)
cargo build --release --locked --bin lily-bench
```

On an M5 Max 40-core/128 GB host on AC power, run:

```sh
python benchmarks/run_matrix.py \
  /path/to/Qwen3.6-35B-A3B-4bit \
  target/release/lily-bench \
  "$VIRTUAL_ENV/bin/python" \
  /path/to/results/pplx-garden-lily-mlx0312-full \
  --source-id "$LILY_BENCH_SOURCE_ID" \
  --rounds 2 \
  --decode-steps 64 \
  --cooldown-short 30 \
  --cooldown-long 300 \
  --cooldown-max 300 \
  --ignore-foreign-load
```

[mlx-0312]: https://pypi.org/project/mlx/0.31.2/
[mlx-lm-0313]: https://pypi.org/project/mlx-lm/0.31.3/
