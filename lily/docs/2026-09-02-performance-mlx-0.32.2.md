# 2026-09-02: Lily vs MLX 0.32.2

This report compares Lily with MLX on Qwen3.6-35B-A3B. It measures in-process
production generation throughput, not identical isolated model graphs and not
HTTP end-to-end latency. Model loading, tokenization, chat-template rendering,
HTTP, and detokenization are outside the timed regions.

## Results

Unit: tokens/second. Each value is the median of two recorded rounds. Ratios
are Lily divided by MLX.

| Context | Lily decode | MLX decode | Lily / MLX | Lily prefill | MLX prefill | Lily / MLX |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 194.7 | 148.5 | 1.311x | 2866.6 | 2536.4 | 1.130x |
| 512 | 193.8 | 149.5 | 1.296x | 3859.9 | 3309.5 | 1.166x |
| 1,024 | 193.3 | 148.1 | 1.305x | 4734.6 | 4471.7 | 1.059x |
| 2,048 | 190.4 | 147.6 | 1.290x | 5274.3 | 5121.9 | 1.030x |
| 4,096 | 182.8 | 145.6 | 1.255x | 5724.4 | 5165.7 | 1.108x |
| 8,192 | 180.5 | 136.8 | 1.319x | 5457.6 | 5031.8 | 1.085x |
| 16,384 | 167.0 | 129.6 | 1.289x | 4968.5 | 4722.5 | 1.052x |
| 32,768 | 151.1 | 118.2 | 1.279x | 3927.5 | 3963.7 | 0.991x |
| 65,536 | 125.4 | 98.5 | 1.273x | 2752.5 | 3051.1 | 0.902x |
| 131,072 | 92.7 | 75.0 | 1.236x | 1792.7 | 2237.5 | 0.801x |

The 256-token row comes from production job `4128`; contexts 512 through
131,072 come from the paired records in job `4129`. Every published
engine/context pair repeated its 64-token digest exactly. The largest
round-to-round throughput difference in the table is 0.995%.

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
cache materialization and clearing, the single-token tail, full-vocabulary
log-sum-exp/logprobs, greedy argmax, first-token host extraction, and async
submission of the following token.

Lily mirrors that boundary: it runs `Qwen3_5Model::prefill`, submits the first
production decode command buffer, reads the first greedy token, and then ends
the prefill timer. Its decode timer therefore starts with the first step already
submitted. The depth-2 path submits the following command buffer before waiting
for the current one, and generated IDs remain on the GPU between steps.

MLX decode is the next 64 literal inter-yield intervals from `generate_step`;
Lily decode ends when its 64th generated token is available. Both represent
post-fill token completion cadence rather than isolated single-kernel latency.
Both schedule one-token lookahead before every delivery and explicitly drain
the final unused lookahead outside the timer, including after warmup, so no
asynchronous work leaks into the measured pass.

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
- Testbed jobs: `4128` (context 256) and `4129` (contexts 512–131,072),
  exclusive `bench` class
- Power and health: AC power required; thermal and performance warnings cause
  immediate refusal before or after an arm
- Rust: 1.92.0, release profile, `--locked`
- Lily binary SHA-256:
  `8e7d2b6b5c57f7586ed67fd044db2484dbcbc9db9ba086fdf8f2298e515e3996`
- `Cargo.lock` SHA-256:
  `ac89db7238fd2114563e065cf86001dbee1f0113cda6eb7428f9013ae41a86dd`
- Matrix runner SHA-256:
  `6d3651e1f043ca313eb1877e2e61defa3b8fc58f61483883903069f2963a9655`
- MLX harness SHA-256:
  `b112c8a9bcee48d53d4fb0e98b89627970fad65540d7af3747daa85813bacd4e`
- Python requirements SHA-256:
  `deda00e5e8f1b1e00260691384e1a80f7127fc81b2e5ee2fe37de74f8b95a740`
- MLX environment: Python 3.12.14, [mlx 0.32.2][mlx-0322],
  [mlx-lm 0.31.3][mlx-lm-0313], Transformers 5.14.1. MLX 0.32.2 was the
  current core release on the measurement date; mlx-lm 0.31.3 declares
  `mlx>=0.31.2` on macOS.
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

On an otherwise idle M5 Max 40-core/128 GB host on AC power, run:

```sh
python benchmarks/run_matrix.py \
  /path/to/Qwen3.6-35B-A3B-4bit \
  target/release/lily-bench \
  "$VIRTUAL_ENV/bin/python" \
  /path/to/results/pplx-garden-lily-mlx0322-full-600s-20260901 \
  --source-id "$LILY_BENCH_SOURCE_ID" \
  --rounds 2 \
  --decode-steps 64 \
  --cooldown-short 30 \
  --cooldown-long 120 \
  --cooldown-max 600
```

[mlx-0322]: https://pypi.org/project/mlx/0.32.2/
[mlx-lm-0313]: https://pypi.org/project/mlx-lm/0.31.3/
