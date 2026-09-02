#!/usr/bin/env python3
"""Measure MLX's production generate_step + async-eval pipeline.

The first generator yield uses mlx-lm's production prompt-TPS boundary. The
next N inter-yield intervals measure its steady-state token completion cadence.
generate_step intentionally enqueues the following token before each yield.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import pathlib
import platform
import socket
import subprocess
import time

import mlx.core as mx
import mlx_lm
from mlx_lm import load
from mlx_lm.generate import generate_step, generation_stream


EXPECTED_VERSIONS = {
    "mlx": "0.31.2",
    "mlx-lm": "0.31.3",
    "transformers": "5.14.1",
}


def synthetic_ids(count: int, vocab: int) -> list[int]:
    return [((i * 2_654_435_761) & 0xFFFFFFFF) % vocab for i in range(count)]


def fnv1a(tokens: list[int]) -> str:
    digest = 0xCBF29CE484222325
    for token in tokens:
        digest = ((digest ^ token) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{digest:016x}"


def probe(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def run_generator(model, prompt: mx.array, steps: int):
    """Time first yield plus exactly N production inter-yield intervals."""
    generator = generate_step(prompt, model, max_tokens=steps + 1)
    yielded: list[int] = []
    decode_intervals: list[float] = []
    start = time.perf_counter()
    previous = start
    prefill_secs = 0.0
    for index, (token, _logprobs) in enumerate(generator):
        now = time.perf_counter()
        if index == 0:
            prefill_secs = now - start
        else:
            decode_intervals.append(now - previous)
        previous = now
        yielded.append(int(token))
        if index == steps:
            break
    # generate_step has already async-submitted one-token lookahead before the
    # final yield. Drain it outside the timers so neither warmup nor a completed
    # arm can leak work into the next phase/process teardown.
    mx.synchronize(generation_stream)
    if len(yielded) != steps + 1 or len(decode_intervals) != steps:
        raise RuntimeError(
            f"generate_step yielded {len(yielded)} tokens and "
            f"{len(decode_intervals)} decode intervals, expected {steps + 1}/{steps}"
        )
    if not math.isfinite(prefill_secs) or prefill_secs <= 0:
        raise RuntimeError(f"invalid prefill time {prefill_secs}")
    if any(not math.isfinite(value) or value <= 0 for value in decode_intervals):
        raise RuntimeError("invalid decode interval")
    return prefill_secs, decode_intervals, yielded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--prompt-len", type=int, required=True)
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--json-out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.prompt_len <= 0 or args.decode_steps <= 0:
        raise ValueError("prompt length and decode steps must be positive")

    versions = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    if versions != EXPECTED_VERSIONS:
        raise RuntimeError(f"package pin mismatch: got {versions}, want {EXPECTED_VERSIONS}")

    config = json.loads((args.model / "config.json").read_text())
    text_config = config.get("text_config", config)
    prompt = mx.array(synthetic_ids(args.prompt_len, text_config["vocab_size"]))
    model, _tokenizer = load(str(args.model))

    # Exact-shape warmup. run_generator synchronizes the generator's async
    # lookahead before its cache is released.
    run_generator(model, prompt, 1)
    mx.clear_cache()
    mx.reset_peak_memory()

    prefill_secs, decode_intervals, yielded = run_generator(
        model, prompt, args.decode_steps
    )
    decode_secs = sum(decode_intervals)
    decode_tokens = yielded[1 : args.decode_steps + 1]
    report = {
        "schema_version": 1,
        "meta": {
            "engine": "mlx",
            "harness": "benchmarks/bench_mlx.py",
            "versions": versions,
            "python": platform.python_version(),
            "hostname": socket.gethostname().split(".")[0],
        },
        "machine": {
            "chip": probe(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "os": probe(["sysctl", "-n", "kern.osproductversion"]),
            "mem_bytes": int(probe(["sysctl", "-n", "hw.memsize"]) or 0),
        },
        "workload": {
            "prompt_len": args.prompt_len,
            "prompt_kind": "u32_golden_ratio_hash_mod_vocab",
            "decode_steps": args.decode_steps,
            "decode_mode": "mlx_lm.generate_step_inter_yield_async_eval",
        },
        "results": {
            "prefill": {
                "wall_secs": prefill_secs,
                "tok_s": args.prompt_len / prefill_secs,
                "first_token_id": yielded[0],
            },
            "decode": {
                "wall_secs": decode_secs,
                "tok_s": args.decode_steps / decode_secs,
                "step_wall_secs": decode_intervals,
                "token_digest": fnv1a(decode_tokens),
                "token_ids": decode_tokens,
            },
        },
        "peak_memory_bytes": int(mx.get_peak_memory()),
    }
    args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"prefill: {args.prompt_len} tok in {prefill_secs:.6f}s "
        f"({args.prompt_len / prefill_secs:.1f} tok/s) | decode: "
        f"{args.decode_steps} steps in {decode_secs:.6f}s "
        f"({args.decode_steps / decode_secs:.1f} tok/s) | "
        f"digest={report['results']['decode']['token_digest']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
