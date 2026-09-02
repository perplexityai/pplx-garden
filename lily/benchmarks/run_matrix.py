#!/usr/bin/env python3
"""Run a fail-closed Lily/MLX throughput matrix.

Every arm is a fresh process. Engine order alternates by round, context order
reverses on odd rounds, and the manifest is atomically checkpointed before and
after every arm. A row is refused if either engine's prefill or decode drifts
by 3% or more, produces a non-finite result, or changes its token digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import statistics
import string
import subprocess
import sys
import time
from typing import Any


MAX_DRIFT = 0.03
EXPECTED_MLX = {"mlx": "0.31.2", "mlx-lm": "0.31.3", "transformers": "5.14.1"}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: pathlib.Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def probe(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def interference_sample(ours: int) -> dict[str, str]:
    output = probe(["ps", "-Ao", "pid=,pgid=,pcpu=,rss=,command="])
    if not output:
        raise RuntimeError("ps probe failed or returned no rows")
    candidates: dict[str, str] = {}
    parsed_rows = 0
    for line in output.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        try:
            pgid, cpu, rss = int(parts[1]), float(parts[2]), int(parts[3])
        except ValueError:
            continue
        parsed_rows += 1
        if pgid == ours or (cpu <= 20.0 and rss <= 2 * 1024 * 1024):
            continue
        candidates[parts[0]] = (
            f"pid={parts[0]} cpu={cpu:.1f}% "
            f"rss={rss / 2**20:.2f}GiB {parts[4]}"
        )
    if parsed_rows == 0:
        raise RuntimeError("ps probe returned no parseable process rows")
    return candidates


def interference_snapshot() -> dict[str, Any]:
    """Return sustained foreign load plus the samples that established it.

    A one-shot `ps` sample can catch a harmless launchd/airportd burst exactly
    between cooldown and process launch. Require the same PID to exceed the CPU
    or resident-memory threshold in at least two of three one-second samples.
    This still fails closed on a resident model or sustained CPU consumer while
    avoiding publication refusal for a single completed system-service burst.
    """
    ours = os.getpgid(0)
    samples: list[dict[str, Any]] = []

    def capture() -> dict[str, Any]:
        return {
            "wall_unix": time.time(),
            "monotonic_secs": time.monotonic(),
            "candidates": interference_sample(ours),
        }

    try:
        for index in range(3):
            if index > 0:
                time.sleep(1)
            samples.append(capture())
    except RuntimeError as error:
        return {
            "samples": samples,
            "sustained": [f"interference probe uncertain: {error}"],
            "error": str(error),
        }

    appearances: dict[str, list[str]] = {}
    for sample in samples:
        for pid, description in sample["candidates"].items():
            appearances.setdefault(pid, []).append(description)
    sustained_pids = {
        pid for pid, descriptions in appearances.items() if len(descriptions) >= 2
    }

    # A process that first appears in the last sample has not yet had a chance
    # to prove persistence. Confirm the trailing edge before launching the arm.
    # If each confirmation introduces a new candidate, refuse the unstable
    # window after a bounded number of samples instead of eventually racing it.
    for _ in range(3):
        unresolved = set(samples[-1]["candidates"]) - sustained_pids
        if not unresolved:
            break
        time.sleep(1)
        try:
            confirmation = capture()
        except RuntimeError as error:
            return {
                "samples": samples,
                "sustained": [f"interference probe uncertain: {error}"],
                "error": str(error),
            }
        samples.append(confirmation)
        for pid, description in confirmation["candidates"].items():
            appearances.setdefault(pid, []).append(description)
        sustained_pids.update(unresolved & set(confirmation["candidates"]))

    if set(samples[-1]["candidates"]) - sustained_pids:
        return {
            "samples": samples,
            "sustained": ["unstable foreign-load window at arm boundary"],
            "error": "trailing candidates did not settle",
        }

    sustained = [appearances[pid][-1] for pid in sorted(sustained_pids)]
    return {"samples": samples, "sustained": sustained, "error": None}


def machine_health() -> dict[str, Any]:
    thermal = probe(["pmset", "-g", "therm"])
    power = probe(["pmset", "-g", "ps"])
    thermal_clear = thermal is not None and (
        "No thermal warning level has been recorded" in thermal
        and "No performance warning level has been recorded" in thermal
    )
    ac_power = power is not None and "AC Power" in power.splitlines()[0]
    return {
        "thermal": thermal,
        "power": power,
        "thermal_clear": thermal_clear,
        "ac_power": ac_power,
    }


def model_hashes(model: pathlib.Path) -> dict[str, str]:
    names = [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ]
    files = [model / name for name in names if (model / name).is_file()]
    files.extend(sorted(model.glob("*.safetensors")))
    return {path.name: sha256_file(path) for path in files}


def source_tree_sha256(root: pathlib.Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts[0] in {".git", "target"} or "__pycache__" in relative.parts:
            continue
        digest.update(str(relative).encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def cooldown_for(context: int, args: argparse.Namespace) -> int:
    if context >= 131_072:
        return args.cooldown_max
    if context >= 65_536:
        return args.cooldown_long
    return args.cooldown_short


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def validate_lily_report(
    report: dict[str, Any], source_id: str, decode_steps: int, diagnostic: bool
) -> None:
    if report.get("meta", {}).get("source_id") != source_id:
        raise ValueError("Lily report source id mismatch")
    workload = report.get("workload", {})
    if workload.get("gpu_timing_diagnostic") is not diagnostic:
        raise ValueError("Lily report diagnostic mode mismatch")
    decode = report.get("results", {}).get("decode", {})
    if len(decode.get("step_wall_secs", [])) != decode_steps:
        raise ValueError("Lily report decode interval count mismatch")
    if len(decode.get("token_ids", [])) != decode_steps:
        raise ValueError("Lily report token count mismatch")
    gpu_passes = decode.get("gpu_passes")
    lookahead = decode.get("lookahead_gpu_pass")
    if diagnostic:
        if not isinstance(gpu_passes, list) or len(gpu_passes) != decode_steps:
            raise ValueError("Lily diagnostic GPU pass count mismatch")
        if not isinstance(lookahead, dict):
            raise ValueError("Lily diagnostic lookahead timing missing")
    elif gpu_passes != [] or lookahead is not None:
        raise ValueError("Lily production report contains diagnostic GPU timing")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("lily_bin", type=pathlib.Path)
    parser.add_argument("mlx_python", type=pathlib.Path)
    parser.add_argument("out_dir", type=pathlib.Path)
    parser.add_argument("--source-id", required=True)
    parser.add_argument(
        "--contexts",
        default="256,512,1024,2048,4096,8192,16384,32768,65536,131072",
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--lily-gpu-timing", action="store_true")
    parser.add_argument(
        "--ignore-foreign-load",
        action="store_true",
        help="record foreign-load samples without refusing otherwise stable measurements",
    )
    parser.add_argument("--cooldown-short", type=int, default=30)
    parser.add_argument("--cooldown-long", type=int, default=120)
    parser.add_argument("--cooldown-max", type=int, default=300)
    args = parser.parse_args()

    contexts = [int(value) for value in args.contexts.split(",")]
    if len(args.source_id) != 40 or any(char not in string.hexdigits for char in args.source_id):
        raise ValueError("--source-id must be a full 40-character git SHA")
    if args.rounds < 2 or args.decode_steps <= 0 or any(value <= 0 for value in contexts):
        raise ValueError("need at least two rounds and positive context/decode sizes")
    for path in (args.model, args.lily_bin, args.mlx_python):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty output directory {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    root = pathlib.Path(__file__).resolve().parents[1]
    mlx_script = pathlib.Path(__file__).resolve().with_name("bench_mlx.py")
    requirements = mlx_script.with_name("requirements.txt")
    version_code = (
        "import importlib.metadata,json; "
        "print(json.dumps({n:importlib.metadata.version(n) for n in "
        "['mlx','mlx-lm','transformers']}))"
    )
    version_result = subprocess.run(
        [str(args.mlx_python), "-c", version_code], capture_output=True, text=True, check=True
    )
    mlx_versions = json.loads(version_result.stdout)
    if mlx_versions != EXPECTED_MLX:
        raise RuntimeError(f"MLX pin mismatch: got {mlx_versions}, want {EXPECTED_MLX}")
    git_head = probe(["git", "-C", str(root), "rev-parse", "HEAD"])
    git_dirty = probe(["git", "-C", str(root), "status", "--porcelain"])
    if git_head is None:
        raise RuntimeError("benchmark source must be a git checkout")
    if git_head != args.source_id:
        raise RuntimeError(f"source checkout {git_head} != --source-id {args.source_id}")
    if git_dirty:
        raise RuntimeError("source checkout is dirty")

    schedule: list[dict[str, Any]] = []
    for round_index in range(args.rounds):
        round_contexts = contexts if round_index % 2 == 0 else list(reversed(contexts))
        engines = ["lily", "mlx"] if round_index % 2 == 0 else ["mlx", "lily"]
        for context in round_contexts:
            for engine in engines:
                schedule.append(
                    {
                        "round": round_index,
                        "context": context,
                        "engine": engine,
                        "cooldown_secs": cooldown_for(context, args),
                    }
                )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "incomplete",
        "measurement_mode": "diagnostic" if args.lily_gpu_timing else "production",
        "source_id": args.source_id,
        "started_unix": time.time(),
        "protocol": {
            "rounds": args.rounds,
            "contexts": contexts,
            "decode_steps": args.decode_steps,
            "lily_gpu_timing": args.lily_gpu_timing,
            "ignore_foreign_load": args.ignore_foreign_load,
            "max_drift": MAX_DRIFT,
            "context_order": "ascending then descending",
            "engine_order": "lily-mlx then mlx-lily",
            "cooldown_short": args.cooldown_short,
            "cooldown_long": args.cooldown_long,
            "cooldown_max": args.cooldown_max,
        },
        "machine": {
            "chip": probe(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "os": probe(["sysctl", "-n", "kern.osproductversion"]),
            "mem_bytes": int(probe(["sysctl", "-n", "hw.memsize"]) or 0),
            "gpu": probe(["system_profiler", "SPDisplaysDataType"]),
        },
        "inputs": {
            "model": str(args.model.resolve()),
            "model_hashes": model_hashes(args.model),
            "source_tree_sha256": source_tree_sha256(root),
            "git_head": git_head,
            "git_dirty": git_dirty,
            "lily_binary": str(args.lily_bin.resolve()),
            "lily_binary_sha256": sha256_file(args.lily_bin),
            "cargo_lock_sha256": sha256_file(root / "Cargo.lock"),
            "matrix_sha256": sha256_file(pathlib.Path(__file__).resolve()),
            "mlx_harness_sha256": sha256_file(mlx_script),
            "requirements_sha256": sha256_file(requirements),
            "mlx_versions": mlx_versions,
            "mlx_python": probe([str(args.mlx_python), "--version"]),
        },
        "schedule": schedule,
        "runs": [],
    }
    manifest_path = args.out_dir / "manifest.json"
    atomic_json(manifest_path, manifest)

    clean_env = {key: value for key, value in os.environ.items() if not key.startswith("LILY_")}
    clean_env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    for index, arm in enumerate(schedule):
        cooldown = arm["cooldown_secs"]
        print(
            f"arm {index + 1}/{len(schedule)}: cooldown {cooldown}s, "
            f"r{arm['round']} {arm['engine']} @{arm['context']}",
            flush=True,
        )
        time.sleep(cooldown)
        destination = args.out_dir / (
            f"r{arm['round']}-{arm['engine']}-{arm['context']}.json"
        )
        if arm["engine"] == "lily":
            command = [
                str(args.lily_bin),
                "--model",
                str(args.model),
                "--prompt-len",
                str(arm["context"]),
                "--decode-steps",
                str(args.decode_steps),
                "--json-out",
                str(destination),
            ]
            if args.lily_gpu_timing:
                command.append("--gpu-timing")
        else:
            command = [
                str(args.mlx_python),
                str(mlx_script),
                "--model",
                str(args.model),
                "--prompt-len",
                str(arm["context"]),
                "--decode-steps",
                str(args.decode_steps),
                "--json-out",
                str(destination),
            ]
        interference_before = interference_snapshot()
        busy = interference_before["sustained"]
        health_before = machine_health()
        record = {
            **arm,
            "status": "running",
            "json": destination.name,
            "command": command,
            "interference_before": interference_before,
            "memory_pressure": probe(["memory_pressure", "-Q"]),
            "health_before": health_before,
            "started_unix": time.time(),
        }
        manifest["runs"].append(record)
        atomic_json(manifest_path, manifest)
        if busy and not args.ignore_foreign_load:
            record["status"] = "refused"
            manifest["status"] = "refused"
            manifest["failure"] = f"foreign load before arm {index}: {busy}"
            atomic_json(manifest_path, manifest)
            print(manifest["failure"], file=sys.stderr)
            return 3
        if busy:
            print(f"foreign load before arm {index} (recorded, not gated): {busy}", flush=True)
        if not health_before["thermal_clear"] or not health_before["ac_power"]:
            record["status"] = "refused"
            manifest["status"] = "refused"
            manifest["failure"] = f"unhealthy machine state before arm {index}: {health_before}"
            atomic_json(manifest_path, manifest)
            print(manifest["failure"], file=sys.stderr)
            return 3
        result = subprocess.run(command, capture_output=True, text=True, env=clean_env)
        validation_error = None
        if result.returncode == 0 and arm["engine"] == "lily":
            try:
                validate_lily_report(
                    json.loads(destination.read_text()),
                    args.source_id,
                    args.decode_steps,
                    args.lily_gpu_timing,
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                validation_error = str(error)
        interference_after = interference_snapshot()
        busy_after = interference_after["sustained"]
        health_after = machine_health()
        record.update(
            {
                "status": (
                    "complete"
                    if result.returncode == 0 and validation_error is None
                    else "failed"
                ),
                "returncode": result.returncode,
                "validation_error": validation_error,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "interference_after": interference_after,
                "health_after": health_after,
                "finished_unix": time.time(),
            }
        )
        atomic_json(manifest_path, manifest)
        if result.returncode != 0 or validation_error is not None:
            manifest["status"] = "failed"
            manifest["failure"] = (
                f"arm {index} returned {result.returncode}; "
                f"validation_error={validation_error!r}"
            )
            atomic_json(manifest_path, manifest)
            print(result.stdout, end="")
            print(result.stderr, end="", file=sys.stderr)
            return 1
        if (
            (busy_after and not args.ignore_foreign_load)
            or not health_after["thermal_clear"]
            or not health_after["ac_power"]
        ):
            record["status"] = "refused"
            manifest["status"] = "refused"
            manifest["failure"] = (
                f"unhealthy machine state after arm {index}: "
                f"foreign_load={busy_after} health={health_after}"
            )
            atomic_json(manifest_path, manifest)
            print(manifest["failure"], file=sys.stderr)
            return 3
        if busy_after:
            print(f"foreign load after arm {index} (recorded, not gated): {busy_after}", flush=True)
        print((result.stdout + result.stderr).strip(), flush=True)

    reports: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for record in manifest["runs"]:
        report = json.loads((args.out_dir / record["json"]).read_text())
        if record["engine"] == "lily" and report["meta"]["source_id"] != args.source_id:
            raise RuntimeError(
                f"Lily source id {report['meta']['source_id']} != {args.source_id}"
            )
        reports.setdefault((record["context"], record["engine"]), []).append(report)

    failures: list[str] = []
    rows: list[dict[str, Any]] = []
    for context in contexts:
        row: dict[str, Any] = {"context": context, "engines": {}}
        for engine in ("lily", "mlx"):
            engine_reports = reports.get((context, engine), [])
            if len(engine_reports) != args.rounds:
                failures.append(f"@{context} {engine}: {len(engine_reports)} reports")
                continue
            prefill = [float(item["results"]["prefill"]["tok_s"]) for item in engine_reports]
            decode = [float(item["results"]["decode"]["tok_s"]) for item in engine_reports]
            digests = [item["results"]["decode"]["token_digest"] for item in engine_reports]
            for metric, values in (("prefill", prefill), ("decode", decode)):
                if any(not math.isfinite(value) or value <= 0 for value in values):
                    failures.append(f"@{context} {engine} {metric}: non-finite/non-positive")
                    continue
                drift = values[-1] / values[0]
                if abs(drift - 1.0) >= MAX_DRIFT:
                    failures.append(
                        f"@{context} {engine} {metric}: drift {drift:.6f} exceeds {MAX_DRIFT:.0%}"
                    )
            if len(set(digests)) != 1:
                failures.append(f"@{context} {engine}: token digest changed {digests}")
            row["engines"][engine] = {
                "prefill_rounds": prefill,
                "prefill_median": median(prefill),
                "decode_rounds": decode,
                "decode_median": median(decode),
                "token_digests": digests,
            }
        if {"lily", "mlx"} <= row["engines"].keys():
            lily = row["engines"]["lily"]
            mlx = row["engines"]["mlx"]
            row["prefill_ratio"] = lily["prefill_median"] / mlx["prefill_median"]
            row["decode_ratio"] = lily["decode_median"] / mlx["decode_median"]
            row["prefill_paired_ratios"] = [
                left / right
                for left, right in zip(lily["prefill_rounds"], mlx["prefill_rounds"])
            ]
            row["decode_paired_ratios"] = [
                left / right
                for left, right in zip(lily["decode_rounds"], mlx["decode_rounds"])
            ]
            row["cross_engine_digest_match"] = (
                lily["token_digests"][0] == mlx["token_digests"][0]
            )
        rows.append(row)

    summary = {
        "schema_version": 1,
        "status": "refused" if failures else "complete",
        "measurement_mode": "diagnostic" if args.lily_gpu_timing else "production",
        "source_id": args.source_id,
        "failures": failures,
        "rows": rows,
    }
    atomic_json(args.out_dir / "summary.json", summary)
    manifest["status"] = summary["status"]
    manifest["finished_unix"] = time.time()
    manifest["failures"] = failures
    atomic_json(manifest_path, manifest)

    if failures:
        print("REFUSED:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 2
    if args.lily_gpu_timing:
        print(f"MATRIX_DIAGNOSTIC_COMPLETE {args.out_dir}")
        return 0
    print("| ctx | Lily decode | MLX decode | decode | Lily prefill | MLX prefill | prefill |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        if {"lily", "mlx"} <= row["engines"].keys():
            lily = row["engines"]["lily"]
            mlx = row["engines"]["mlx"]
            print(
                f"| {row['context']:,} | {lily['decode_median']:.1f} | "
                f"{mlx['decode_median']:.1f} | {row['decode_ratio']:.3f}x | "
                f"{lily['prefill_median']:.1f} | {mlx['prefill_median']:.1f} | "
                f"{row['prefill_ratio']:.3f}x |"
            )
    print(f"MATRIX_COMPLETE {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
