"""Pilot runner for the Specification channel: fresh specs per replicate, shuffled downstream calls, resumable JSONL logs.

Usage: python -I runner.py <run_id> --workflow c1|c2|c2_full [--replicates 3] [--items N] [--cells all|factorial] [--ref-executor] [--seed 0]
Outputs results/runs/<run_id>/{config.json, specs.jsonl, calls.jsonl}. Design and results: REPORT.md.
Exit codes: 0 done, 2 transient stop (provider outage; re-run resumes), 3 config mismatch (do not retry).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import ask  # noqa: E402
from probe_api import REPO  # noqa: E402
from workflows import WORKFLOWS  # noqa: E402

WRITERS = {"A0": "gemini-3.5-flash-lite", "A1": "qwen2.5:72b"}
EXECUTORS = {"B0": "gemma3:12b", "B1": "llama3.3:70b"}
REF_EXECUTORS = {"B2": "gpt-oss:120b", "B3": "gemma3:27b"}  # profiling-only references, never candidate plans (D6)
WRITER_TEMP, EXECUTOR_TEMP = 0.7, 0.0
REF = "S_ref"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


MAX_CONSECUTIVE_FAILURES = 5


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"skipping truncated line in {path.name}", flush=True)  # killed mid-write
    return rows


def append(path: Path, row: dict) -> None:
    # A kill mid-write leaves a line without "\n"; start a fresh line so this row is not glued onto it.
    torn = path.exists() and path.stat().st_size > 0 and path.read_bytes()[-1:] != b"\n"
    with path.open("a") as f:
        f.write(("\n" if torn else "") + json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def latest_ok(rows: list[dict]) -> dict[tuple, dict]:
    # Failed attempts stay in calls.jsonl as reliability data; a later successful row for the same key supersedes them.
    out = {}
    for r in rows:
        key = (r["replicate"], r["arm_a"], r["arm_b"], r["item_id"])
        if not r["error"] or key not in out:
            out[key] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--workflow", required=True, choices=sorted(WORKFLOWS))
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--items", type=int, default=0, help="limit pilot items (0 = all)")
    ap.add_argument("--cells", choices=["all", "factorial"], default="all")
    ap.add_argument("--ref-executor", action="store_true", help="add the held-out reference executors B2, B3")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    wf = WORKFLOWS[args.workflow]()
    items = wf.pilot[: args.items] if args.items else wf.pilot
    a_arms = list(WRITERS) + ([REF] if args.cells == "all" else [])
    executors = EXECUTORS | (REF_EXECUTORS if args.ref_executor else {})
    cells = [(a, b) for a in a_arms for b in executors]

    run_dir = REPO / "experiments" / "spec_channel" / "results" / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "writers": WRITERS, "executors": executors, "writer_temp": WRITER_TEMP,
              "executor_temp": EXECUTOR_TEMP, "cells": cells, "n_items": len(items), "writer_prompt": wf.writer_prompt,
              "ref_spec": wf.ref_spec}
    cfg_path = run_dir / "config.json"
    config = json.loads(json.dumps(config))  # tuples -> lists, so a resumed run compares equal
    if cfg_path.exists() and json.loads(cfg_path.read_text()) | {"started": None} != config | {"started": None}:
        print(f"config mismatch with existing run {args.run_id}; use a new run_id", flush=True)
        return 3
    if not cfg_path.exists():
        cfg_path.write_text(json.dumps({**config, "started": now()}, indent=1))
    specs_path, calls_path = run_dir / "specs.jsonl", run_dir / "calls.jsonl"
    rng = random.Random(args.seed)

    # Phase 1: one fresh spec per writer arm per replicate, generated in shuffled order. Resumes from specs.jsonl.
    specs = {(s["replicate"], s["arm_a"]): s for s in read_jsonl(specs_path) if not s["error"]}
    spec_jobs = [(rep, a) for rep in range(args.replicates) for a in WRITERS]
    rng.shuffle(spec_jobs)
    for rep, a in spec_jobs:
        if (rep, a) in specs:
            continue
        r = ask(WRITERS[a], wf.writer_prompt, WRITER_TEMP)
        row = {"run_id": args.run_id, "workflow": wf.name, "seed": args.seed, "replicate": rep, "arm_a": a, "operator": "up",
               "spec_hash": hashlib.sha256((r["content"] or "").encode()).hexdigest()[:12], "spec_words": len((r["content"] or "").split()),
               "timestamp": now(), **r}
        append(specs_path, row)
        if row["error"]:
            print(f"spec generation failed for replicate {rep} arm {a}: {row['error']}; stopping. Re-run to resume.", flush=True)
            return 2
        specs[(rep, a)] = row
        print(f"spec rep={rep} {a} words={row['spec_words']} tokens_out={row['tokens_out']}", flush=True)

    def spec_for(rep: int, a: str) -> tuple[str, str]:
        if a == REF:
            return wf.ref_spec, hashlib.sha256(wf.ref_spec.encode()).hexdigest()[:12]
        return specs[(rep, a)]["content"], specs[(rep, a)]["spec_hash"]

    # Phase 2: downstream calls, randomized within (replicate, executor) blocks; replicate order and executor order per
    # replicate are shuffled too (R4). Not fully shuffled: Purdue reloads the model on every switch (30-50 s measured).
    done = {k for k, c in latest_ok(read_jsonl(calls_path)).items() if not c["error"]}
    jobs = []
    for rep in rng.sample(range(args.replicates), args.replicates):
        for b in rng.sample(list(executors), len(executors)):
            block = [(rep, a, b, it) for a, bb in cells if bb == b for it in items]
            rng.shuffle(block)
            jobs += block
    call_index = {(j[0], j[1], j[2], j[3]["id"]): i for i, j in enumerate(jobs)}
    todo = [j for j in jobs if (j[0], j[1], j[2], j[3]["id"]) not in done]
    print(f"downstream: {len(jobs)} planned, {len(jobs) - len(todo)} done, {len(todo)} to run", flush=True)

    prev_model, t0, failures = None, time.time(), 0
    for k, (rep, a, b, it) in enumerate(todo):
        spec, spec_hash = spec_for(rep, a)
        r = ask(executors[b], wf.executor_prompt(spec, it), EXECUTOR_TEMP)
        pred = wf.parse(r["content"])
        append(calls_path, {
            "run_id": args.run_id, "workflow": wf.name, "seed": args.seed, "replicate": rep, "arm_a": a, "arm_b": b,
            "operator": "down", "item_id": it["id"], "stratum": it.get("stratum"), "spec_hash": spec_hash,
            "call_index": call_index[(rep, a, b, it["id"])], "model_switch": prev_model not in (None, executors[b]),
            "label": it["label"], "prediction": pred, "correct": pred == it["label"], "unparsed": pred is None,
            "timestamp": now(), **r})
        prev_model = executors[b]
        failures = failures + 1 if r["error"] else 0
        if failures >= MAX_CONSECUTIVE_FAILURES:
            print(f"{failures} consecutive failed calls (last: {r['error']}); stopping. Re-run the same command to resume.", flush=True)
            return 2
        if (k + 1) % 25 == 0:
            rate = (k + 1) / (time.time() - t0) * 60
            print(f"{k + 1}/{len(todo)} calls, {rate:.1f}/min, eta {(len(todo) - k - 1) / rate:.0f} min", flush=True)

    summarize(calls_path, cells, args.replicates)
    return 0


def summarize(calls_path: Path, cells: list, replicates: int) -> None:
    all_rows = read_jsonl(calls_path)
    rows = list(latest_ok(all_rows).values())
    print(f"rows: {len(all_rows)} logged, {len(rows)} unique calls, {len(all_rows) - len(rows)} superseded failures")
    print(f"\n{'rep':>3} {'cell':10s} {'n':>4} {'acc':>5} {'unparsed':>8} {'retries':>7} {'errors':>6} {'tok_in':>7} {'tok_out':>7} {'lat_med':>7}")
    for rep in range(replicates):
        for a, b in cells:
            rs = [r for r in rows if r["replicate"] == rep and r["arm_a"] == a and r["arm_b"] == b]
            if not rs:
                continue
            lat = sorted(r["latency_s"] for r in rs)
            print(f"{rep:>3} {a + '/' + b:10s} {len(rs):>4} {sum(r['correct'] for r in rs) / len(rs):5.2f} {sum(r['unparsed'] for r in rs):>8} "
                  f"{sum(r['retry_count'] for r in rs):>7} {sum(bool(r['error']) for r in rs):>6} {sum(r['tokens_in'] or 0 for r in rs):>7} "
                  f"{sum(r['tokens_out'] or 0 for r in rs):>7} {lat[len(lat) // 2]:6.2f}s")


if __name__ == "__main__":
    sys.exit(main())
