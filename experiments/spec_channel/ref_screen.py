"""Screen candidate held-out reference executors for C2 on validation pairs (D6): degeneracy, accuracy, spec sensitivity, estimate error.

Usage: python -I ref_screen.py. Appends to results/ref_screen.jsonl (resumable). Design: REPORT.md.
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c2_setup import load_pairs, stratified  # noqa: E402
from em_lexical_baseline import f1  # noqa: E402
from llm import ask  # noqa: E402
from probe_api import REPO  # noqa: E402
from runner import EXECUTOR_TEMP, EXECUTORS, append, read_jsonl  # noqa: E402
from workflows import WORKFLOWS, parse_yes  # noqa: E402

CANDIDATES = ["mistral:latest", "gemma3:27b", "llama3.1:latest", "gpt-oss:120b", "gemma4:26b-a4b", "llama4:latest"]
SEED = 3
STRATA = (10, 10, 4)  # pos, hard neg, random neg
PILOT_SPECS = REPO / "experiments" / "spec_channel" / "results" / "runs" / "pilot-c2" / "specs.jsonl"
OUT = REPO / "experiments" / "spec_channel" / "results" / "ref_screen.jsonl"


def main() -> int:
    wf = WORKFLOWS["c2"]()
    pairs = stratified(load_pairs("valid"), *STRATA, random.Random(SEED))
    pilot = {(s["replicate"], s["arm_a"]): s["content"] for s in read_jsonl(PILOT_SPECS) if not s["error"]}
    specs = {"S_ref": wf.ref_spec, "A0": pilot[(0, "A0")], "A1": pilot[(0, "A1")]}
    models = list(EXECUTORS.values()) + CANDIDATES  # candidate executors give the truth the reference must predict

    done = {(r["model"], r["spec"], r["item_id"]) for r in read_jsonl(OUT) if not r["error"]}
    todo = [(m, s, p) for m in models for s in specs for p in pairs if (m, s, p["id"]) not in done]
    print(f"{len(models)} models x {len(specs)} specs x {len(pairs)} pairs; {len(todo)} to run", flush=True)
    for k, (m, s, p) in enumerate(todo):
        r = ask(m, wf.executor_prompt(specs[s], p), EXECUTOR_TEMP)
        pred = parse_yes(r["content"])
        append(OUT, {"spec": s, "item_id": p["id"], "stratum": p["stratum"], "label": p["label"], "prediction": pred,
                     "correct": pred == p["label"], "unparsed": pred is None, **{x: r[x] for x in
                     ("model", "content", "tokens_in", "tokens_out", "reasoning_chars", "latency_s", "retry_count", "error")}})
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(todo)}", flush=True)
    report(models, list(specs), [p["id"] for p in pairs])
    return 0


def report(models: list[str], specs: list[str], ids: list[str]) -> None:
    rows = {(r["model"], r["spec"], r["item_id"]): r for r in read_jsonl(OUT) if not r["error"]}
    acc = {(m, s): statistics.mean(rows[(m, s, i)]["correct"] for i in ids) for m in models for s in specs}
    print(f"\n{'model':16s} {'spec':6s} {'acc':>5} {'f1':>5} {'yes':>5} {'unp':>4} {'out_med':>7} {'lat_med':>7}")
    for m in models:
        for s in specs:
            rs = [rows[(m, s, i)] for i in ids]
            print(f"{m:16s} {s:6s} {acc[(m, s)]:5.2f} {f1([r['prediction'] or 0 for r in rs], [r['label'] for r in rs]):5.2f} "
                  f"{statistics.mean((r['prediction'] or 0) for r in rs):5.2f} {sum(r['unparsed'] for r in rs):>4} "
                  f"{statistics.median(r['tokens_out'] or 0 for r in rs):7.0f} {statistics.median(r['latency_s'] for r in rs):6.2f}s")
    # Estimate error: predict acc(writer, executor) = acc(writer, ref) * acc(S_ref, executor) / acc(S_ref, ref), vs measured.
    print("\nreference -> mean |estimate - measured| over writers x candidate executors (lower is better)")
    for ref in list(EXECUTORS.values()) + CANDIDATES:
        errs = [abs(acc[(ref, a)] * acc[(b, "S_ref")] / max(acc[(ref, "S_ref")], 1e-9) - acc[(b, a)])
                for a in specs if a != "S_ref" for b in EXECUTORS.values() if b != ref]
        print(f"  {ref:16s} {statistics.mean(errs):.3f}")


if __name__ == "__main__":
    sys.exit(main())
