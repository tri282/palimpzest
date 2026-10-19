"""Model drop-rule screen for the C2 extension: candidate executors under the generic instruction on validation pairs.

Usage: python -I drop_screen.py [--summary] [--host chameleon]. Appends to results/drop_screen[_<host>]/calls.jsonl
(resumable). Design: REPORT.md, REPORT-C2-ext.md.
Drop if accuracy <= majority rate, one answer >= 95% of outputs, or > 10% unparsed (rule fixed before running).
With --host chameleon it is the host check (PREREG-C2-ext.md II.2): self-hosted answers vs the Purdue screen, >= 38 / 40.
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c2_setup import load_pairs, stratified  # noqa: E402
from llm import ask, model_snapshot  # noqa: E402
from probe_api import PURDUE_BASE, REPO, call, load_env  # noqa: E402
from runner import EXECUTOR_TEMP, append, now, read_jsonl  # noqa: E402
from workflows import WORKFLOWS, parse_yes  # noqa: E402

CANDIDATES = ["gemma3:1b", "gemma4:26b-a4b", "qwen3:8b", "qwen3:32b", "llama3.1:latest", "gpt-oss:latest",
              "deepseek-r1:14b", "mistral:latest", "phi4:latest", "qwen2.5:72b", "llama4:latest"]
SEED = 4
STRATA = (20, 15, 5)  # pos, hard neg, random neg: C2 proportions, majority rate 0.50
MAX_ONE_ANSWER, MAX_UNPARSED = 0.95, 0.10
OUT_DIR = REPO / "experiments" / "spec_channel" / "results" / "drop_screen"
# Host check (PREREG-C2-ext.md II.2): the executors moved to the self-hosted server, and the agreement they need.
SELF_HOSTED = ["gemma4:26b-a4b", "qwen3:8b", "qwen3:32b", "deepseek-r1:14b", "mistral:latest", "phi4:latest"]
MIN_AGREE = 38


def snapshot_models() -> None:
    # Purdue exposes no digest or quantization; the registry entry time flags a re-pulled model between runs.
    s, _, resp, _ = call(f"{PURDUE_BASE}/models", {"Authorization": f"Bearer {load_env()['PURDUE']}"})
    models = [{"id": m["id"], "updated_at": m["info"].get("updated_at")} for m in (resp or {}).get("data", [])]
    (OUT_DIR / f"models-{now()[:10]}.json").write_text(json.dumps({"status": s, "fetched": now(), "models": models}, indent=1))


def main(host: str) -> int:
    wf = WORKFLOWS["c2"]()
    pairs = stratified(load_pairs("valid"), *STRATA, random.Random(SEED))
    out_dir, models = (OUT_DIR, CANDIDATES) if host == "purdue" else (OUT_DIR.with_name(f"drop_screen_{host}"), SELF_HOSTED)
    out_dir.mkdir(parents=True, exist_ok=True)
    if host == "purdue":
        snapshot_models()
    else:
        (out_dir / f"models-{now()[:10]}.json").write_text(json.dumps(model_snapshot(host, models), indent=1))
    out = out_dir / "calls.jsonl"

    done = {(r["model"], r["item_id"]) for r in read_jsonl(out) if not r["error"]}
    todo = [(m, p) for m in models for p in pairs if (m, p["id"]) not in done]
    print(f"{len(models)} models x {len(pairs)} pairs on {host}; {len(todo)} to run", flush=True)
    for k, (m, p) in enumerate(todo):
        r = ask(m, wf.executor_prompt(wf.ref_spec, p), EXECUTOR_TEMP, host=host)
        pred = parse_yes(r["content"])
        append(out, {"spec": "S_ref", "item_id": p["id"], "stratum": p["stratum"], "label": p["label"], "prediction": pred,
                     "correct": pred == p["label"], "unparsed": pred is None, "timestamp": now(), **r})
        if (k + 1) % 40 == 0:
            print(f"{k + 1}/{len(todo)} (last model {m})", flush=True)
    ids = [p["id"] for p in pairs]
    if host == "purdue":
        report(out, ids)
    else:
        host_check(out, OUT_DIR / "calls.jsonl", ids, out_dir / "host_check.json")
    return 0


def host_check(out: Path, purdue: Path, ids: list[str], dest: Path) -> None:
    """Per model: answers equal to the Purdue screen's on the same pairs; below MIN_AGREE the model stays on Purdue."""
    mine = {(r["model"], r["item_id"]): r["prediction"] for r in read_jsonl(out) if not r["error"]}
    ref = {(r["model"], r["item_id"]): r["prediction"] for r in read_jsonl(purdue) if not r["error"]}
    verdicts = {}
    for m in SELF_HOSTED:
        both = [i for i in ids if (m, i) in mine and (m, i) in ref]
        agree = sum(mine[(m, i)] == ref[(m, i)] for i in both)
        verdicts[m] = {"compared": len(both), "agree": agree, "self_hosted": len(both) == len(ids) and agree >= MIN_AGREE}
        print(f"{m:18s} {agree:>2}/{len(both)} agree  {'self-host' if verdicts[m]['self_hosted'] else 'stay on Purdue'}")
    dest.write_text(json.dumps({"min_agree": MIN_AGREE, "models": verdicts}, indent=1))


def report(out: Path, ids: list[str]) -> None:
    rows = {(r["model"], r["item_id"]): r for r in read_jsonl(out) if not r["error"]}
    labels = Counter(rows[k]["label"] for k in rows if k[0] == CANDIDATES[0])
    majority = max(labels.values()) / sum(labels.values()) if labels else 0.5
    verdicts = {}
    print(f"majority rate {majority:.2f}\n{'model':18s} {'n':>3} {'acc':>5} {'top':>5} {'unp':>5} {'out_med':>7} {'reas_med':>8} {'lat_med':>7}  verdict")
    for m in CANDIDATES:
        rs = [rows[(m, i)] for i in ids if (m, i) in rows]
        if not rs:
            continue
        acc = statistics.mean(r["correct"] for r in rs)
        top = Counter(r["prediction"] for r in rs).most_common(1)[0][1] / len(rs)  # unparsed counts as an answer class
        unp = statistics.mean(r["unparsed"] for r in rs)
        drop = [why for why, bad in [("acc<=majority", acc <= majority), ("one answer", top >= MAX_ONE_ANSWER),
                                     ("unparsed", unp > MAX_UNPARSED)] if bad]
        verdicts[m] = {"n": len(rs), "acc": acc, "top_share": top, "unparsed": unp, "drop": drop}
        print(f"{m:18s} {len(rs):>3} {acc:5.2f} {top:5.2f} {unp:5.2f} {statistics.median(r['tokens_out'] or 0 for r in rs):7.0f} "
              f"{statistics.median(r['reasoning_chars'] for r in rs):8.0f} {statistics.median(r['latency_s'] for r in rs):6.2f}s  "
              f"{'DROP ' + ', '.join(drop) if drop else 'keep'}")
    (OUT_DIR / "verdicts.json").write_text(json.dumps({"majority": majority, "models": verdicts}, indent=1))


if __name__ == "__main__":
    host = sys.argv[sys.argv.index("--host") + 1] if "--host" in sys.argv else "purdue"
    sys.exit(report(OUT_DIR / "calls.jsonl", [p["id"] for p in stratified(load_pairs("valid"), *STRATA, random.Random(SEED))])
             if "--summary" in sys.argv else main(host))
