"""Cheap screens S1 (spec variation across writer arms) and S2 (executor sensitivity to spec) for one workflow.

Usage: python -I screen.py c1|c2. Design and results: REPORT.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from em_lexical_baseline import jaccard, tokens  # noqa: E402
from llm import ask  # noqa: E402
from probe_api import REPO  # noqa: E402
from workflows import WORKFLOWS  # noqa: E402

WRITERS = ["gemini-3.5-flash-lite", "qwen2.5:72b"]
EXECUTORS = ["gemma3:12b", "llama3.3:70b"]
WRITER_TEMP, DRAWS = 0.7, 2


def main(name: str) -> int:
    wf = WORKFLOWS[name]()
    out = REPO / "experiments" / "spec_channel" / "results" / f"{name}_screen.json"

    # S1: two draws per writer arm.
    specs = {}
    for w in WRITERS:
        for d in range(DRAWS):
            r = ask(w, wf.writer_prompt, WRITER_TEMP)
            specs[f"{w}#{d}"] = r
            print(f"spec {w}#{d}: status={r['status']} words={len((r['content'] or '').split())} tokens_out={r['tokens_out']}", flush=True)
    toks = {k: tokens((v["content"] or "").lower()) for k, v in specs.items()}
    keys = list(specs)
    sim = {f"{a} ~ {b}": round(jaccard(toks[a], toks[b]), 3) for i, a in enumerate(keys) for b in keys[i + 1:]}
    for k, v in sim.items():
        print(f"  jaccard {k}: {v}")

    # S2: screen items under S_ref and each writer's draw 0.
    spec_text = {"S_ref": wf.ref_spec, **{w: specs[f"{w}#0"]["content"] for w in WRITERS}}
    calls = []
    for spec_name, text in spec_text.items():
        for ex in EXECUTORS:
            for it in wf.screen:
                r = ask(ex, wf.executor_prompt(text, it))
                calls.append({"spec": spec_name, "executor": ex, "item": it["id"], "stratum": it.get("stratum"), "label": it["label"],
                              "pred": wf.parse(r["content"]), **{k: v for k, v in r.items() if k != "content"}, "raw": (r["content"] or "")[:80]})
        print(f"done spec {spec_name}", flush=True)

    print(f"\n{'spec':24s} {'executor':14s} acc  unparsed  flips_vs_ref  tokens_in_med")
    for spec_name in spec_text:
        for ex in EXECUTORS:
            cs = [c for c in calls if c["spec"] == spec_name and c["executor"] == ex]
            ref = {c["item"]: c["pred"] for c in calls if c["spec"] == "S_ref" and c["executor"] == ex}
            tin = sorted(c["tokens_in"] or 0 for c in cs)
            print(f"{spec_name:24s} {ex:14s} {sum(c['pred'] == c['label'] for c in cs) / len(cs):.2f}  {sum(c['pred'] is None for c in cs):8d}  "
                  f"{sum(c['pred'] != ref[c['item']] for c in cs):12d}  {tin[len(tin) // 2]:13d}")

    out.write_text(json.dumps({"specs": specs, "spec_similarity": sim, "items": [it["id"] for it in wf.screen], "calls": calls}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
