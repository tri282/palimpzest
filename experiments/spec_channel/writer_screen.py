"""Writer screen for the C2 extension: 2 drafts per new writer, checked mechanically before their run drafts are made.

Usage: python -I writer_screen.py [--writers A2,A3,A4,A5] [--host purdue]. Appends to results/writer_screen/drafts.jsonl
(resumable), writes verdicts.json. Rule: PREREG-C2-ext.md II.1. Design: REPORT-C2-ext.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import ask  # noqa: E402
from probe_api import REPO  # noqa: E402
from prompts import em_record  # noqa: E402
from runner import ALL_WRITERS, NEW_WRITERS, WRITER_TEMP, append, now, read_jsonl  # noqa: E402
from workflows import C2_FULL_SAMPLE, WORKFLOWS  # noqa: E402

N_DRAFTS = 2
MAX_WORDS = 375  # 1.5x the 250-word limit; the saved qwen2.5 drafts reach 330
MIN_CRITERIA = 3
# Matching criteria the writer prompt asks for; a criterion counts if any of its keywords appears (lowercased).
CRITERIA = {
    "model number": ("model number", "modelno", "model no", "model #", "part number"),
    "brand": ("brand", "manufacturer"),
    "variant": ("color", "colour", "size", "capacity", "variant"),
    "bundle or accessory": ("bundle", "accessor", "kit", "pack"),
    "missing or noisy field": ("missing", "noisy", "empty", "absent", "incomplete"),
}
OUT_DIR = REPO / "experiments" / "spec_channel" / "results" / "writer_screen"


def check(draft: str | None, example_records: list[str]) -> dict:
    text = (draft or "").lower()
    words = len(text.split())
    hits = [c for c, kws in CRITERIA.items() if any(k in text for k in kws)]
    copied = [r for r in example_records if r.lower() in text]
    ok = bool(text.strip()) and words <= MAX_WORDS and not copied and len(hits) >= MIN_CRITERIA
    return {"words": words, "criteria": hits, "copied_examples": len(copied), "pass": ok}


def check_check() -> None:
    good = "Match only if the model number and brand agree; ignore color; bundles differ; missing price is fine."
    assert check(good, ["title: foo; brand: bar"])["pass"]
    assert not check("Products match if similar.", [])["pass"]  # too few criteria
    assert not check(good + " title: foo; brand: bar", ["title: foo; brand: bar"])["pass"]  # copies an example
    assert not check(" ".join(["brand model number color"] * 100), [])["pass"]  # too long
    assert not check(None, [])["pass"]


def main() -> int:
    check_check()
    ids = sys.argv[sys.argv.index("--writers") + 1].split(",") if "--writers" in sys.argv else list(NEW_WRITERS)
    host = sys.argv[sys.argv.index("--host") + 1] if "--host" in sys.argv else "purdue"
    wf = WORKFLOWS["c2_full"]()
    examples = json.loads(C2_FULL_SAMPLE.read_text())["writer_examples"]
    records = [em_record(p[side]) for p in examples for side in ("left", "right")]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "drafts.jsonl"

    done = {(r["arm_a"], r["draft"]) for r in read_jsonl(out) if not r["error"]}
    for a in ids:
        for k in range(N_DRAFTS):
            if (a, k) in done:
                continue
            r = ask(ALL_WRITERS[a], wf.writer_prompt, WRITER_TEMP, host=host)
            append(out, {"arm_a": a, "draft": k, "timestamp": now(), **check(r["content"], records), **r})
            print(f"{a} {ALL_WRITERS[a]} draft {k}: {'error' if r['error'] else 'ok'}", flush=True)

    rows = {(r["arm_a"], r["draft"]): r for r in read_jsonl(out) if not r["error"]}
    verdicts = {}
    for a in ids:
        rs = [rows[(a, k)] for k in range(N_DRAFTS) if (a, k) in rows]
        verdicts[a] = {"model": ALL_WRITERS[a], "drafts": len(rs), "pass": len(rs) == N_DRAFTS and all(r["pass"] for r in rs),
                       "words": [r["words"] for r in rs], "criteria": [len(r["criteria"]) for r in rs],
                       "copied": [r["copied_examples"] for r in rs]}
        v = verdicts[a]
        print(f"{a} {v['model']:16s} words {v['words']} criteria {v['criteria']} copied {v['copied']}  {'PASS' if v['pass'] else 'FAIL'}")
    (OUT_DIR / "verdicts.json").write_text(json.dumps(verdicts, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
