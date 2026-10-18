"""Abacus-style champion writer for C2: PZ's default LLM judge scores each saved rule draft without any executor.

Usage: python -I p9_scores.py [--style pz|raw] [--samples 3] [--dry-run]
Outputs results/runs/p9-scores[-pz]/calls.jsonl and results/p9_scores[_pz].json. Design and results: REPORT.md.
Exit codes: 0 done, 2 transient stop (re-run resumes).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import ask  # noqa: E402
from probe_api import REPO  # noqa: E402
from pydantic import Field, create_model  # noqa: E402
from runner import MAX_CONSECUTIVE_FAILURES, append, now, read_jsonl  # noqa: E402

from palimpzest.constants import Cardinality, Model, PromptStrategy  # noqa: E402
from palimpzest.core.elements.records import DataRecord  # noqa: E402
from palimpzest.prompts import MAP_VALIDATOR_PROMPT, PromptFactory  # noqa: E402
from palimpzest.query.generators.generators import get_json_from_answer  # noqa: E402

JUDGE = "gpt-oss:120b"  # stand-in for PZ's default o4-mini
SOURCE_RUN = "full-c2"
RESULTS = REPO / "experiments" / "spec_channel" / "results"
EXAMPLES_SPLIT = "\n\nOutput the rules only.\n\n"  # EM_WRITER: instruction, this line, then the examples
EXAMPLES_DESC = "Labeled example pairs of a Walmart product record (Record A) and an Amazon product record (Record B), each with whether they refer to the same product."


def output_message(draft: str) -> str:
    # Same block _default_map_score_fn appends after the input messages; the writer's single output field is "rules".
    return f"OUTPUT:\n--------\n{json.dumps({'rules': draft}, indent=2)}\n\nEVALUATION: "


def judge_text(writer_prompt: str, draft: str) -> str:
    # Raw style (first run): the writer prompt stands in for PZ's input message, joined with the output block.
    return f"{writer_prompt}\n\n{output_message(draft)}"


def pz_input_messages(writer_prompt: str) -> list[str]:
    """The writer as a PZ sem_map: input field `examples`, output field `rules` whose description is the writer's instruction.
    Built as _default_map_score_fn does (MAP / ONE_TO_ONE factory with the validator's default model, system messages
    dropped). The <<cache-boundary>> marker is kept: the validator, unlike the generators, does not strip it."""
    instruction, examples = writer_prompt.split(EXAMPLES_SPLIT)
    instruction = instruction.replace("Below are labeled example pairs.", "The input field `examples` holds labeled example pairs.")
    assert "labeled example pairs" in instruction and examples.startswith("Record A:")
    inp = create_model("WriterInput", examples=(str, Field(description=EXAMPLES_DESC)))
    out = create_model("WriterOutput", examples=(str, Field(description=EXAMPLES_DESC)), rules=(str, Field(description=instruction)))
    record = DataRecord(inp(examples=examples), source_indices=0)
    factory = PromptFactory(PromptStrategy.MAP, Model.o4_MINI, Cardinality.ONE_TO_ONE)
    messages = factory.create_messages(record, ["rules"], output_schema=out, project_cols=["examples"])
    return [m["content"] for m in messages if m["role"] != "system"]


def parse_score(text: str | None) -> float | None:
    """Mean of the judge's field scores, as _default_map_score_fn; None where PZ's try/except would yield None."""
    try:
        # Model only selects llama-specific trimming in get_json_from_answer; o4-mini = the default (no trimming).
        d = get_json_from_answer(text or "", Model.o4_MINI, Cardinality.ONE_TO_ONE)
        return sum(d.values()) / len(d)
    except Exception:
        return None


def check_parse() -> None:
    assert parse_score('{"rules": 1.0}') == 1.0
    assert parse_score('The rules are fine.\n{"rules": 0.0}\nDone.') == 0.0
    assert parse_score('{"a": 1.0, "b": 0.0}') == 0.5
    assert parse_score("no json here") is None
    assert parse_score(None) is None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", choices=["pz", "raw"], default="pz", help="judge input message: PZ sem_map messages or raw writer prompt")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    check_parse()

    src = RESULTS / "runs" / SOURCE_RUN
    writer_prompt = json.loads((src / "config.json").read_text())["writer_prompt"]
    drafts = {(s["replicate"], s["arm_a"]): s for s in read_jsonl(src / "specs.jsonl") if not s["error"]}
    jobs = [(rep, a, k) for (rep, a) in sorted(drafts) for k in range(args.samples)]
    random.Random(args.seed).shuffle(jobs)

    suffix = "-pz" if args.style == "pz" else ""
    run_dir = RESULTS / "runs" / f"p9-scores{suffix}"
    pz_msgs = pz_input_messages(writer_prompt)

    def judge_input(draft: str) -> str | list[str]:
        return pz_msgs + [output_message(draft)] if args.style == "pz" else judge_text(writer_prompt, draft)

    calls_path = run_dir / "calls.jsonl"
    done = {(r["replicate"], r["arm_a"], r["sample"]) for r in read_jsonl(calls_path) if not r["error"]}
    todo = [j for j in jobs if j not in done]
    print(f"judge: {len(drafts)} drafts x {args.samples} samples = {len(jobs)} planned, {len(todo)} to run", flush=True)
    if args.dry_run:
        rep, a, _ = jobs[0]
        print(f"--- system ({len(MAP_VALIDATOR_PROMPT)} chars), first 200:\n{MAP_VALIDATOR_PROMPT[:200]}")
        msgs = judge_input(drafts[(rep, a)]["content"])
        for i, m in enumerate([msgs] if isinstance(msgs, str) else msgs):
            print(f"--- user message {i} ({len(m)} chars):\n{m[:1200]}\n[...]\n{m[-500:]}")
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for rep, a, k in todo:
        d = drafts[(rep, a)]
        r = ask(JUDGE, judge_input(d["content"]), temperature=None, system=MAP_VALIDATOR_PROMPT)
        score = parse_score(r["content"])
        append(calls_path, {"run_id": run_dir.name, "style": args.style, "source_run": SOURCE_RUN, "replicate": rep, "arm_a": a, "sample": k,
                            "spec_hash": d["spec_hash"], "spec_words": d["spec_words"], "score": score,
                            "unparsed": r["content"] is not None and score is None, "timestamp": now(), **r})
        print(f"rep={rep} {a} sample={k} score={score}", flush=True)
        failures = failures + 1 if r["error"] else 0
        if failures >= MAX_CONSECUTIVE_FAILURES:
            print(f"{failures} consecutive failed calls (last: {r['error']}); stopping. Re-run to resume.", flush=True)
            return 2

    summarize(calls_path, sorted({rep for rep, _ in drafts}), RESULTS / f"p9_scores{suffix.replace('-', '_')}.json")
    return 0


def summarize(calls_path: Path, replicates: list[int], out_path: Path) -> None:
    rows = [r for r in read_jsonl(calls_path) if not r["error"]]
    out = {"judge": JUDGE, "source_run": SOURCE_RUN, "replicates": {}}
    print(f"\n{'rep':>3} {'A0 scores':>16} {'A1 scores':>16} {'A0 w':>5} {'A1 w':>5}  champion")
    for rep in replicates:
        cell = {}
        for a in ("A0", "A1"):
            rs = [r for r in rows if r["replicate"] == rep and r["arm_a"] == a]
            vals = [r["score"] for r in rs if r["score"] is not None]
            cell[a] = {"scores": [r["score"] for r in sorted(rs, key=lambda r: r["sample"])],
                       "mean": sum(vals) / len(vals) if vals else None, "spec_words": rs[0]["spec_words"] if rs else None}
        m0, m1 = cell["A0"]["mean"], cell["A1"]["mean"]
        # Ties reported as ties (Abacus would take the first operator, an order artefact).
        champ = None if m0 is None or m1 is None else "tie" if m0 == m1 else "A0" if m0 > m1 else "A1"
        out["replicates"][rep] = {**cell, "champion": champ}
        print(f"{rep:>3} {str(cell['A0']['scores']):>16} {str(cell['A1']['scores']):>16} {cell['A0']['spec_words']!s:>5} "
              f"{cell['A1']['spec_words']!s:>5}  {champ}")
    out_path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(main())
