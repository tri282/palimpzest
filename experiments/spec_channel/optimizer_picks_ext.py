"""Optimizer replay on the C2 extension matrix: which writer x executor plan each policy picks (descriptive).

Usage: python -I optimizer_picks_ext.py. Policies and results: REPORT-C2-ext.md III.3, V.3.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_ext as ax  # noqa: E402

EXECS = ["B0", "B1", "B2", "B3", "B4", "B5", "B7", "B11"]  # primary set, analysis_ext.json
W = ax.WRITERS
BARS = (0.95, 0.98)  # cheapest plan within this share of the best estimate
# USD per 1M tokens (input, output), list prices read 2026-10-19 (DeepInfra; Nscale for qwen3:8b); reasoning billed as output.
PRICES = {"gemini-3.5-flash-lite": (0.30, 2.50), "qwen2.5:72b": (0.36, 0.40), "gpt-oss:120b": (0.037, 0.17),
          "llama3.3:70b": (0.10, 0.32), "gemma3:27b": (0.08, 0.16), "qwen3:8b": (0.07, 0.18), "gemma3:12b": (0.05, 0.15),
          "gemma4:26b-a4b": (0.07, 0.34), "gpt-oss:latest": (0.03, 0.14)}
WRITER_MODEL = {"A0": "gemini-3.5-flash-lite", "A1": "qwen2.5:72b", "A2": "gpt-oss:120b", "A3": "llama3.3:70b",
                "A4": "gemma3:27b", "A5": "qwen3:8b"}
PER = 1000  # records per writer draft: the draft is written once and amortized


def load():
    calls = ax.load()
    S = ax.scored(calls, lenient=False)
    model = {b: next(r["model"] for k, r in calls.items() if k[2] == b and not r["error"]) for b in EXECS}
    tok = {}  # (arm, executor) -> mean (tokens_in, tokens_out) per call over successful rows
    for (rep, a, b, item), r in calls.items():
        if b in EXECS and not r["error"]:
            tok.setdefault((a, b), []).append((r["tokens_in"] or 0, r["tokens_out"] or 0))
    tok = {k: np.mean(v, axis=0) for k, v in tok.items()}
    drafts = {}
    for run in ("full-c2", "ext-writer-drafts"):
        for line in (ax.RUNS / run / "specs.jsonl").read_text().splitlines():
            s = json.loads(line)
            if s.get("arm_a") in W and not s.get("error"):
                drafts.setdefault(s["arm_a"], []).append((s.get("tokens_in") or 0, s.get("tokens_out") or 0))
    return S, model, tok, {a: np.mean(v, axis=0) for a, v in drafts.items()}


def main() -> int:
    S, model, tok, drafts = load()
    A = ax.matrix(S, EXECS, own=True)  # [w, b, r]; own-model cells are real plans
    M = A.mean(axis=2)
    ref_acc = np.array([np.mean([ax.acc(S[(ax.REF, b, r)]) for r in ax.REPS]) for b in EXECS])

    def usd(m, t):
        return (t[0] * PRICES[m][0] + t[1] * PRICES[m][1]) / 1e6

    def cost(w, b, arm=None):  # USD per 1k records; arm=REF prices the executor under the generic instruction
        ex = usd(model[EXECS[b]], tok[(arm or W[w], EXECS[b])]) * 1000
        return ex + usd(WRITER_MODEL[W[w]], drafts[W[w]]) * 1000 / PER

    C = np.array([[cost(w, b) for b in range(len(EXECS))] for w in range(len(W))])
    best_q = M.max()

    def name(w, b):
        return f"{W[w]} x {EXECS[b]}"

    def report(label, w, b, bar=None):
        out = f"  {label:42s} {name(w, b):10s} acc {M[w, b]:.3f}  ${C[w, b]:.4f}/1k"
        if bar:
            ok = M[w, b] >= bar * best_q
            cheapest = min(C[i, j] for i in range(len(W)) for j in range(len(EXECS)) if M[i, j] >= bar * best_q)
            out += f"  bar {'met' if ok else 'MISSED'}  overpay {C[w, b] / cheapest:.2f}x"
        print(out)

    def cheapest(est_q, est_c, bar, mask=None):
        ok = [(est_c[i, j], i, j) for i in range(len(W)) for j in range(len(EXECS))
              if (mask is None or mask[i, j]) and est_q[i, j] >= bar * np.nanmax(np.where(mask, est_q, np.nan) if mask is not None else est_q)]
        return min(ok)[1:]

    print("Cost per 1k records (USD), executor x writer:")
    for b in range(len(EXECS)):
        print(f"  {EXECS[b]:4s} {model[EXECS[b]]:16s} " + " ".join(f"{W[w]} {C[w, b]:.4f}" for w in range(len(W)))
              + f"  | S_ref {cost(0, b, ax.REF) - usd(WRITER_MODEL['A0'], drafts['A0']):.4f}")

    print("\nMax quality:")
    report("full search", *np.unravel_index(np.argmax(M), M.shape))
    champ = int(np.argmax(M.mean(axis=1)))
    report(f"Abacus, champion {W[champ]}", champ, int(np.argmax(M[champ])))
    for r_b in range(len(EXECS)):
        report(f"separate, writer scored on {EXECS[r_b]}", int(np.argmax(M[:, r_b])), int(np.argmax(ref_acc)))
    # Abacus with the judge: per replicate, the top-scored writers' drafts; executor picked under each; tie rule chooses.
    judge = {}
    for f in ("p9_scores_pz.json", "p9_scores_ext.json"):
        for rep, cell in json.loads((ax.RESULTS / f).read_text())["replicates"].items():
            for w, v in cell.items():
                if w in W:
                    judge.setdefault(int(rep), {})[w] = v["scores"][0]
    for rule in ("best", "first", "worst"):
        reg = []
        for r in ax.REPS:
            top = max(s for s in judge[r].values() if s is not None)
            plans = {W.index(w): int(np.argmax(A[W.index(w), :, r])) for w in W if judge[r].get(w) == top}
            val = {w: A[w, b, r] for w, b in plans.items()}
            w = {"best": max(val, key=val.get), "worst": min(val, key=val.get), "first": min(val)}[rule]
            reg.append(A[:, :, r].max() - val[w])
        print(f"  {'Abacus, judge, tie ' + rule:42s} regret {np.mean(reg):.3f} (per replicate vs its best plan)")

    for bar in BARS:
        print(f"\nCheapest within {bar:.0%} of the best estimate (bar on true accuracy: {bar * best_q:.3f}):")
        report("full search", *cheapest(M, C, bar), bar=bar)
        mask = np.zeros_like(M, dtype=bool)
        mask[champ] = True
        report(f"Abacus, champion {W[champ]}", *cheapest(M, C, bar, mask), bar=bar)
        # Separate profiles: executor under S_ref (accuracy and tokens), writer via a reference executor.
        est_c = np.array([[cost(w, b, ax.REF) for b in range(len(EXECS))] for w in range(len(W))])
        for r_b in range(len(EXECS)):
            est_q = np.outer(M[:, r_b], ref_acc / ref_acc[r_b])
            report(f"separate, writer scored on {EXECS[r_b]}", *cheapest(est_q, est_c, bar), bar=bar)
    return 0


if __name__ == "__main__":
    sys.exit(main())
