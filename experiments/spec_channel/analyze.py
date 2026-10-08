"""Offline analysis of Specification channel pilot runs: interaction, independence baseline and regret, cost model, workload sizing.

Usage: python -I analyze.py pilot-c2 pilot-c1. Design and results: REPORT.md.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_api import REPO  # noqa: E402

RUNS = REPO / "experiments" / "spec_channel" / "results" / "runs"
A_ARMS, B_ARMS, REF, B_REF = ["A0", "A1"], ["B0", "B1"], "S_ref", "B1"
# ponytail: relative price proxy = parameter count (B), since Purdue models have no list price. Swap for real $/Mtok when known.
PRICE_PROXY = {"gemma3:12b": 12, "llama3.3:70b": 70, "qwen2.5:72b": 72}
N_BOOT, SEED = 2000, 0
Z = 1.96 + 0.84  # alpha 0.05 two-sided, power 0.8


def load(run_id: str):
    d = RUNS / run_id
    cfg = json.loads((d / "config.json").read_text())
    calls = {}
    for line in (d / "calls.jsonl").read_text().splitlines():
        r = json.loads(line)
        k = (r["replicate"], r["arm_a"], r["arm_b"], r["item_id"])
        if not r["error"] or k not in calls:
            calls[k] = r
    specs = {(s["replicate"], s["arm_a"]): s for s in map(json.loads, (d / "specs.jsonl").read_text().splitlines()) if not s["error"]}
    return cfg, calls, specs


def interaction(L: dict) -> float:
    return L[("A0", "B0")] - L[("A0", "B1")] - L[("A1", "B0")] + L[("A1", "B1")]


def analyze(run_id: str) -> dict:
    cfg, calls, specs = load(run_id)
    reps = sorted({k[0] for k in calls})
    items = sorted({k[3] for k in calls})
    cells = [(a, b) for a in A_ARMS + [REF] for b in B_ARMS]
    # correct[rep][cell] -> vector over items (same item order everywhere, so cells are paired by record).
    correct = {rep: {c: np.array([calls[(rep, *c, it)]["correct"] for it in items], dtype=float) for c in cells} for rep in reps}
    acc = {rep: {c: correct[rep][c].mean() for c in cells} for rep in reps}
    tok = {rep: {c: np.mean([calls[(rep, *c, it)]["tokens_in"] + calls[(rep, *c, it)]["tokens_out"] for it in items]) for c in cells} for rep in reps}
    tok_in = {rep: {c: np.mean([calls[(rep, *c, it)]["tokens_in"] for it in items]) for c in cells} for rep in reps}
    model = {b: next(calls[k]["model"] for k in calls if k[2] == b) for b in B_ARMS}
    rng = np.random.default_rng(SEED)
    out = {"run_id": run_id, "workflow": cfg["workflow"], "n_items": len(items), "replicates": len(reps)}

    # 1. Interaction contrast on loss (1 - acc): per replicate, pooled, record-bootstrap CI, multiplicative form.
    per_rep = {rep: interaction({c: 1 - acc[rep][c] for c in cells}) for rep in reps}
    mult = {}
    for rep in reps:
        L = {c: max(1 - acc[rep][c], 0.5 / len(items)) for c in cells}  # floor avoids log(0)
        mult[rep] = math.log(L[("A0", "B0")]) - math.log(L[("A0", "B1")]) - math.log(L[("A1", "B0")]) + math.log(L[("A1", "B1")])
    # Per-record contrast d_r, averaged over replicates; bootstrap resamples records (paired across cells).
    d = np.mean([sum(s * (1 - correct[rep][c]) for s, c in [(1, ("A0", "B0")), (-1, ("A0", "B1")), (-1, ("A1", "B0")), (1, ("A1", "B1"))])
                 for rep in reps], axis=0)
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
    pooled = float(d.mean())
    out["interaction"] = {
        "per_replicate_additive": {r: round(v, 4) for r, v in per_rep.items()},
        "per_replicate_log": {r: round(v, 3) for r, v in mult.items()},
        "pooled_additive": round(pooled, 4),
        "ci95_records": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
        "sd_across_replicates": round(float(np.std(list(per_rep.values()), ddof=1)), 4),
        "sign_agreement": f"{sum(np.sign(v) == np.sign(pooled) for v in per_rep.values())}/{len(reps)}",
    }

    # 2. Context dependence: executor quality under the neutral spec vs under each writer's spec.
    out["executor_ranking"] = {
        rep: {a: {b: round(acc[rep][(a, b)], 3) for b in B_ARMS} | {"B1_minus_B0": round(acc[rep][(a, "B1")] - acc[rep][(a, "B0")], 3)}
              for a in A_ARMS + [REF]} for rep in reps}

    # 3. Independence baseline: profile executors under S_ref, writers with B_ref, compose, pick; compare to truth.
    def price(rep, a, b):
        up = 0.0 if a == REF else (specs[(rep, a)]["tokens_in"] + specs[(rep, a)]["tokens_out"]) * PRICE_PROXY.get(specs[(rep, a)]["model"], 0)
        return up + len(items) * tok[rep][(a, b)] * PRICE_PROXY[model[b]]

    regret = {}
    for rep in reps:
        true_q = {c: acc[rep][c] for c in cells}
        true_cost = {c: price(rep, *c) for c in cells}
        est_q, est_cost = {}, {}
        for a, b in cells:
            if a == REF:
                est_q[(a, b)], est_cost[(a, b)] = true_q[(a, b)], true_cost[(a, b)]  # no upstream: profiled directly
                continue
            q_up, q_b, q_bref = true_q[(a, B_REF)], true_q[(REF, b)], true_q[(REF, B_REF)]
            est_q[(a, b)] = q_up * q_b / q_bref  # multiplicative composition
            # Naive independent cost: executor profiled under S_ref (spec length ignored) + writer cost.
            est_cost[(a, b)] = price(rep, a, b) - len(items) * (tok[rep][(a, b)] - tok[rep][(REF, b)]) * PRICE_PROXY[model[b]]
        pools = {"factorial": [c for c in cells if c[0] != REF], "all": cells}
        regret[rep] = {}
        for name, pool in pools.items():
            best = max(pool, key=lambda c: true_q[c])
            pick = max(pool, key=lambda c: est_q[c])
            entry = {"true_best": "/".join(best), "optimizer_pick": "/".join(pick),
                     "quality_regret": round(true_q[best] - true_q[pick], 4)}
            # Min-cost-above-bar policy at bar = 95% of the best true quality in the pool.
            bar = 0.95 * true_q[best]
            feas_true = [c for c in pool if true_q[c] >= bar]
            feas_est = [c for c in pool if est_q[c] >= bar]
            if feas_true and feas_est:
                t, e = min(feas_true, key=lambda c: true_cost[c]), min(feas_est, key=lambda c: est_cost[c])
                entry |= {"bar": round(bar, 3), "true_cheapest": "/".join(t), "optimizer_cheapest": "/".join(e),
                          "pick_meets_bar": bool(true_q[e] >= bar), "cost_ratio_pick_vs_true": round(true_cost[e] / true_cost[t], 3)}
            regret[rep][name] = entry
        regret[rep]["estimates"] = {"/".join(c): {"est_q": round(est_q[c], 3), "true_q": round(true_q[c], 3)} for c in cells}
    out["independence_baseline"] = regret

    # 4. Cost model (R7): downstream input tokens per call, naive (S_ref profile) vs composed (scale by B_ref spec ratio).
    cost_rows = {}
    for rep in reps:
        for a in A_ARMS:
            for b in B_ARMS:
                measured = tok_in[rep][(a, b)]
                naive = tok_in[rep][(REF, b)]
                composed = tok_in[rep][(REF, b)] * tok_in[rep][(a, B_REF)] / tok_in[rep][(REF, B_REF)]
                cost_rows[f"rep{rep} {a}/{b}"] = {"measured": round(measured, 1), "naive_err": round(naive / measured - 1, 3),
                                                  "composed_err": round(composed / measured - 1, 3)}
    out["cost_model"] = {"rows": cost_rows,
                         "mean_abs_err_naive": round(float(np.mean([abs(v["naive_err"]) for v in cost_rows.values()])), 3),
                         "mean_abs_err_composed": round(float(np.mean([abs(v["composed_err"]) for v in cost_rows.values()])), 3),
                         "spec_words": {f"rep{k[0]} {k[1]}": s["spec_words"] for k, s in sorted(specs.items())}}

    # 5. Latency (non-switch calls) per cell.
    out["latency_median_s"] = {"/".join(c): round(float(np.median([calls[(rep, *c, it)]["latency_s"] for rep in reps for it in items
                                                                  if not calls[(rep, *c, it)]["model_switch"]])), 3) for c in cells}

    # 6. Workload sizing (plan Section VI). delta = optimizer's estimated margin between its top two factorial plans.
    margins = []
    for rep in reps:
        est = sorted((v["est_q"] for k, v in regret[rep]["estimates"].items() if not k.startswith(REF)), reverse=True)
        margins.append(est[0] - est[1])
    delta = max(float(np.mean(margins)), 1 / len(items))
    sigma_d = float(np.std(d, ddof=1))
    sd_rep = out["interaction"]["sd_across_replicates"]
    out["sizing"] = {"sigma_d_per_record": round(sigma_d, 3), "delta_margin": round(delta, 4),
                     "records_needed": math.ceil((Z * sigma_d / delta) ** 2),
                     "replicates_needed": math.ceil((Z * sd_rep / delta) ** 2) if sd_rep else None}
    return out


def report(res: dict) -> None:
    i = res["interaction"]
    print(f"\n=== {res['run_id']} ({res['workflow']}, n={res['n_items']}, reps={res['replicates']}) ===")
    print(f"interaction additive per rep {i['per_replicate_additive']}  log {i['per_replicate_log']}")
    print(f"  pooled {i['pooled_additive']} CI95(records) {i['ci95_records']} sd_reps {i['sd_across_replicates']} sign {i['sign_agreement']}")
    print("executor gap B1-B0 by spec:", {r: {a: v["B1_minus_B0"] for a, v in s.items()} for r, s in res["executor_ranking"].items()})
    for rep, r in res["independence_baseline"].items():
        for pool in ("factorial", "all"):
            print(f"  rep{rep} {pool:9s} {r[pool]}")
    c = res["cost_model"]
    print(f"cost model |err| naive {c['mean_abs_err_naive']} composed {c['mean_abs_err_composed']}; spec words {c['spec_words']}")
    print("latency median:", res["latency_median_s"])
    print("sizing:", res["sizing"])


def main(run_ids: list[str]) -> int:
    results = [analyze(r) for r in run_ids]
    for r in results:
        report(r)
    out = REPO / "experiments" / "spec_channel" / "results" / "analysis.json"
    out.write_text(json.dumps(results, indent=1, default=str))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
