"""Offline comparison of optimizer quality-estimation policies under reference dependence, plus draft features vs executor gain.

Usage: python -I optimizer_policies.py full-c2 (after analyze.py). Design context: REPORT.md.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import A_ARMS, B_ARMS, PRICE_PROXY, REF, RUNS, load  # noqa: E402

KS, DRAWS, SEED = [5, 10, 20, 40, 100, 125], 500, 0  # 100, 125: same calls as P0 with a candidate / held-out ref
OUT = RUNS.parent / "optimizer_policies.json"
FEATURES = {  # keyword flags on draft text; read hits by hand before trusting them
    "model_number": r"model|part number|sku|mpn",
    "capacity_size": r"capacity|\bsize|\bgb\b|\btb\b|storage|watt|dimension",
    "color": r"colou?r",
    "brand": r"brand|manufacturer",
}


def rank(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    r = np.empty(len(x))
    r[x.argsort(kind="stable")] = np.arange(len(x))
    for v in np.unique(x):
        r[x == v] = r[x == v].mean()  # average ranks for ties (binary flags)
    return r


def spearman(x, y) -> float | None:
    rx, ry = rank(x), rank(y)
    return None if rx.std() == 0 or ry.std() == 0 else round(float(np.corrcoef(rx, ry)[0, 1]), 3)


def decide(cells, true_q, true_cost, est_q, est_cost) -> dict:
    """Same decision rules as analyze.baseline: best estimate (pool all), cheapest estimate above 95% of best true (pool factorial)."""
    out = {}
    for name, pool in {"factorial": [c for c in cells if c[0] != REF], "all": cells}.items():
        best, pick = max(pool, key=lambda c: true_q[c]), max(pool, key=lambda c: est_q[c])
        entry = {"true_best": "/".join(best), "optimizer_pick": "/".join(pick), "quality_regret": round(true_q[best] - true_q[pick], 4)}
        bar = 0.95 * true_q[best]
        feas_true, feas_est = [c for c in pool if true_q[c] >= bar], [c for c in pool if est_q[c] >= bar]
        if feas_true and feas_est:
            t, e = min(feas_true, key=lambda c: true_cost[c]), min(feas_est, key=lambda c: est_cost[c])
            entry |= {"optimizer_cheapest": "/".join(e), "pick_meets_bar": bool(true_q[e] >= bar),
                      "cost_ratio_pick_vs_true": round(true_cost[e] / true_cost[t], 3)}
        out[name] = entry
    return out


def summarize(entries: list[dict], calls: int) -> dict:
    a = [e["all"] for e in entries]
    f = [e["factorial"] for e in entries if "cost_ratio_pick_vs_true" in e["factorial"]]
    reg = [e["quality_regret"] for e in a]
    return {"n": len(entries), "profiling_calls": calls,
            "all_best_picked": round(float(np.mean([e["true_best"] == e["optimizer_pick"] for e in a])), 3),
            "all_regret_mean": round(float(np.mean(reg)), 4), "all_regret_p95": round(float(np.percentile(reg, 95)), 4),
            "all_regret_max": round(float(max(reg)), 4),
            "factorial_n_decided": len(f),
            "factorial_meets_bar": round(float(np.mean([e["pick_meets_bar"] for e in f])), 3) if f else None,
            "factorial_cost_ratio_median": round(float(np.median([e["cost_ratio_pick_vs_true"] for e in f])), 3) if f else None,
            "factorial_cost_ratio_max": round(float(max(e["cost_ratio_pick_vs_true"] for e in f)), 3) if f else None}


def analyze_policies(run_id: str) -> dict:
    cfg, calls, specs = load(run_id)
    reps, items = sorted({k[0] for k in calls}), sorted({k[3] for k in calls})
    n = len(items)
    held = sorted({k[2] for k in calls} - set(B_ARMS))
    execs = B_ARMS + held
    cells = [(a, b) for a in A_ARMS + [REF] for b in B_ARMS]  # candidate plans, same order as analyze.py
    grid = [(a, b) for a in A_ARMS + [REF] for b in execs]
    vec = lambda rep, c, f: np.array([float(calls[(rep, *c, it)][f]) for it in items])  # noqa: E731
    C = {rep: {c: vec(rep, c, "correct") for c in grid} for rep in reps}
    TI = {rep: {c: vec(rep, c, "tokens_in") for c in grid} for rep in reps}
    TO = {rep: {c: vec(rep, c, "tokens_out") for c in grid} for rep in reps}
    acc = {rep: {c: C[rep][c].mean() for c in grid} for rep in reps}
    model = {b: next(calls[k]["model"] for k in calls if k[2] == b) for b in B_ARMS}

    def up_cost(rep, a):
        return 0.0 if a == REF else (specs[(rep, a)]["tokens_in"] + specs[(rep, a)]["tokens_out"]) * PRICE_PROXY.get(specs[(rep, a)]["model"], 0)

    def cost(rep, a, b, idx=slice(None)):
        return up_cost(rep, a) + n * (TI[rep][(a, b)][idx].mean() + TO[rep][(a, b)][idx].mean()) * PRICE_PROXY[model[b]]

    def est_cost(rep, a, b, mode):
        if mode == "naive":  # executor profiled under S_ref, spec length ignored (analyze.py)
            return cost(rep, REF, b) + up_cost(rep, a)
        ratio = np.mean([TI[rep][(a, r)].mean() / TI[rep][(REF, r)].mean() for r in held])  # R7 composed, held-out refs
        return up_cost(rep, a) + n * (TI[rep][(REF, b)].mean() * ratio + TO[rep][(REF, b)].mean()) * PRICE_PROXY[model[b]]

    # 1. Reference-based quality estimators for writer plans (a, b); S_ref plans are profiled directly in all of them.
    def p0(q, a, b, r):
        return q[(a, r)] * q[(REF, b)] / q[(REF, r)]

    policies = {f"P0_ref_{r}": (lambda q, a, b, r=r: p0(q, a, b, r), [r]) for r in execs}
    policies |= {
        "P1_heldout_mean": (lambda q, a, b: np.mean([p0(q, a, b, r) for r in held]), held),
        "P2_all_mean": (lambda q, a, b: np.mean([p0(q, a, b, r) for r in execs]), execs),
        "P3_heldout_min": (lambda q, a, b: min(p0(q, a, b, r) for r in held), held),
        "P4_additive": (lambda q, a, b: q[(REF, b)] + np.mean([q[(a, r)] - q[(REF, r)] for r in held]), held),
        "P5_matched": (lambda q, a, b: p0(q, a, b, min(held, key=lambda r: abs(q[(REF, r)] - q[(REF, b)]))), held),
    }
    res, per_rep = {}, {}
    for mode in ("naive", "composed"):
        for name, (f, used) in policies.items():
            entries = []
            for rep in reps:
                q = acc[rep]
                true_q, true_cost = {c: q[c] for c in cells}, {c: cost(rep, *c) for c in cells}
                est_q = {(a, b): q[(a, b)] if a == REF else f(q, a, b) for a, b in cells}
                ec = {(a, b): true_cost[(a, b)] if a == REF else est_cost(rep, a, b, mode) for a, b in cells}
                entries.append(decide(cells, true_q, true_cost, est_q, ec))
            calls_used = n * (len(set(B_ARMS) | set(used)) + len(A_ARMS) * len(used))  # S_ref profiles + writer x ref profiles
            res[f"{name}/{mode}"] = summarize(entries, calls_used)
            per_rep[f"{name}/{mode}"] = dict(zip(reps, entries))

    # 2. P6 joint sampling: every candidate plan run on k records. Scored on the disjoint n - k (truth gets noisy as k grows)
    # and on all n (same truth as P0-P5, which profile on all n; optimistic for P6 since the sample is in it).
    rng = np.random.default_rng(SEED)
    for k in KS:
        entries = {"disjoint": [], "all_n": []}
        for rep in reps:
            true_cost = {c: cost(rep, *c) for c in cells}
            for _ in range(DRAWS):
                perm = rng.permutation(n)
                s, h = perm[:k], perm[k:]
                est_q, ec = {c: C[rep][c][s].mean() for c in cells}, {c: cost(rep, *c, s) for c in cells}
                for truth, idx in (("disjoint", h), ("all_n", slice(None))):
                    entries[truth].append(decide(cells, {c: C[rep][c][idx].mean() for c in cells}, true_cost, est_q, ec))
        for truth, e in entries.items():
            res[f"P6_joint_k{k}/{truth}"] = summarize(e, len(cells) * k)

    # 3. Draft features vs executor gain under that draft, q(a, b) - q(S_ref, b).
    drafts = {}
    for (rep, a), s in sorted(specs.items()):
        text = s["content"]
        drafts[f"rep{rep} {a}"] = {
            "words": s["spec_words"], "rules": len(re.findall(r"(?m)^\s*(?:\d+[.)]|[-*•])\s", text)),
            **{k: int(bool(re.search(p, text, re.I))) for k, p in FEATURES.items()},
            **{f"gain_{b}": round(acc[rep][(a, b)] - acc[rep][(REF, b)], 3) for b in execs},
            "gain_B0_minus_B1": round(acc[rep][(a, "B0")] - acc[rep][(REF, "B0")] - acc[rep][(a, "B1")] + acc[rep][(REF, "B1")], 3)}
    feats = ["words", "rules", *FEATURES]
    targets = [f"gain_{b}" for b in execs] + ["gain_B0_minus_B1"]
    corr = {f: {t: spearman([d[f] for d in drafts.values()], [d[t] for d in drafts.values()]) for t in targets} for f in feats}
    return {"run_id": run_id, "n_items": n, "replicates": len(reps), "held_out": held,
            "summary": res, "per_replicate": per_rep, "drafts": drafts, "draft_spearman": corr}


def check_p0(res: dict) -> None:
    """P0/naive must reproduce analyze.py's independence baseline for every reference executor."""
    ref = next(r for r in json.loads((RUNS.parent / "analysis.json").read_text()) if r["run_id"] == res["run_id"])
    for b_ref, by_rep in ref["independence_baseline"].items():
        mine = res["per_replicate"][f"P0_ref_{b_ref}/naive"]
        for rep, r in by_rep.items():
            for pool in ("factorial", "all"):
                for key in ("optimizer_pick", "quality_regret", "cost_ratio_pick_vs_true"):
                    assert r[pool].get(key) == mine[int(rep)][pool].get(key), (b_ref, rep, pool, key, r[pool].get(key), mine[int(rep)][pool].get(key))


def report(res: dict) -> None:
    print(f"\n=== {res['run_id']} (n={res['n_items']}, reps={res['replicates']}, held-out refs {res['held_out']}) ===")
    cols = ["n", "profiling_calls", "all_best_picked", "all_regret_mean", "all_regret_p95", "all_regret_max",
            "factorial_n_decided", "factorial_meets_bar", "factorial_cost_ratio_median", "factorial_cost_ratio_max"]
    print(f"{'policy':28s} " + " ".join(f"{c[:14]:>14s}" for c in cols))
    for name, s in res["summary"].items():
        print(f"{name:28s} " + " ".join(f"{str(s[c]):>14s}" for c in cols))
    print("\ndrafts:")
    for k, d in res["drafts"].items():
        print(f"  {k:10s} {d}")
    print("\nSpearman (feature x gain, n = drafts, descriptive):")
    for f, row in res["draft_spearman"].items():
        print(f"  {f:14s} {row}")


def main(run_ids: list[str]) -> int:
    assert spearman([1, 2, 3], [3, 2, 1]) == -1.0 and list(rank([0, 1, 1])) == [0.0, 1.5, 1.5]
    results = [analyze_policies(r) for r in run_ids]
    for r in results:
        check_p0(r)
        report(r)
    OUT.write_text(json.dumps(results, indent=1, default=str))
    print(f"\nP0 matches analysis.json; wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
