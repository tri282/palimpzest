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
from analyze import A_ARMS, B_ARMS, CACHE_K, PRICE_PROXY, REF, RUNS, load  # noqa: E402

KS, DRAWS, SEED = [5, 10, 20, 40, 100, 125], 500, 0  # 100, 125: same calls as P0 with a candidate / held-out ref
# USD per 1M tokens (input, output), list prices checked 2026-10-12: DeepInfra (Llama = 3.3-70B-Instruct-Turbo FP8), Gemini API Standard.
PRICES_USD = {"gemma3:12b": (0.05, 0.15), "llama3.3:70b": (0.10, 0.32), "qwen2.5:72b": (0.36, 0.40), "gemini-3.5-flash-lite": (0.30, 2.50)}
COSTINGS = [("proxy", "naive"), ("proxy", "composed"), ("usd", "composed")]  # (price model, cost estimate)
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


def decide(cells, true_q, true_cost, est_q, est_cost, bar_from: str = "true") -> dict:
    """Same decision rules as analyze.baseline: best estimate (pool all), cheapest estimate above 95% of best (pool factorial).

    bar_from="true" puts the optimizer's bar at 95% of the true best (oracle); "est" at 95% of its own estimated best.
    Overpay and pick_meets_bar are always judged against the true bar.
    """
    out = {}
    for name, pool in {"factorial": [c for c in cells if c[0] != REF], "all": cells}.items():
        best, pick = max(pool, key=lambda c: true_q[c]), max(pool, key=lambda c: est_q[c])
        entry = {"true_best": "/".join(best), "optimizer_pick": "/".join(pick), "quality_regret": round(true_q[best] - true_q[pick], 4)}
        bar = 0.95 * true_q[best]
        est_bar = bar if bar_from == "true" else 0.95 * est_q[pick]
        feas_true, feas_est = [c for c in pool if true_q[c] >= bar], [c for c in pool if est_q[c] >= est_bar]
        if feas_true and feas_est:
            t, e = min(feas_true, key=lambda c: true_cost[c]), min(feas_est, key=lambda c: est_cost[c])
            entry |= {"true_cheapest": "/".join(t), "optimizer_cheapest": "/".join(e), "pick_meets_bar": bool(true_q[e] >= bar),
                      "cost_ratio_pick_vs_true": round(true_cost[e] / true_cost[t], 3)}
        out[name] = entry
    return out


def targeted(cells, est_q: dict, est_cost: dict, sample) -> tuple[dict, dict, list]:
    """P7: re-estimate by joint sampling only the writer plans that could change a decision; sample(cell) -> (quality, cost).

    Verified: the isolated writer-required pick itself (overrated case), writer plans cheaper than it (underrated case), and
    writer plans estimated above the best S_ref plan, which is measured directly.
    """
    writer = [c for c in cells if c[0] != REF]
    best_w = max(est_q[c] for c in writer)
    pick_req = min((c for c in writer if est_q[c] >= 0.95 * best_w), key=lambda c: est_cost[c])
    best_measured = max(est_q[c] for c in cells if c[0] == REF)
    verify = [c for c in writer if c == pick_req or est_cost[c] < est_cost[pick_req] or est_q[c] > best_measured]
    q, cost = dict(est_q), dict(est_cost)
    for c in verify:
        q[c], cost[c] = sample(c)
    return q, cost, verify


def check_targeted() -> None:
    """Synthetic cases for targeted(): underrated cheap plan, overrated writer plan above S_ref, overrated writer-required pick,
    accurate estimates."""
    cells = [(a, b) for a in A_ARMS + [REF] for b in B_ARMS]
    true_q = {("A0", "B0"): .88, ("A1", "B0"): .84, ("A0", "B1"): .87, ("A1", "B1"): .86, (REF, "B0"): .79, (REF, "B1"): .91}
    true_cost = {("A0", "B0"): 1.0, ("A1", "B0"): 1.1, ("A0", "B1"): 5.6, ("A1", "B1"): 5.7, (REF, "B0"): .9, (REF, "B1"): 5.0}

    def run(est_q, tq=true_q):
        before = decide(cells, tq, true_cost, est_q, true_cost)
        q, cost, verify = targeted(cells, est_q, true_cost, lambda c: (tq[c], true_cost[c]))
        return before, decide(cells, tq, true_cost, q, cost), verify

    # Llama-like reference: cheap Gemma writer plans estimated below the bar -> overpay fixed.
    before, after, verify = run(true_q | {("A0", "B0"): .75, ("A1", "B0"): .74})
    assert before["factorial"]["cost_ratio_pick_vs_true"] > 5 and after["factorial"]["cost_ratio_pick_vs_true"] == 1.0
    assert {("A0", "B0"), ("A1", "B0")} <= set(verify)
    # Gemma-like reference: Llama writer plans estimated above S_ref + Llama -> optional regret fixed.
    before, after, _ = run(true_q | {("A0", "B1"): .95, ("A1", "B1"): .93})
    assert before["all"]["quality_regret"] > 0 and after["all"]["quality_regret"] == 0 and after["all"]["optimizer_pick"] == f"{REF}/B1"
    # Gemma-like reference overrates the cheapest writer plan, which is truly below the bar -> pick verified, next one taken.
    tq = true_q | {("A0", "B0"): .80}
    before, after, verify = run(tq | {("A0", "B0"): .90}, tq)
    assert not before["factorial"]["pick_meets_bar"] and after["factorial"]["pick_meets_bar"] and verify == [("A0", "B0")]
    # Accurate estimates, cheapest writer plan already picked: only the pick is sampled.
    _, _, verify = run(dict(true_q))
    assert verify == [("A0", "B0")]


def summarize(entries: list[dict], calls: int) -> dict:
    a = [e["all"] for e in entries]
    f = [e["factorial"] for e in entries if "cost_ratio_pick_vs_true" in e["factorial"]]
    ac = [e for e in a if "cost_ratio_pick_vs_true" in e]  # writer optional, cheapest-above-bar policy
    reg = [e["quality_regret"] for e in a]
    return {"n": len(entries), "profiling_calls": calls,
            "all_cheapest_meets_bar": round(float(np.mean([e["pick_meets_bar"] for e in ac])), 3) if ac else None,
            "all_cheapest_cost_ratio_median": round(float(np.median([e["cost_ratio_pick_vs_true"] for e in ac])), 3) if ac else None,
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

    def unit(m, pm):  # (input, output) price per token
        if pm == "proxy":
            return PRICE_PROXY.get(m, 0), PRICE_PROXY.get(m, 0)
        return PRICES_USD[m][0] / 1e6, PRICES_USD[m][1] / 1e6

    def up_cost(rep, a, pm="proxy"):
        if a == REF:
            return 0.0
        pi, po = unit(specs[(rep, a)]["model"], pm)
        return specs[(rep, a)]["tokens_in"] * pi + specs[(rep, a)]["tokens_out"] * po

    def cost(rep, a, b, idx=slice(None), pm="proxy", k=1.0):
        # Rules tokens (over the S_ref one-liner) are the shared prompt prefix; priced at k of input (cache what-if).
        pi, po = unit(model[b], pm)
        tin = TI[rep][(a, b)][idx].mean()
        spec_tok = 0.0 if a == REF else max(tin - TI[rep][(REF, b)].mean(), 0.0)
        return up_cost(rep, a, pm) + n * ((tin - (1 - k) * spec_tok) * pi + TO[rep][(a, b)][idx].mean() * po)

    def est_cost(rep, a, b, mode, pm):
        if mode == "naive":  # executor profiled under S_ref, spec length ignored (analyze.py)
            return cost(rep, REF, b, pm=pm) + up_cost(rep, a, pm)
        pi, po = unit(model[b], pm)
        ratio = np.mean([TI[rep][(a, r)].mean() / TI[rep][(REF, r)].mean() for r in held])  # R7 composed, held-out refs
        return up_cost(rep, a, pm) + n * (TI[rep][(REF, b)].mean() * ratio * pi + TO[rep][(REF, b)].mean() * po)

    # 1. Quality estimators for writer plans (a, b); S_ref plans are profiled directly in all of them.
    # P0-P5 profile the executor under S_ref; P8 under the writers' rules (design-doc estimate: executor averaged over upstream outputs).
    def p0(q, a, b, r):
        return q[(a, r)] * q[(REF, b)] / q[(REF, r)]

    def m(q, x, excl=None):  # executor x averaged over writers' rules, optionally leaving out writer excl
        return np.mean([q[(w, x)] for w in A_ARMS if w != excl])

    s_ref_calls = lambda used: n * (len(set(B_ARMS) | set(used)) + len(A_ARMS) * len(used))  # noqa: E731
    w_calls = lambda r: n * (len(B_ARMS) + len(A_ARMS) * len(set(B_ARMS) | {r}))  # noqa: E731
    policies = {f"P0_ref_{r}": (lambda q, a, b, r=r: p0(q, a, b, r), s_ref_calls([r])) for r in execs}
    policies |= {
        "P1_heldout_mean": (lambda q, a, b: np.mean([p0(q, a, b, r) for r in held]), s_ref_calls(held)),
        "P2_all_mean": (lambda q, a, b: np.mean([p0(q, a, b, r) for r in execs]), s_ref_calls(execs)),
        "P3_heldout_min": (lambda q, a, b: min(p0(q, a, b, r) for r in held), s_ref_calls(held)),
        "P4_additive": (lambda q, a, b: q[(REF, b)] + np.mean([q[(a, r)] - q[(REF, r)] for r in held]), s_ref_calls(held)),
        "P5_matched": (lambda q, a, b: p0(q, a, b, min(held, key=lambda r: abs(q[(REF, r)] - q[(REF, b)]))), s_ref_calls(held)),
    }
    for r in execs:
        policies[f"P8_all_ref_{r}"] = (lambda q, a, b, r=r: q[(a, r)] * m(q, b) / m(q, r), w_calls(r))
        policies[f"P8_loo_ref_{r}"] = (lambda q, a, b, r=r: q[(a, r)] * m(q, b, a) / m(q, r, a), w_calls(r))
    res, per_rep = {}, {}
    for pm, mode in COSTINGS:
        for name, (f, calls_used) in policies.items():
            entries = []
            for rep in reps:
                q = acc[rep]
                true_q, true_cost = {c: q[c] for c in cells}, {c: cost(rep, *c, pm=pm) for c in cells}
                est_q = {(a, b): q[(a, b)] if a == REF else f(q, a, b) for a, b in cells}
                ec = {(a, b): true_cost[(a, b)] if a == REF else est_cost(rep, a, b, mode, pm) for a, b in cells}
                entries.append(decide(cells, true_q, true_cost, est_q, ec))
            res[f"{name}/{pm}/{mode}"] = summarize(entries, calls_used)
            per_rep[f"{name}/{pm}/{mode}"] = dict(zip(reps, entries))
    executor_avg = {b: {"S_ref": round(float(np.mean([acc[r][(REF, b)] for r in reps])), 3),
                        "writers_avg": round(float(np.mean([m(acc[r], b) for r in reps])), 3)} for b in execs}

    # 1b. Writer optional, cheapest-above-bar on true values: which plan is cheapest under each price model and cache k.
    optional_cheapest = {}
    for pm in ("proxy", "usd"):
        for k in CACHE_K:
            rows = []
            for rep in reps:
                tq, tc = {c: acc[rep][c] for c in cells}, {c: cost(rep, *c, pm=pm, k=k) for c in cells}
                bar = 0.95 * max(tq.values())
                rows.append({"cheapest": "/".join(min((c for c in cells if tq[c] >= bar), key=tc.get)),
                             "A0B0_clears_bar": bool(tq[("A0", "B0")] >= bar),
                             "A0B0_over_SrefB1_cost": round(tc[("A0", "B0")] / tc[(REF, "B1")], 3)})
            optional_cheapest[f"{pm}/k{k}"] = {
                "cheapest_counts": {p: sum(r["cheapest"] == p for r in rows) for p in sorted({r["cheapest"] for r in rows})},
                "A0B0_clears_bar": sum(r["A0B0_clears_bar"] for r in rows),
                "A0B0_over_SrefB1_cost_median": round(float(np.median([r["A0B0_over_SrefB1_cost"] for r in rows])), 3)}

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
            "summary": res, "per_replicate": per_rep, "executor_avg": executor_avg, "optional_cheapest": optional_cheapest,
            "drafts": drafts, "draft_spearman": corr}


def check_p0(res: dict) -> None:
    """P0 (proxy prices, naive cost) must reproduce analyze.py's independence baseline for every reference executor."""
    ref = next(r for r in json.loads((RUNS.parent / "analysis.json").read_text()) if r["run_id"] == res["run_id"])
    for b_ref, by_rep in ref["independence_baseline"].items():
        mine = res["per_replicate"][f"P0_ref_{b_ref}/proxy/naive"]
        for rep, r in by_rep.items():
            for pool in ("factorial", "all"):
                for key in ("optimizer_pick", "quality_regret", "cost_ratio_pick_vs_true"):
                    assert r[pool].get(key) == mine[int(rep)][pool].get(key), (b_ref, rep, pool, key, r[pool].get(key), mine[int(rep)][pool].get(key))


def report(res: dict) -> None:
    print(f"\n=== {res['run_id']} (n={res['n_items']}, reps={res['replicates']}, held-out refs {res['held_out']}) ===")
    cols = ["n", "profiling_calls", "all_cheapest_meets_bar", "all_cheapest_cost_ratio_median", "all_best_picked", "all_regret_mean", "all_regret_p95", "all_regret_max",
            "factorial_n_decided", "factorial_meets_bar", "factorial_cost_ratio_median", "factorial_cost_ratio_max"]
    print(f"{'policy':34s} " + " ".join(f"{c[:14]:>14s}" for c in cols))
    for name, s in res["summary"].items():
        print(f"{name:34s} " + " ".join(f"{str(s[c]):>14s}" for c in cols))
    print("\nexecutor accuracy, under S_ref vs averaged over writers' rules:", res["executor_avg"])
    print("writer optional, true cheapest above bar:")
    for k, v in res["optional_cheapest"].items():
        print(f"  {k:12s} {v}")
    print("\ndrafts:")
    for k, d in res["drafts"].items():
        print(f"  {k:10s} {d}")
    print("\nSpearman (feature x gain, n = drafts, descriptive):")
    for f, row in res["draft_spearman"].items():
        print(f"  {f:14s} {row}")


def main(run_ids: list[str]) -> int:
    assert spearman([1, 2, 3], [3, 2, 1]) == -1.0 and list(rank([0, 1, 1])) == [0.0, 1.5, 1.5]
    check_targeted()
    results = [analyze_policies(r) for r in run_ids]
    for r in results:
        check_p0(r)
        report(r)
    OUT.write_text(json.dumps(results, indent=1, default=str))
    print(f"\nP0 matches analysis.json; wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
