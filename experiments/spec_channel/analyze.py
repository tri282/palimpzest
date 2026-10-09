"""Offline analysis of Specification channel runs: interaction, estimability, independence baseline and regret, cost model, cache sensitivity, sizing.

Usage: python -I analyze.py pilot-c2 pilot-c1. Design and results: REPORT.md.
"""
from __future__ import annotations

import json
import math
import re
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lexical_baseline import TARBALL, load as load_c1, rules as c1_rules  # noqa: E402
from em_lexical_baseline import EM, f1, jaccard, model_tokens, pairs, table, tokens  # noqa: E402
from probe_api import REPO  # noqa: E402

RUNS = REPO / "experiments" / "spec_channel" / "results" / "runs"
A_ARMS, B_ARMS, REF = ["A0", "A1"], ["B0", "B1"], "S_ref"
B_REFS = ["B1", "B0"]  # reference executors for the writer profile; B1 was the pilot's only one
CACHE_K = [1.0, 0.5, 0.25, 0.1]  # spec-token price relative to uncached input (prefix-cache sensitivity)
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


def t_test(x) -> dict:
    """One-sample two-sided t-test against 0. p from Simpson integration of the t pdf (no scipy here)."""
    x = np.asarray(x, dtype=float)
    n, m, sd = len(x), float(x.mean()), float(x.std(ddof=1))
    if sd == 0:
        return {"mean": round(m, 4), "t": None, "df": n - 1, "p": None}  # identical replicates: test undefined
    t, df = m / (sd / math.sqrt(n)), n - 1
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    steps, h = 2000, abs(t) / 2000
    ys = [c * (1 + (i * h) ** 2 / df) ** (-(df + 1) / 2) for i in range(steps + 1)]
    area = h / 3 * (ys[0] + ys[-1] + 4 * sum(ys[1:-1:2]) + 2 * sum(ys[2:-1:2]))
    return {"mean": round(m, 4), "t": round(t, 3), "df": df, "p": round(max(1 - 2 * area, 0.0), 4)}


def boot2(stat, reps: list, n: int, rng) -> list[float]:
    """Two-level bootstrap CI95: resample replicates (spec draws), then records within each; stat(rep, idx) -> float."""
    vals = [np.mean([stat(r, rng.integers(0, n, n)) for r in rng.choice(reps, len(reps))]) for _ in range(N_BOOT)]
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def lexical_c2(item_ids: list[str], labels: list[int]) -> dict:
    """em_lexical_baseline rules (train-tuned Jaccard, model-number overlap) scored on this run's records."""
    d = EM / "structured" / "Walmart-Amazon"
    A, B = table(d / "tableA.csv"), table(d / "tableB.csv")
    jac = lambda l, r: jaccard(tokens(A[l]), tokens(B[r]))  # noqa: E731
    train = pairs(d / "train.csv")
    scores, train_labels = [jac(l, r) for l, r, _ in train], [y for *_, y in train]
    best_t = max((t / 100 for t in range(5, 95)), key=lambda t: f1([s >= t for s in scores], train_labels))
    ids = [it.split("-")[1:3] for it in item_ids]  # item_id = "<split>-<ltable_id>-<rtable_id>"
    out = {}
    for name, preds in [(f"jaccard_t{best_t:.2f}", [int(jac(l, r) >= best_t) for l, r in ids]),
                        ("model_token_overlap", [int(bool(model_tokens(A[l]) & model_tokens(B[r]))) for l, r in ids])]:
        out[name] = {"acc": round(float(np.mean([p == y for p, y in zip(preds, labels)])), 3), "f1": round(f1(preds, labels), 3)}
    return out


def analyze(run_id: str) -> dict:
    cfg, calls, specs = load(run_id)
    reps = sorted({k[0] for k in calls})
    items = sorted({k[3] for k in calls})
    cells = [(a, b) for a in A_ARMS + [REF] for b in B_ARMS]  # candidate plans
    held_out = sorted({k[2] for k in calls} - set(B_ARMS))  # held-out reference executors (B2, B3), profiling only
    refs = B_REFS + held_out
    all_cells = cells + [(a, b) for b in held_out for a in A_ARMS + [REF]]
    # correct[rep][cell] -> vector over items (same item order everywhere, so cells are paired by record).
    correct = {rep: {c: np.array([calls[(rep, *c, it)]["correct"] for it in items], dtype=float) for c in all_cells} for rep in reps}
    acc = {rep: {c: correct[rep][c].mean() for c in all_cells} for rep in reps}
    tok = {rep: {c: np.mean([calls[(rep, *c, it)]["tokens_in"] + calls[(rep, *c, it)]["tokens_out"] for it in items]) for c in all_cells}
           for rep in reps}
    tok_in = {rep: {c: np.mean([calls[(rep, *c, it)]["tokens_in"] for it in items]) for c in all_cells} for rep in reps}
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

    # 3. Independence baseline: profile executors under S_ref, writers with each B_ref, compose, pick; compare to truth.
    def up_cost(rep, a):
        return 0.0 if a == REF else (specs[(rep, a)]["tokens_in"] + specs[(rep, a)]["tokens_out"]) * PRICE_PROXY.get(specs[(rep, a)]["model"], 0)

    def price(rep, a, b, k=1.0):
        # The spec is the prompt's shared prefix; its tokens over the S_ref one-liner are priced at k (cache discount).
        spec_tok = 0.0 if a == REF else max(tok_in[rep][(a, b)] - tok_in[rep][(REF, b)], 0.0)
        return up_cost(rep, a) + len(items) * (tok[rep][(a, b)] - (1 - k) * spec_tok) * PRICE_PROXY[model[b]]

    def baseline(b_ref, k=1.0):
        regret = {}
        for rep in reps:
            true_q = {c: acc[rep][c] for c in cells}
            true_cost = {c: price(rep, *c, k) for c in cells}
            est_q, est_cost = {}, {}
            for a, b in cells:
                if a == REF:
                    est_q[(a, b)], est_cost[(a, b)] = true_q[(a, b)], true_cost[(a, b)]  # no upstream: profiled directly
                    continue
                q_up, q_b, q_bref = acc[rep][(a, b_ref)], acc[rep][(REF, b)], acc[rep][(REF, b_ref)]
                est_q[(a, b)] = q_up * q_b / q_bref  # multiplicative composition
                # Naive independent cost: executor profiled under S_ref (spec length ignored) + writer cost.
                est_cost[(a, b)] = up_cost(rep, a) + len(items) * tok[rep][(REF, b)] * PRICE_PROXY[model[b]]
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
        return regret

    out["independence_baseline"] = {b_ref: baseline(b_ref) for b_ref in refs}
    keep = ("true_cheapest", "optimizer_cheapest", "cost_ratio_pick_vs_true")
    out["cache_sensitivity"] = {b_ref: {k: {rep: {f: r["factorial"].get(f) for f in keep} for rep, r in baseline(b_ref, k).items()}
                                        for k in CACHE_K} for b_ref in refs}
    out["spec_share_of_downstream_cost"] = {
        f"rep{rep} {a}/{b}": round(1 - price(rep, a, b, 0.0) / price(rep, a, b), 3) for rep in reps for a in A_ARMS for b in B_ARMS}

    # 4. Cost model (R7): downstream input tokens per call, naive (S_ref profile) vs composed (scale by B_ref spec ratio).
    cost_rows = {}
    for rep in reps:
        for a in A_ARMS:
            for b in B_ARMS:
                measured = tok_in[rep][(a, b)]
                naive = tok_in[rep][(REF, b)]
                composed = {r: tok_in[rep][(REF, b)] * tok_in[rep][(a, r)] / tok_in[rep][(REF, r)] for r in refs}
                cost_rows[f"rep{rep} {a}/{b}"] = {"measured": round(measured, 1), "naive_err": round(naive / measured - 1, 3),
                                                  "composed_err": {r: round(v / measured - 1, 3) for r, v in composed.items()}}
    out["cost_model"] = {"rows": cost_rows,
                         "mean_abs_err_naive": round(float(np.mean([abs(v["naive_err"]) for v in cost_rows.values()])), 3),
                         "mean_abs_err_composed": {r: round(float(np.mean([abs(v["composed_err"][r]) for v in cost_rows.values()])), 3)
                                                   for r in refs},
                         "spec_words": {f"rep{k[0]} {k[1]}": s["spec_words"] for k, s in sorted(specs.items())}}

    # 5. Latency (non-switch calls) per cell.
    out["latency_median_s"] = {"/".join(c): round(float(np.median([calls[(rep, *c, it)]["latency_s"] for rep in reps for it in items
                                                                  if not calls[(rep, *c, it)]["model_switch"]])), 3) for c in cells}

    # 6. Workload sizing (plan Section VI). delta = optimizer's estimated margin between its top two factorial plans.
    margins = []
    for rep in reps:
        est = sorted((v["est_q"] for k, v in out["independence_baseline"][B_REFS[0]][rep]["estimates"].items() if not k.startswith(REF)),
                     reverse=True)
        margins.append(est[0] - est[1])
    delta = max(float(np.mean(margins)), 1 / len(items))
    sigma_d = float(np.std(d, ddof=1))
    sd_rep = out["interaction"]["sd_across_replicates"]
    out["sizing"] = {"sigma_d_per_record": round(sigma_d, 3), "delta_margin": round(delta, 4),
                     "records_needed": math.ceil((Z * sigma_d / delta) ** 2),
                     "replicates_needed": math.ceil((Z * sd_rep / delta) ** 2) if sd_rep else None}

    # 7. Replicate-level tests (spec draw = unit): H1 interaction, H2 estimability (gap under S_ref minus gap under each writer).
    signs = {("A0", "B0"): 1, ("A0", "B1"): -1, ("A1", "B0"): -1, ("A1", "B1"): 1}
    gap = {rep: {a: acc[rep][(a, "B1")] - acc[rep][(a, "B0")] for a in A_ARMS + [REF]} for rep in reps}
    out["replicate_tests"] = {
        "H1_interaction_acc": t_test(list(per_rep.values()))
        | {"ci95_two_level": boot2(lambda r, idx: sum(s * (1 - correct[r][c][idx]).mean() for c, s in signs.items()), reps, len(items), rng)},
        "H2_estimability": {a: t_test([gap[rep][REF] - gap[rep][a] for rep in reps]) for a in A_ARMS},
    }

    # 8. F1 alongside accuracy (binary workflows only) and the lexical baseline on the same records.
    if cfg["workflow"].startswith("c2"):
        labels = [calls[(reps[0], REF, "B0", it)]["label"] for it in items]
        pred = {rep: {c: np.array([calls[(rep, *c, it)]["prediction"] or 0 for it in items]) for c in cells} for rep in reps}
        lab = np.array(labels)
        f1s = {rep: {c: f1(list(pred[rep][c]), labels) for c in cells} for rep in reps}
        out["f1"] = {
            "per_cell": {rep: {"/".join(c): round(v, 3) for c, v in f1s[rep].items()} for rep in reps},
            "interaction_per_replicate": {rep: round(interaction({c: 1 - f1s[rep][c] for c in cells}), 4) for rep in reps},
            "interaction_ci95_two_level": boot2(lambda r, idx: sum(s * (1 - f1(list(pred[r][c][idx]), list(lab[idx]))) for c, s in signs.items()),
                                               reps, len(items), rng),
        }
        out["lexical_baseline"] = lexical_c2(items, labels)

    # 9. C1 redesign checks (c1-redesign.md II.3): P1 best factorial cell vs the area-code x phone-format rule on the
    # same records; P2 spec lines that state the phone-format convention (flagged by keyword, then read by hand).
    if cfg["workflow"].startswith("c1"):
        with tarfile.open(TARBALL) as tar:
            split = {s: load_c1(tar, s) for s in ("train", "valid", "test")}
        rule = c1_rules(split["train"])["area_code_x_phone_format"]
        labels = [calls[(reps[0], REF, "B0", it)]["label"] for it in items]
        rule_ok = np.array([rule(split[it.split("-")[0]][int(it.split("-")[1])]) == y for it, y in zip(items, labels)], dtype=float)
        fact = [c for c in cells if c[0] != REF]
        best = max(fact, key=lambda c: np.mean([acc[r][c] for r in reps]))
        diff = np.mean([correct[r][best] for r in reps], axis=0) - rule_ok
        boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(N_BOOT)]
        out["p1_vs_rule"] = {
            "rule_acc": round(float(rule_ok.mean()), 3), "best_cell": "/".join(best),
            "best_acc_per_rep": {r: round(acc[r][best], 3) for r in reps},
            "reps_beating_rule": f"{sum(acc[r][best] > rule_ok.mean() for r in reps)}/{len(reps)}",
            "pooled_diff": round(float(diff.mean()), 4),
            "ci95_records": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)]}
        pat = re.compile(r"slash|dash|hyphen|\d{3}\s*/|phone[^.\n]{0,40}format|format[^.\n]{0,40}phone", re.I)
        out["p2_phone_format"] = {f"rep{r} {a}": [ln.strip() for ln in s["content"].splitlines() if pat.search(ln)]
                                  for (r, a), s in sorted(specs.items())}
    return out


def report(res: dict) -> None:
    i = res["interaction"]
    print(f"\n=== {res['run_id']} ({res['workflow']}, n={res['n_items']}, reps={res['replicates']}) ===")
    print(f"interaction additive per rep {i['per_replicate_additive']}  log {i['per_replicate_log']}")
    print(f"  pooled {i['pooled_additive']} CI95(records) {i['ci95_records']} sd_reps {i['sd_across_replicates']} sign {i['sign_agreement']}")
    print("executor gap B1-B0 by spec:", {r: {a: v["B1_minus_B0"] for a, v in s.items()} for r, s in res["executor_ranking"].items()})
    for b_ref, by_rep in res["independence_baseline"].items():
        for rep, r in by_rep.items():
            for pool in ("factorial", "all"):
                print(f"  B_ref={b_ref} rep{rep} {pool:9s} {r[pool]}")
    for b_ref, by_k in res["cache_sensitivity"].items():
        print(f"cache k -> cost ratio pick/true (B_ref={b_ref}):", {k: [v["cost_ratio_pick_vs_true"] for v in r.values()] for k, r in by_k.items()})
    c = res["cost_model"]
    print(f"cost model |err| naive {c['mean_abs_err_naive']} composed {c['mean_abs_err_composed']}; spec words {c['spec_words']}")
    print("latency median:", res["latency_median_s"])
    print("sizing:", res["sizing"])
    print("replicate tests:", res["replicate_tests"])
    if "f1" in res:
        print("F1 per cell:", res["f1"]["per_cell"])
        print(f"F1 interaction per rep {res['f1']['interaction_per_replicate']} CI95(two-level) {res['f1']['interaction_ci95_two_level']}")
        print("lexical baseline (same records):", res["lexical_baseline"])
    if "p1_vs_rule" in res:
        print("P1 vs rule:", res["p1_vs_rule"])
        hits = {k: v for k, v in res["p2_phone_format"].items() if v}
        print(f"P2 specs flagged for phone format: {len(hits)}/{len(res['p2_phone_format'])} (read each by hand)")
        for k, lines in hits.items():
            print(f"  {k}: {lines[:3]}")


def main(run_ids: list[str]) -> int:
    assert abs(t_test([1.0, 2.0, 3.0])["p"] - 0.0742) < 1e-3  # t=3.464, df=2
    results = [analyze(r) for r in run_ids]
    for r in results:
        report(r)
    out = REPO / "experiments" / "spec_channel" / "results" / "analysis.json"
    out.write_text(json.dumps(results, indent=1, default=str))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
