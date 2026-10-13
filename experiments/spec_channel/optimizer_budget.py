"""Equal-budget comparison of optimizer quality estimates on sampled profiles (P0, P8, Abacus-style P9, P6, P7), plus per-stratum quality.

Usage: python -I optimizer_budget.py full-c2. Design context: REPORT.md.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import A_ARMS, B_ARMS, REF, REPO, RUNS, interaction, load  # noqa: E402
from optimizer_policies import PRICES_USD, decide, summarize, targeted  # noqa: E402

BUDGETS, DRAWS, SEED = [150, 300, 600, 900], 500, 0  # profiling calls per decision
# Llama 3.3 70B, USD per 1M tokens (input, output): DeepInfra Turbo FP8 (2026-10-12, non-Turbo page 404), Together (2026-10-13).
LLAMA_PRICES = {"turbo": (0.10, 0.32), "together": (1.04, 1.04)}
LLAMA_MULT = [1, 2, 3, 4, 6, 8, 10]  # multiples of the Turbo price, for the writer-optional break-even sweep
TEST_CSV = REPO / "testdata" / "fm_data_tasks" / "datasets" / "entity_matching" / "structured" / "Walmart-Amazon" / "test.csv"
STRATA = ("pos", "hard_neg", "rand_neg")
OUT = RUNS.parent / "optimizer_budget.json"


def natural_weights() -> dict:
    # Test-split mix. Hard = top-Jaccard non-matches: the pilot's 30 plus the full run's 56 (c2_setup.stratified).
    with TEST_CSV.open() as f:
        labels = [int(r["label"]) for r in csv.DictReader(f)]
    samples = [json.loads((RUNS.parent / name).read_text()) for name in ("c2_sample.json", "c2_full_sample.json")]
    n_hard = sum(p["stratum"] == "hard_neg" for s in samples for p in s.get("pilot", []) + s.get("items", []))
    n, n_pos = len(labels), sum(labels)
    return {"pos": n_pos / n, "hard_neg": n_hard / n, "rand_neg": (n - n_pos - n_hard) / n}


def plan_cells(name: str, r: str) -> list[tuple]:
    """Cells a policy profiles (its call count per record); P9's champion writer is unknown up front, A0 stands in."""
    s_ref = [(REF, b) for b in B_ARMS]
    if name == "P0" or name == "P7":
        return sorted(set(s_ref + [(REF, r)] + [(a, r) for a in A_ARMS]))
    if name.startswith("P8"):
        return sorted(set(s_ref + [(w, x) for w in A_ARMS for x in B_ARMS + [r]]))
    if name == "P9":
        return sorted(set(s_ref + [(a, r) for a in A_ARMS] + [("A0", b) for b in B_ARMS]))
    return [(a, b) for a in A_ARMS + [REF] for b in B_ARMS]  # P6


def estimate(name: str, r: str, q: dict) -> dict:
    """Writer-plan quality estimates from sampled means q; S_ref plans are profiled directly by every policy."""
    est = {}
    for a in A_ARMS:
        other = [w for w in A_ARMS if w != a]
        champ = max(A_ARMS, key=lambda w: q[(w, r)])  # P9: best upstream on the sample, as Abacus feeds downstream
        for b in B_ARMS:
            est[(a, b)] = {
                "P0": lambda: q[(a, r)] * q[(REF, b)] / q[(REF, r)],
                "P8_all": lambda: q[(a, r)] * np.mean([q[(w, b)] for w in A_ARMS]) / np.mean([q[(w, r)] for w in A_ARMS]),
                "P8_loo": lambda: q[(a, r)] * np.mean([q[(w, b)] for w in other]) / np.mean([q[(w, r)] for w in other]),
                "P9": lambda: q[(champ, b)] * q[(a, r)] / q[(champ, r)],
                "P6": lambda: q[(a, b)],
            }[name]()
    return est


def check_estimate() -> None:
    q = {("A0", "B0"): .88, ("A1", "B0"): .84, ("A0", "B1"): .87, ("A1", "B1"): .86, (REF, "B0"): .79, (REF, "B1"): .91}
    p9 = estimate("P9", "B1", q)  # champion on B1 is A0: its own plans exact, A1 scaled by the B1 ratio
    assert np.allclose([p9[("A0", "B0")], p9[("A0", "B1")], p9[("A1", "B0")]], [.88, .87, .88 * .86 / .87])
    p8 = estimate("P8_loo", "B1", q)  # leave-out with two writers: the other writer's profile
    assert abs(p8[("A0", "B0")] - .87 * .84 / .86) < 1e-12


def analyze_budget(run_id: str) -> dict:
    cfg, calls, specs = load(run_id)
    reps, items = sorted({k[0] for k in calls}), sorted({k[3] for k in calls})
    n = len(items)
    held = sorted({k[2] for k in calls} - set(B_ARMS))
    execs = B_ARMS + held
    cells = [(a, b) for a in A_ARMS + [REF] for b in B_ARMS]
    grid = [(a, b) for a in A_ARMS + [REF] for b in execs]
    vec = lambda rep, c, f: np.array([float(calls[(rep, *c, it)][f] or 0) for it in items])  # noqa: E731
    C = {rep: {c: vec(rep, c, "correct") for c in grid} for rep in reps}
    TI = {rep: {c: vec(rep, c, "tokens_in") for c in grid} for rep in reps}
    TO = {rep: {c: vec(rep, c, "tokens_out") for c in grid} for rep in reps}
    model = {b: next(calls[k]["model"] for k in calls if k[2] == b) for b in B_ARMS}

    def prices(llama_in_out):
        p = dict(PRICES_USD) | {"llama3.3:70b": llama_in_out}
        return {m: (i / 1e6, o / 1e6) for m, (i, o) in p.items()}

    def up_cost(rep, a, P):
        if a == REF:
            return 0.0
        pi, po = P[specs[(rep, a)]["model"]]
        return specs[(rep, a)]["tokens_in"] * pi + specs[(rep, a)]["tokens_out"] * po

    def run_cost(rep, a, b, P, tin, tout):  # whole run on n records at mean per-call tokens
        pi, po = P[model[b]]
        return up_cost(rep, a, P) + n * (tin * pi + tout * po)

    # 1. Equal budget: all policies draw from one record permutation per draw; prefix means via cumulative sums.
    rng = np.random.default_rng(SEED)
    policies = [(p, r) for p in ("P0", "P8_all", "P8_loo", "P9", "P7") for r in execs] + [("P6", None)]
    keys = [(llama, bar) for llama in LLAMA_PRICES for bar in ("true", "est")]
    entries = {(p, r, B, llama, bar): [] for p, r in policies for B in BUDGETS for llama, bar in keys}
    spent = {(p, r, B): [] for p, r in policies for B in BUDGETS}
    for rep in reps:
        Ps = {llama: prices(v) for llama, v in LLAMA_PRICES.items()}
        true_q = {c: C[rep][c].mean() for c in cells}
        true_cost = {llama: {c: run_cost(rep, *c, P, TI[rep][c].mean(), TO[rep][c].mean()) for c in cells} for llama, P in Ps.items()}
        for _ in range(DRAWS):
            perm = rng.permutation(n)
            cs = {f: {c: np.cumsum(M[rep][c][perm]) for c in grid} for f, M in (("q", C), ("ti", TI), ("to", TO))}
            mean = lambda f, c, k: cs[f][c][k - 1] / k  # noqa: E731
            for p, r in policies:
                base = "P0" if p == "P7" else p
                ref = r or B_ARMS[1]
                prof = plan_cells(p, ref)
                for B in BUDGETS:
                    k = min(n, B // (len(prof) + (2 if p == "P7" else 0)))  # P7 reserves ~2 cells for verification
                    q = {c: mean("q", c, k) for c in grid}
                    est_q = {c: q[c] for c in cells if c[0] == REF} | estimate(base, ref, q)
                    measured = set(prof)
                    if base == "P9":
                        champ = max(A_ARMS, key=lambda w: q[(w, ref)])
                        measured = {c for c in prof if c[0] == REF or c[1] == ref} | {(champ, b) for b in B_ARMS}
                    n_calls = len(prof) * k
                    for llama, bar in keys:
                        P = Ps[llama]

                        def ec(a, b):  # measured on the sample if profiled, else composed via the writer-on-reference token ratio
                            if (a, b) in measured:
                                return run_cost(rep, a, b, P, mean("ti", (a, b), k), mean("to", (a, b), k))
                            ratio = mean("ti", (a, ref), k) / mean("ti", (REF, ref), k)
                            return run_cost(rep, a, b, P, mean("ti", (REF, b), k) * ratio, mean("to", (REF, b), k))

                        e_q, e_c = dict(est_q), {c: ec(*c) for c in cells}
                        if p == "P7":
                            e_q, e_c, verify = targeted(cells, e_q, e_c, lambda c: (q[c], run_cost(rep, *c, P, mean("ti", c, k), mean("to", c, k))))
                            if (llama, bar) == keys[0]:
                                n_calls += len(set(verify) - set(prof)) * k
                        entries[(p, r, B, llama, bar)].append(decide(cells, true_q, true_cost[llama], e_q, e_c, bar_from=bar))
                    spent[(p, r, B)].append(n_calls)
    summary = {f"{p}{'_ref_' + r if r else ''}/{llama}/bar_{bar}/B{B}": summarize(e, round(float(np.mean(spent[(p, r, B)]))))
               for (p, r, B, llama, bar), e in entries.items()}

    # 2. Writer optional, true values: cheapest plan above the oracle bar as the Llama price rises.
    sweep = {}
    for mult in LLAMA_MULT:
        P = prices(tuple(mult * x for x in LLAMA_PRICES["turbo"]))
        rows = []
        for rep in reps:
            tq = {c: C[rep][c].mean() for c in cells}
            tc = {c: run_cost(rep, *c, P, TI[rep][c].mean(), TO[rep][c].mean()) for c in cells}
            bar = 0.95 * max(tq.values())
            rows.append(("/".join(min((c for c in cells if tq[c] >= bar), key=tc.get)), tc[("A0", "B0")] / tc[(REF, "B1")]))
        sweep[f"x{mult}"] = {"cheapest_counts": {p: sum(x == p for x, _ in rows) for p in sorted({x for x, _ in rows})},
                             "A0B0_over_SrefB1_cost_median": round(float(np.median([v for _, v in rows])), 3)}

    # 3. Per stratum (exact, all records): accuracy, yes-rate, natural-mix reweighting, executor gap and interaction.
    strat = np.array([next(calls[k]["stratum"] for k in calls if k[3] == it) for it in items])
    pred = {rep: {c: np.array([calls[(rep, *c, it)]["prediction"] == 1 for it in items]) for c in grid} for rep in reps}
    w = natural_weights()
    by_stratum = {}
    for c in grid:
        row = {s: round(float(np.mean([C[rep][c][strat == s].mean() for rep in reps])), 3) for s in STRATA}
        row["yes_rate"] = round(float(np.mean([pred[rep][c].mean() for rep in reps])), 3)
        row["sample_mix"] = round(float(np.mean([C[rep][c].mean() for rep in reps])), 3)
        row["natural_mix"] = round(sum(w[s] * row[s] for s in STRATA), 3)
        by_stratum["/".join(c)] = row
    contrasts = {}
    for s in (*STRATA, "sample_mix", "natural_mix"):
        L = {c: by_stratum["/".join(c)][s] for c in cells}
        contrasts[s] = {f"gap_B1_minus_B0_{a}": round(L[(a, "B1")] - L[(a, "B0")], 3) for a in A_ARMS + [REF]}
        contrasts[s]["interaction"] = round(-interaction(L), 3)  # on error rates, as pre-registered (analyze.py)
    return {"run_id": run_id, "n_items": n, "replicates": len(reps), "held_out": held, "draws": DRAWS, "budgets": BUDGETS,
            "llama_prices": LLAMA_PRICES, "natural_weights": {k: round(v, 4) for k, v in w.items()},
            "strata_counts": {s: int((strat == s).sum()) for s in STRATA},
            "summary": summary, "llama_sweep": sweep, "by_stratum": by_stratum, "stratum_contrasts": contrasts}


def check_exact(res: dict) -> None:
    """At k = n the sampled estimates equal the exact ones: same quality picks as optimizer_policies.py (P0, P8)."""
    ref = next(r for r in json.loads((RUNS.parent / "optimizer_policies.json").read_text()) if r["run_id"] == res["run_id"])
    exact = analyze_exact_picks(res["run_id"])
    for key, picks in exact.items():
        for rep, entry in ref["per_replicate"][f"{key}/usd/composed"].items():
            for pool in ("all", "factorial"):
                assert picks[int(rep)][pool] == entry[pool]["optimizer_pick"], (key, rep, pool)


def analyze_exact_picks(run_id: str) -> dict:
    cfg, calls, specs = load(run_id)
    reps, items = sorted({k[0] for k in calls}), sorted({k[3] for k in calls})
    execs = B_ARMS + sorted({k[2] for k in calls} - set(B_ARMS))
    cells = [(a, b) for a in A_ARMS + [REF] for b in B_ARMS]
    out = {}
    for p, name in (("P0", "P0_ref_{}"), ("P8_all", "P8_all_ref_{}"), ("P8_loo", "P8_loo_ref_{}")):
        for r in execs:
            by_rep = {}
            for rep in reps:
                q = {(a, b): np.mean([calls[(rep, a, b, it)]["correct"] for it in items]) for a in A_ARMS + [REF] for b in execs}
                est = {c: q[c] for c in cells if c[0] == REF} | estimate(p, r, q)
                by_rep[rep] = {"all": "/".join(max(cells, key=est.get)), "factorial": "/".join(max((c for c in cells if c[0] != REF), key=est.get))}
            out[name.format(r)] = by_rep
    return out


def report(res: dict) -> None:
    print(f"\n=== {res['run_id']} (n={res['n_items']}, reps={res['replicates']}, draws={res['draws']}) ===")
    cols = ["profiling_calls", "all_best_picked", "all_regret_mean", "all_regret_max", "factorial_meets_bar", "factorial_cost_ratio_median",
            "all_cheapest_meets_bar", "all_cheapest_cost_ratio_median"]
    print(f"{'policy':38s} " + " ".join(f"{c[:14]:>14s}" for c in cols))
    for name, s in res["summary"].items():
        print(f"{name:38s} " + " ".join(f"{str(s[c]):>14s}" for c in cols))
    print("\nwriter optional, Llama price sweep (x Turbo):")
    for k, v in res["llama_sweep"].items():
        print(f"  {k:5s} {v}")
    print(f"\nnatural weights {res['natural_weights']}, strata {res['strata_counts']}")
    for c, row in res["by_stratum"].items():
        print(f"  {c:10s} {row}")
    for s, row in res["stratum_contrasts"].items():
        print(f"  {s:12s} {row}")


def main(run_ids: list[str]) -> int:
    check_estimate()
    results = [analyze_budget(r) for r in run_ids]
    for r in results:
        check_exact(r)
        report(r)
    OUT.write_text(json.dumps(results, indent=1, default=str))
    print(f"\nexact picks match optimizer_policies.json; wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
