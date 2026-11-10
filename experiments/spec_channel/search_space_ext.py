"""Search-space reduction replay on the C2 extension matrix: pruned and sequential searches vs full search (descriptive).

Usage: python -I search_space_ext.py [--check] [--add B8] [--out search_space_ext.json]. Plans and prices as in
optimizer_picks_ext.py (REPORT-C2-ext.md III.3, V.3).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_ext as ax  # noqa: E402
import optimizer_picks_ext as opx  # noqa: E402

GOALS = ("acc", 0.95, 0.98)  # max accuracy, or cheapest within this share of the best estimate
NS, SEEDS = (25, 50, 100, 250), 200
EPS, KS = (0.0, 0.01, 0.02, 0.03, 0.05), (1, 2, 3, 4)
BATCHES, DELTAS = (25, 50), (0.05, 0.1, 0.2)
SIZE_B = {"B0": 12, "B1": 70, "B2": 120, "B3": 27, "B4": 26, "B5": 8, "B7": 20, "B8": 14, "B11": 72}  # params, billions
NEAR = ax.SESOI["d2"]
HALVES = ([r for r in ax.REPS if r % 2 == 0], [r for r in ax.REPS if r % 2 == 1])


def tiers(execs: list[str]) -> list[list[int]]:
    t = {"small": [], "mid": [], "large": []}
    for i, b in enumerate(execs):
        t["small" if SIZE_B[b] <= 14 else "mid" if SIZE_B[b] <= 30 else "large"].append(i)
    return [v for v in t.values() if v]


def keep_eps(acc, cost, eps: float) -> list[int]:
    """Indices not eps-dominated: no other point at most as costly and at least eps more accurate."""
    n = len(acc)
    return [i for i in range(n) if not any(
        j != i and cost[j] <= cost[i] and acc[j] - eps >= acc[i] and (acc[j] - eps > acc[i] or cost[j] < cost[i])
        for j in range(n))]


def pick(est, C, mask, goal):
    idx = np.argwhere(mask)
    vals = est[mask]
    if goal == "acc":
        return tuple(idx[int(np.argmax(vals))])
    ok = idx[vals >= goal * vals.max()]
    return tuple(min(ok, key=lambda wb: C[tuple(wb)]))


def score(T, C, wb, goal) -> dict:
    """T: held-out true accuracy [w, b]."""
    best = T.max()
    if goal == "acc":
        reg = best - T[wb]
        return {"regret": reg, "near": reg <= NEAR + 1e-12}
    ok = T >= goal * best
    return {"met": bool(T[wb] >= goal * best), "overpay": C[wb] / C[ok].min()}


def static_masks(E, Eref, C, Cref, tier_idx, names):
    """(strategy, knob) -> (kept cells mask, extra measured cells mask). E: estimates [w, b]; Eref: S_ref [b]."""
    W, B = E.shape
    out = {("F", ""): (np.ones((W, B), bool), np.zeros((W, B), bool))}
    for ref in range(B):
        col = np.zeros((W, B), bool)
        col[:, ref] = True
        for eps in EPS:
            ke, kw = keep_eps(Eref, Cref, eps), keep_eps(E[:, ref], C[:, ref], eps)
            m = np.zeros((W, B), bool)
            m[np.ix_(kw, ke)] = True
            out[("P1", f"eps={eps} ref={names[ref]}")] = (m, col)
        for k in KS:
            ke, kw = np.argsort(-Eref, kind="stable")[:k], np.argsort(-E[:, ref], kind="stable")[:k]
            m = np.zeros((W, B), bool)
            m[np.ix_(kw, ke)] = True
            out[("P2", f"k={k} ref={names[ref]}")] = (m, col)
    m = np.zeros((W, B), bool)
    m[:, [min(t, key=lambda b: Cref[b]) for t in tier_idx]] = True
    out[("P4", "")] = (m, np.zeros((W, B), bool))
    return out


def sequential(Xp, C, goal, m, delta, rng):
    """Xp: per-pair score [w, b, p] averaged over the pick replicates. Returns (pick, cell-pairs evaluated)."""
    W, B, P = Xp.shape
    order, active, seen, used = rng.permutation(P), np.ones((W, B), bool), 0, 0
    z = NormalDist().inv_cdf(1 - delta)
    while seen < P:
        used += int(active.sum()) * min(m, P - seen)
        seen = min(seen + m, P)
        x = Xp[:, :, order[:seen]]
        mu = x.mean(axis=2)
        lead = pick(mu, C, active, "acc")
        target = x[lead] if goal == "acc" else goal * x[lead]
        d = target[None, None, :] - x
        lb = d.mean(axis=2) - z * d.std(axis=2, ddof=1) / np.sqrt(seen)
        active &= ~(lb > 0)  # confidently worse than the leader (acc) or below the bar
        active[lead] = True
        if goal == "acc" and active.sum() == 1:
            break
        if goal != "acc":
            c = min(map(tuple, np.argwhere(active)), key=lambda wb: C[wb])
            if -(d[c].mean() + z * d[c].std(ddof=1) / np.sqrt(seen)) > 0:  # cheapest survivor confidently above bar
                return c, used
    return pick(x.mean(axis=2), C, active, goal), used


def replay(X, C, Cref, tier_idx, names, seeds=SEEDS, ns=NS) -> list[dict]:
    """X: correct [writer + S_ref (last), b, r, p]. Mean over seeds and both halves per (goal, strategy, knob, n)."""
    W, B, _, P = X.shape[0] - 1, X.shape[1], X.shape[2], X.shape[3]
    acc = {}

    def add(key, s, share):
        a = acc.setdefault(key, {"share": []})
        a["share"].append(share)
        for k, v in s.items():
            a.setdefault(k, []).append(float(v))

    for pick_r, meas_r in (HALVES, HALVES[::-1]):
        pick_r, meas_r = [r for r in pick_r if r < X.shape[2]], [r for r in meas_r if r < X.shape[2]]
        T = X[:W][:, :, meas_r].mean(axis=(2, 3))
        full = W * B * P * len(pick_r)
        for n in ns:
            for s in range(seeds if n < P else 1):
                pairs = np.random.default_rng([s, n]).choice(P, n, replace=False) if n < P else np.arange(P)
                E = X[:W][:, :, pick_r][:, :, :, pairs].mean(axis=(2, 3))
                Eref = X[W, :, pick_r[0]][:, pairs].mean(axis=1)
                for (strat, knob), (m, extra) in static_masks(E, Eref, C, Cref, tier_idx, names).items():
                    cells = int((m | extra).sum()) * n * len(pick_r) + (B * n if strat != "F" else 0)
                    for g in GOALS:
                        add((g, strat, knob, n), score(T, C, pick(E, C, m, g), g), cells / full)
        Xp = X[:W][:, :, pick_r].mean(axis=2)
        for m_ in BATCHES:
            for delta in DELTAS:
                for g in GOALS:
                    for s in range(seeds):
                        wb, used = sequential(Xp, C, g, m_, delta, np.random.default_rng([s, m_, int(delta * 100)]))
                        add((g, "P3", f"m={m_} delta={delta}", None), score(T, C, wb, g), used / (W * B * P))
    rows = []
    for (g, strat, knob, n), a in acc.items():
        row = {"goal": g if g == "acc" else f"bar{g}", "strategy": strat, "knob": knob, "n": n}
        for k, v in a.items():
            row[k] = round(float(np.mean(v)), 4)
            if k in ("regret", "overpay"):
                row[k + "_p90"] = round(float(np.percentile(v, 90)), 4)
        rows.append(row)
    return rows


def insample(X, C) -> dict:
    E = X[:-1].mean(axis=(2, 3))
    return {str(g): pick(E, C, np.ones(E.shape, bool), g) for g in GOALS}


def check() -> None:
    """Planted crossover: the cheap executor is good only under a writer that looks weak on the reference executor."""
    assert keep_eps([0.9, 0.89, 0.8], [5, 5, 5], 0.0) == [0] and keep_eps([0.7, 0.88, 0.89], [1, 5, 6], 0.0) == [0, 1, 2]
    assert keep_eps([0.9, 0.89], [5, 5], 0.02) == [0, 1]
    true = np.array([[0.70, 0.90, 0.88], [0.70, 0.89, 0.90], [0.90, 0.80, 0.80], [0.70, 0.88, 0.89]])  # last row S_ref
    P, R = 100, 2
    X = np.zeros((4, 3, R, P))
    for a in range(4):
        for b in range(3):
            X[a, b, :, : int(round(true[a, b] * P))] = 1
    C, Cref = np.array([[1.0, 5, 6]] * 3), np.array([1.0, 5, 6])
    rows = replay(X, C, Cref, [[0], [1, 2]], ["B0", "B1", "B2"], seeds=1, ns=(P,))
    get = {(r["goal"], r["strategy"], r["knob"]): r for r in rows}
    assert get[("bar0.98", "F", "")]["overpay"] == 1.0 and get[("bar0.98", "F", "")]["met"] == 1.0
    assert get[("bar0.98", "P1", "eps=0.0 ref=B1")]["overpay"] == 5.0  # writer 2 pruned on B1, cheap plan lost
    assert get[("acc", "F", "")]["regret"] == 0.0 and get[("acc", "F", "")]["share"] == 1.0
    clear = np.zeros((4, 3, R, P))
    clear[:, :, :, ::2] = 1  # 0.5 everywhere
    clear[1, 2] = 1  # one clear winner
    rows = replay(clear, C, Cref, [[0], [1, 2]], ["B0", "B1", "B2"], seeds=20, ns=(P,))
    p3 = [r for r in rows if r["strategy"] == "P3" and r["goal"] == "acc"]
    assert all(r["regret"] == 0.0 and r["share"] == int(r["knob"].split()[0][2:]) / P for r in p3), p3  # one batch
    print("check: ok")


def load_matrix(execs: list[str]):
    opx.EXECS[:] = execs
    S, model, tok, drafts = opx.load()
    items = sorted(S[(ax.WRITERS[0], execs[0], 0)])
    for a in ax.WRITERS + [ax.REF]:
        for b in execs:
            for r in ax.REPS:
                assert set(S[(a, b, r)]) == set(items), (a, b, r)
    X = np.array([[[[S[(a, b, r)][i][0] for i in items] for r in ax.REPS] for b in execs]
                  for a in ax.WRITERS + [ax.REF]], dtype=float)

    def usd(m, t):
        return (t[0] * opx.PRICES[m][0] + t[1] * opx.PRICES[m][1]) / 1e6

    C = np.array([[usd(model[b], tok[(w, b)]) * 1000 + usd(opx.WRITER_MODEL[w], drafts[w]) * 1000 / opx.PER
                   for b in execs] for w in ax.WRITERS])
    Cref = np.array([usd(model[b], tok[(ax.REF, b)]) * 1000 for b in execs])
    return X, C, Cref


def report(rows: list[dict]) -> None:
    def agg(sel):
        keys = [k for k in sel[0] if k not in ("goal", "strategy", "knob", "n")]
        return {k: (np.mean([r[k] for r in sel]), min(r[k] for r in sel) if k in ("met", "near") else max(r[k] for r in sel))
                for k in keys}

    for g in ("acc", "bar0.95", "bar0.98"):
        print(f"\n== goal {g} (P1 / P2: mean over reference executors, worst in brackets)")
        groups = {}
        for r in rows:
            if r["goal"] == g:
                knob = r["knob"].split(" ref=")[0]
                groups.setdefault((r["strategy"], knob, r["n"]), []).append(r)
        for (strat, knob, n), sel in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or 0)):
            a = agg(sel)
            metric = (f"regret {a['regret'][0]:.3f} [{a['regret'][1]:.3f}]  near {a['near'][0]:.2f}" if g == "acc"
                      else f"met {a['met'][0]:.2f} [{a['met'][1]:.2f}]  overpay {a['overpay'][0]:.2f}x [{a['overpay'][1]:.2f}x]")
            print(f"  {strat:3s} {knob:18s} n={str(n):4s} calls {a['share'][0]:.2f}  {metric}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="synthetic checks only")
    ap.add_argument("--add", default="", help="supplementary executors, e.g. B8")
    ap.add_argument("--out", default="search_space_ext.json", help="output file under results/; never overwritten")
    args = ap.parse_args()
    check()
    if args.check:
        return 0
    path = ax.RESULTS / args.out
    if path.exists():
        print(f"{path} exists; pass a new --out", file=sys.stderr)
        return 1
    execs = list(opx.EXECS) + [b for b in args.add.split(",") if b]
    X, C, Cref = load_matrix(execs)
    ins = {g: f"{ax.WRITERS[w]} x {execs[b]}" for g, (w, b) in insample(X, C).items()}
    print("F in-sample picks:", ins)
    rows = replay(X, C, Cref, tiers(execs), execs)
    report(rows)
    path.write_text(json.dumps({"execs": execs, "insample": ins, "rows": rows}, indent=1))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
