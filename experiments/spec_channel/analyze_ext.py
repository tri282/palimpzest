"""Pre-registered analysis of the C2 extension: tests (a)-(d2), own-model what-if, champion-writer view.

Usage: python -I analyze_ext.py [--check]. Tests and rules: PREREG-C2-ext.md; results: REPORT-C2-ext.md.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import RUNS, t_test  # noqa: E402
from workflows import parse_answer_line, parse_yes  # noqa: E402

RESULTS = RUNS.parent
REF = "S_ref"
WRITERS = ["A0", "A1", "A2", "A3", "A4", "A5"]
EXECUTORS = [f"B{i}" for i in range(13)]
OWN = {("A1", "B11"), ("A2", "B2"), ("A3", "B1"), ("A4", "B3"), ("A5", "B5")}
REPS, N_PAIRS, N_EASY = range(9), 250, 100
SIZE_PAIRS = {"b1": ("B0", "B3"), "b2": ("B5", "B6"), "b3": ("B7", "B2")}
SESOI = {"a": 0.6, "b": 0.03, "c": 0.03, "d2": 0.03}
N_BOOT, ALPHA, UNPARSED_MAX = 10_000, 0.05, 0.10
SOURCES = ["full-c2", "ext-easy", "ext-writers-p1", "ext-models-p2", "ext-writers-p2"] + \
          [f"ext-{k}-B{b}" for b in (4, 5, 6, 8, 9, 10) for k in ("models", "writers")]
BOXED = re.compile(r"\\boxed\{\s*(?:\\text\{)?\s*(yes|no)\b", re.I)


def parse_lenient(text: str | None) -> int | None:
    """Sensitivity parser: first word, else last `Answer: Yes/No` line, else last `\\boxed{Yes/No}`."""
    p = parse_answer_line(text)
    if p is None and text:
        hits = BOXED.findall(text)
        p = (1 if hits[-1].lower() == "yes" else 0) if hits else None
    return p


# ---------- data ----------

def load(sources=SOURCES) -> dict:
    """Latest successful row per (replicate, arm, executor, pair); an error row only where no success exists."""
    calls = {}
    for run in sources:
        f = RUNS / run / "calls.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:  # torn last line of a run still being written
                continue
            k = (r["replicate"], r["arm_a"], r["arm_b"], r["item_id"])
            if not r["error"] or k not in calls or calls[k]["error"]:
                calls[k] = r
    return calls


def scored(calls: dict, lenient: bool) -> dict:
    """(arm, executor, replicate) -> {pair: (correct, unparsed, stratum)}; failed calls count as wrong and unparsed."""
    out = defaultdict(dict)
    for (rep, a, b, item), r in calls.items():
        if r["error"]:
            ok, unp = False, True
        else:
            p = parse_lenient(r["content"]) if lenient else parse_yes(r["content"])
            ok, unp = p is not None and p == r["label"], p is None
        out[(a, b, rep)][item] = (ok, unp, r["stratum"])
    return out


def acc(cell: dict, items=None) -> float:
    vals = [v[0] for k, v in cell.items() if items is None or k in items]
    return sum(vals) / len(vals)


def executor_status(S: dict) -> dict:
    """Primary for (d1)-(d2) if every writer and S_ref cell has all 250 pairs in all 9 replicates and every rule cell
    is <= 10% unparsed (PREREG II.5); otherwise the reason it is out."""
    status = {}
    for b in EXECUTORS:
        missing = [(a, r) for a in WRITERS + [REF] for r in REPS if len(S.get((a, b, r), {})) < N_PAIRS]
        unp = {a: np.mean([v[1] for r in REPS for v in S.get((a, b, r), {}).values()] or [0]) for a in WRITERS}
        if missing:
            status[b] = f"incomplete ({len(missing)} of 63 cell-replicates short)"
        elif max(unp.values()) > UNPARSED_MAX:
            status[b] = "exploratory: unparsed " + ", ".join(f"{a} {u:.0%}" for a, u in unp.items() if u > UNPARSED_MAX)
        else:
            status[b] = "primary"
    return status


def matrix(S: dict, execs: list[str], own: bool) -> np.ndarray:
    """acc[w, b, r] on the 250 pairs; NaN for own-model cells unless own=True."""
    A = np.array([[[acc(S[(w, b, r)]) for r in REPS] for b in execs] for w in WRITERS])
    if not own:
        for w, b in OWN:
            if b in execs:
                A[WRITERS.index(w), execs.index(b), :] = np.nan
    return A


# ---------- statistics ----------

def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) by continued fraction (Numerical Recipes betacf); no scipy here."""
    if x <= 0 or x >= 1:
        return float(x >= 1)
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x > (a + 1) / (a + b + 2):
        return 1 - betainc(b, a, 1 - x)
    tiny, c, d = 1e-300, 1.0, 1 - (a + b) * x / (a + 1)
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        for num in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
                    -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1 + num * d
            d = 1 / (d if abs(d) > tiny else tiny)
            c = 1 + num / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1) < 1e-12:
            break
    return math.exp(lbt) * h / a


def f_sf(f: float, d1: int, d2: int) -> float:
    return betainc(d2 / 2, d1 / 2, d2 / (d2 + d1 * f)) if f > 0 else 1.0


def d1_test(A: np.ndarray) -> dict:
    """Split-plot F for writer:executor: nested least squares, draft(writer) fixed, residual = executor x draft(writer)."""
    W, B, R = A.shape
    rows = [(w, b, r, A[w, b, r]) for w in range(W) for b in range(B) for r in range(R) if not np.isnan(A[w, b, r])]
    y = np.array([v for *_, v in rows])
    draft = np.array([[w * R + r == j for j in range(W * R)] for w, b, r, _ in rows], float)
    exe = np.array([[b == j for j in range(B)] for w, b, r, _ in rows], float)
    cell = np.array([[w * B + b == j for j in range(W * B)] for w, b, r, _ in rows], float)

    def rss(X):
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        return float(((y - X @ beta) ** 2).sum()), np.linalg.matrix_rank(X)

    rss_r, k_r = rss(np.hstack([draft, exe]))
    rss_f, k_f = rss(np.hstack([draft, exe, cell]))
    df1, df2 = k_f - k_r, len(y) - k_f
    F = ((rss_r - rss_f) / df1) / (rss_f / df2)
    return {"F": round(F, 3), "df": [int(df1), int(df2)], "p": f_sf(F, df1, df2), "n": len(y)}


def regret(A: np.ndarray, order=None) -> float:
    """Cross-validated champion regret R (PREREG III.5) on acc[w, b, r]; order = draft order per writer."""
    W, B, R = A.shape
    X = A if order is None else np.stack([A[w][:, order[w]] for w in range(W)])
    halves = (list(range(0, R, 2)), list(range(1, R, 2)))
    vals = []
    for pick, meas in (halves, halves[::-1]):
        mp, mm = np.nanmean(X[:, :, pick], axis=2), np.nanmean(X[:, :, meas], axis=2)  # [w, b]
        champ = int(np.nanargmax(np.nanmean(mp, axis=1)))
        for b in range(B):
            if np.isnan(mp[champ, b]):  # champion is this executor's own-model writer: no regret defined
                continue
            vals.append(mm[int(np.nanargmax(mp[:, b])), b] - mm[champ, b])
    return float(np.mean(vals))


def d2_test(A: np.ndarray, seed: int = 0) -> dict:
    R0, rng = regret(A), np.random.default_rng(seed)
    boot = np.array([regret(A, rng.integers(0, A.shape[2], (A.shape[0], A.shape[2]))) for _ in range(N_BOOT)])
    return {"R": round(R0, 4), "p": float((boot <= 0).mean()), "ci95": [round(float(np.percentile(boot, q)), 4) for q in (2.5, 97.5)]}


def ranks(x) -> np.ndarray:
    x = np.asarray(x, float)
    o = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    for v in np.unique(x):  # average ranks for ties
        r[x == v] = r[x == v].mean()
    return r


def spearman_lower(x, y, seed: int = 0) -> dict:
    """Spearman rho and one-sided p for rho < 0: exact over all permutations for n <= 10, else 10^6 Monte Carlo."""
    rx, ry = ranks(x), ranks(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    n, cx = len(rx), rx - rx.mean()
    sd = np.sqrt((cx ** 2).sum() * ((ry - ry.mean()) ** 2).sum())
    le = tot = 0
    if n <= 10:
        perms = itertools.permutations(ry - ry.mean())
        while len(chunk := np.array(list(itertools.islice(perms, 200_000)))):
            le += int(((chunk @ cx) / sd <= rho + 1e-12).sum())
            tot += len(chunk)
    else:
        rng = np.random.default_rng(seed)
        for _ in range(10):
            P = np.argsort(rng.random((100_000, n)), axis=1)
            le += int((((ry - ry.mean())[P] @ cx) / sd <= rho + 1e-12).sum())
            tot += 100_000
    return {"rho": round(rho, 3), "p": le / tot, "n": n, "exact": n <= 10}


def one_sided_t(x) -> dict:
    t = t_test(x)
    p = None if t["p"] is None else (t["p"] / 2 if t["mean"] > 0 else 1 - t["p"] / 2)
    return {**t, "p": p}


def holm(ps: dict) -> dict:
    """Holm-adjusted p per test (None = not computable, kept in the family at p = 1)."""
    items = sorted(((p if p is not None else 1.0), k) for k, p in ps.items())
    m, run, out = len(items), 0.0, {}
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


# ---------- tests on the data ----------

def halves_stratified(S: dict, b: str) -> tuple[set, set]:
    """Pairs split once into H1 / H2 within each stratum, seed 0 (PREREG III.1)."""
    strata = defaultdict(list)
    for item, (_, _, s) in S[(REF, b, 0)].items():
        strata[s].append(item)
    rng, h1 = np.random.default_rng(0), set()
    for s in sorted(strata):
        items = sorted(strata[s])
        h1 |= {items[i] for i in rng.permutation(len(items))[: len(items) // 2]}
    return h1, set(S[(REF, b, 0)]) - h1


def gain(S, b, r, items=None) -> float:
    s_bar = np.mean([acc(S[(REF, b, k)], items) for k in REPS])
    return np.mean([acc(S[(w, b, r)], items) for w in ("A0", "A1")]) - s_bar


def test_a(S, execs) -> dict:
    execs = [b for b in execs if b != "B11"]
    h1, h2 = halves_stratified(S, execs[0])
    gen = [np.mean([acc(S[(REF, b, k)], h1) for k in REPS]) for b in execs]
    gn = [np.mean([gain(S, b, r, h2) for r in REPS]) for b in execs]

    def errors(a_list, b):
        return sum(1 - v[0] for a in a_list for k in REPS for i, v in S[(a, b, k)].items() if i in h2)

    # Rule errors are summed over two writers, so halved to the same 9-replicate scale as the generic errors.
    er = [(errors(["A0", "A1"], b) / 2 + 0.5) / (errors([REF], b) + 0.5) for b in execs]
    main, ceil = spearman_lower(gen, gn), spearman_lower(gen, [-e for e in er])  # rho(gen, ER) > 0 <=> rho(gen, -ER) < 0
    return {"executors": execs, "rho": main["rho"], "p": main["p"], "exact": main["exact"],
            "ceiling_rho": -ceil["rho"], "ceiling_p": ceil["p"], "generic_h1": dict(zip(execs, map(float, gen))),
            "gain_h2": dict(zip(execs, map(float, gn))), "error_ratio": dict(zip(execs, er))}


def test_pair(S, small, large, items=None) -> dict:
    d = [gain(S, small, r, items) - gain(S, large, r, items) for r in REPS]
    return {"small": small, "large": large, **one_sided_t(d), "d": [round(float(x), 4) for x in d]}


def champion_view(A: np.ndarray, execs: list[str]) -> dict:
    """P9 with the oracle champion (all 9 drafts) and the judge champion sets (amendment 1), per executor regret vs
    its own best writer; descriptive."""
    mean_wb = np.nanmean(A, axis=2)
    oracle = int(np.nanargmax(np.nanmean(mean_wb, axis=1)))
    own_best = np.nanargmax(mean_wb, axis=0)
    judge = {}
    for f in ("p9_scores_pz.json", "p9_scores_ext.json"):
        for rep, cell in json.loads((RESULTS / f).read_text())["replicates"].items():
            for w, v in cell.items():
                if w in WRITERS:
                    judge.setdefault(int(rep), {})[w] = v["scores"][0]
    out = {"oracle_champion": WRITERS[oracle], "own_best": dict(zip(execs, (WRITERS[i] for i in own_best)))}
    for rule in ("best", "worst", "first"):
        reg = []
        for r in REPS:
            top = max(s for s in judge[r].values() if s is not None)
            tied = [WRITERS.index(w) for w in WRITERS if judge[r].get(w) == top]
            for b in range(len(execs)):
                vals = {w: A[w, b, r] for w in tied if not np.isnan(A[w, b, r])}
                if not vals:
                    continue
                pick = {"best": max(vals, key=vals.get), "worst": min(vals, key=vals.get), "first": min(vals)}[rule]
                reg.append(A[own_best[b], b, r] - A[pick, b, r])
        out[f"judge_{rule}_regret"] = round(float(np.mean(reg)), 4)
    out["oracle_regret"] = round(float(np.nanmean([mean_wb[own_best[b], b] - mean_wb[oracle, b]
                                                   for b in range(len(execs))])), 4)
    out["judge_set_size"] = {r: sum(judge[r].get(w) == max(s for s in judge[r].values() if s is not None)
                                    for w in WRITERS) for r in REPS}
    return out


def analyze(lenient: bool, calls: dict, extra=()) -> dict:
    S = scored(calls, lenient)
    status = executor_status(S)
    prim = [b for b in EXECUTORS if status[b] == "primary" or b in extra and not status[b].startswith("incomplete")]
    A, A_own = matrix(S, prim, own=False), matrix(S, prim, own=True)
    res = {"parser": "lenient" if lenient else "parse_yes", "executor_status": status, "executors": prim,
           "acc": {f"{WRITERS[w]}/{b}": round(float(np.mean(A_own[w, i])), 4) for w in range(6) for i, b in enumerate(prim)},
           "acc_ref": {b: round(float(np.mean([acc(S[(REF, b, r)]) for r in REPS])), 4) for b in prim},
           "d1": d1_test(A), "d2": d2_test(A), "d1_own": d1_test(A_own), "d2_own": d2_test(A_own),
           "champion": champion_view(A, prim)}
    if lenient:
        return res
    # (a)-(c) only under the pre-registered parser.
    res["a"] = test_a(S, prim)
    for k, (s, l) in SIZE_PAIRS.items():
        res[k] = test_pair(S, s, l) if s in prim and l in prim else {"small": s, "large": l, "p": None, "note": "not computable"}
    easy = {k[3] for k, r in calls.items() if r.get("run_id") == "ext-easy"}
    assert len(easy) == N_EASY, len(easy)
    res["c"] = test_pair(S, "B0", "B1", easy)
    adj = holm({k: res[k]["p"] for k in ("a", "b1", "b2", "b3", "c", "d1", "d2")})
    for k, p in adj.items():
        eff = {"a": -res["a"]["rho"] if k == "a" else None, "d2": res["d2"]["R"]}.get(k, res[k].get("mean"))
        sesoi = SESOI.get(k[0] if k != "d2" else "d2")
        ok = p < ALPHA and (sesoi is None or (eff is not None and eff >= sesoi))
        if k == "a":
            ok = ok and res["a"]["ceiling_rho"] > 0 and res["a"]["ceiling_p"] < ALPHA
        res[k] |= {"p_holm": round(p, 4), "supports": bool(ok)}
    return res


def report(res: dict) -> None:
    print(f"\n=== parser: {res['parser']} ===")
    for b, s in res["executor_status"].items():
        print(f"  {b:4} {s}")
    print(f"  (d1) F={res['d1']['F']} df={res['d1']['df']} p={res['d1']['p']:.4g}   own cells in: F={res['d1_own']['F']} "
          f"p={res['d1_own']['p']:.4g}")
    print(f"  (d2) R={res['d2']['R']} CI95={res['d2']['ci95']} p={res['d2']['p']:.4g}   own cells in: R={res['d2_own']['R']}")
    print(f"  champion: {res['champion']}")
    for k in ("a", "b1", "b2", "b3", "c", "d1", "d2"):
        if "p_holm" in res.get(k, {}):
            print(f"  {k:3} p={res[k]['p']} p_holm={res[k]['p_holm']} supports={res[k]['supports']}")


def check() -> None:
    """Synthetic checks of the statistics; no data needed."""
    assert parse_lenient("## Step 1\n...\nThe final answer is: $\\boxed{Yes}$") == 1
    assert parse_lenient("Reasoning.\n**Answer: No**") == 0 and parse_lenient("No.") == 0 and parse_lenient("hmm") is None
    for f, d1, d2 in ((4.965, 1, 10), (2.711, 5, 20), (3.316, 2, 30)):  # textbook F(0.95) critical values
        assert abs(f_sf(f, d1, d2) - 0.05) < 1e-3, (f, f_sf(f, d1, d2))
    rng = np.random.default_rng(1)
    W, B, R = 6, 8, 9
    base = rng.normal(0.85, 0.03, (1, B, 1)) + rng.normal(0, 0.02, (W, 1, 1))  # same writer order for all executors
    draft = rng.normal(0, 0.01, (W, 1, R))
    flat = base + draft + rng.normal(0, 0.01, (W, B, R))
    assert d1_test(flat)["p"] > 0.01 and regret(flat) < 0.03
    cross = flat.copy()
    cross[np.arange(B) % W, np.arange(B), :] += 0.08  # each executor gets its own best writer
    assert d1_test(cross)["p"] < 0.01 and regret(cross) > 0.03
    assert spearman_lower([1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1])["p"] == 1 / 720
    h = holm({"x": 0.01, "y": 0.04, "z": None})
    assert np.allclose([h["x"], h["y"], h["z"]], [0.03, 0.08, 1.0])
    print("check: ok")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="synthetic checks only")
    if ap.parse_args().check:
        check()
        return 0
    check()
    calls = load()
    out = {"primary": analyze(False, calls), "lenient": analyze(True, calls, extra=("B10", "B12"))}
    for res in out.values():
        report(res)
    path = RESULTS / "analysis_ext.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
