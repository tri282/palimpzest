"""Zero-API lexical baselines for entity matching (C2 screening rule: Lexical).

Design and results: REPORT.md.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

EM = Path(__file__).resolve().parents[2] / "testdata" / "fm_data_tasks" / "datasets" / "entity_matching"
DATASETS = ["structured/Walmart-Amazon", "textual/Abt-Buy", "structured/Amazon-Google"]
TOKEN = re.compile(r"[a-z0-9]+(?:[-/.][a-z0-9]+)*")


def table(path: Path) -> dict[str, str]:
    with path.open() as f:
        return {r["id"]: " ".join(v for k, v in r.items() if k not in ("id", "price")).lower() for r in csv.DictReader(f)}


def pairs(path: Path) -> list[tuple[str, str, int]]:
    with path.open() as f:
        return [(r["ltable_id"], r["rtable_id"], int(r["label"])) for r in csv.DictReader(f)]


def tokens(s: str) -> set[str]:
    return set(TOKEN.findall(s))


def model_tokens(s: str) -> set[str]:
    # Model-number-like: mixed letters and digits, or 4+ digits.
    return {t.replace("-", "").replace("/", "").replace(".", "") for t in tokens(s)
            if (re.search(r"\d", t) and re.search(r"[a-z]", t) and len(t) >= 3) or re.fullmatch(r"\d{4,}", t)}


def f1(preds: list[int], labels: list[int]) -> float:
    tp = sum(p and l for p, l in zip(preds, labels))
    fp = sum(p and not l for p, l in zip(preds, labels))
    fn = sum(l and not p for p, l in zip(preds, labels))
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def main() -> int:
    for name in DATASETS:
        d = EM / name
        A, B = table(d / "tableA.csv"), table(d / "tableB.csv")
        train, test = pairs(d / "train.csv"), pairs(d / "test.csv")

        def jac(p):
            return jaccard(tokens(A[p[0]]), tokens(B[p[1]]))

        # Jaccard threshold tuned on train, applied to test.
        train_scores, train_labels = [jac(p) for p in train], [p[2] for p in train]
        best_t = max((t / 100 for t in range(5, 95)), key=lambda t: f1([s >= t for s in train_scores], train_labels))
        labels = [p[2] for p in test]
        jac_f1 = f1([jac(p) >= best_t for p in test], labels)
        mod_f1 = f1([bool(model_tokens(A[p[0]]) & model_tokens(B[p[1]])) for p in test], labels)
        print(f"{name:28s} test={len(test)} pos={sum(labels)}  jaccard(t={best_t:.2f}) F1={jac_f1:.3f}  model-token-overlap F1={mod_f1:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
