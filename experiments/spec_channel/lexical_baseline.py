"""Zero-API lexical baselines for UniDM-DI Restaurant city imputation (screening rule: Lexical).

Design and results: REPORT.md.
"""
from __future__ import annotations

import csv
import io
import re
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

TARBALL = Path(__file__).resolve().parents[2] / "testdata" / "scope" / "data_workspace.tar.gz"
PREFIX = "data/UNIDM-datasets/data_imputation/Restaurant/"


def load(tar: tarfile.TarFile, split: str) -> list[dict]:
    raw = tar.extractfile(PREFIX + f"{split}.csv").read().decode()
    return list(csv.DictReader(io.StringIO(raw)))


def norm(s: str) -> str:
    return " ".join(s.lower().split())


def area_code(phone: str) -> str:
    m = re.match(r"\D*(\d{3})", phone)
    return m.group(1) if m else ""


def rules(train: list[dict]) -> dict:
    """Train-fit city predictors: area-code majority, name lookup, area code x phone format (the C1 rule)."""
    by_code: dict[str, Counter] = defaultdict(Counter)
    by_name: dict[str, Counter] = defaultdict(Counter)
    by_code_fmt: dict[tuple, Counter] = defaultdict(Counter)
    for r in train:
        by_code[area_code(r["phone"])][norm(r["city"])] += 1
        by_name[norm(r["name"])][norm(r["city"])] += 1
        # Phone format identifies the source (slash vs dash), and each source has its own city convention.
        by_code_fmt[(area_code(r["phone"]), "/" in r["phone"])][norm(r["city"])] += 1
    global_top = Counter(norm(r["city"]) for r in train).most_common(1)[0][0]

    def predict_code(r):
        c = by_code.get(area_code(r["phone"]))
        return c.most_common(1)[0][0] if c else global_top

    def predict_name_then_code(r):
        c = by_name.get(norm(r["name"]))
        return c.most_common(1)[0][0] if c else predict_code(r)

    def predict_code_fmt(r):
        c = by_code_fmt.get((area_code(r["phone"]), "/" in r["phone"]))
        return c.most_common(1)[0][0] if c else predict_code(r)

    return {"area_code_majority": predict_code, "name_lookup_then_area_code": predict_name_then_code,
            "area_code_x_phone_format": predict_code_fmt}


def main() -> int:
    with tarfile.open(TARBALL) as tar:
        train, valid, test = (load(tar, s) for s in ("train", "valid", "test"))
    by_name = {norm(r["name"]) for r in train}
    predictors = rules(train)

    print(f"train={len(train)} valid={len(valid)} test={len(test)}")
    for split, rows in (("valid", valid), ("test", test)):
        leak = sum(norm(r["name"]) in by_name for r in rows)
        print(f"{split}: name seen in train {leak}/{len(rows)}")
        for label, fn in predictors.items():
            hits = sum(fn(r) == norm(r["city"]) for r in rows)
            print(f"  {label:28s} acc={hits / len(rows):.3f} ({hits}/{len(rows)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
