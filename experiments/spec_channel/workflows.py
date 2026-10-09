"""Workflow definitions for the Specification channel pilot: C1 (UniDM-DI Restaurant imputation) and C2 (Walmart-Amazon matching).

Design and results: REPORT.md.
"""
from __future__ import annotations

import json
import random
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from c2_setup import FULL_OUT as C2_FULL_SAMPLE, OUT as C2_SAMPLE, load_pairs, stratified
from lexical_baseline import TARBALL, load, norm
from prompts import (DI_EXECUTOR, DI_REF_SPEC, DI_WRITER, EM_EXECUTOR, EM_REF_SPEC, EM_WRITER,
                     di_examples, di_record, em_examples, em_record)

SEED = 0
SCREEN_SEED = 1
C1_K_EXAMPLES = 12
C1_STRAT_SEED, C1_PER_CELL = 2, 3


@dataclass
class Workflow:
    name: str
    writer_prompt: str
    ref_spec: str
    executor_prompt: Callable[[str, dict], str]
    parse: Callable[[str | None], object]
    pilot: list[dict]
    screen: list[dict]


def parse_yes(text: str | None) -> int | None:
    w = (text or "").strip().lower().lstrip("*").split()
    return None if not w else 1 if w[0].startswith("yes") else 0 if w[0].startswith("no") else None


def parse_city(text: str | None) -> str | None:
    lines = [l for l in (text or "").strip().splitlines() if l.strip()]
    return norm(lines[0].strip(" *\"'.")) if lines else None


def c1_examples(train: list[dict], k: int, rng: random.Random) -> list[dict]:
    # C1_PER_CELL rows from every (phone format, city) cell with >= 5 train rows, so both sources' naming is shown; rest random.
    cells = defaultdict(list)
    for r in train:
        cells[("/" in r["phone"], norm(r["city"]))].append(r)
    picked = [r for rows in cells.values() if len(rows) >= 5 for r in rng.sample(rows, C1_PER_CELL)]
    ids = {id(r) for r in picked}
    out = picked + rng.sample([r for r in train if id(r) not in ids], k - len(picked))
    rng.shuffle(out)
    return out


def c1(name: str = "c1", split: str = "test", k: int = 0) -> Workflow:
    """k = 0: pilot (12 random writer examples); k > 0: c1_examples stratified set."""
    with tarfile.open(TARBALL) as tar:
        train, valid, test = (load(tar, s) for s in ("train", "valid", "test"))
    examples = c1_examples(train, k, random.Random(C1_STRAT_SEED)) if k else random.Random(SEED).sample(train, C1_K_EXAMPLES)
    rows = {"valid": valid, "test": test}[split]
    pilot = [{"id": f"{split}-{i}", "record": r, "label": norm(r["city"])} for i, r in enumerate(rows)]
    # Screen: 5 dash-phone + 5 slash-phone valid rows; the two phone formats carry different city conventions.
    rng = random.Random(SCREEN_SEED)
    rows = list(enumerate(valid))
    screen = [{"id": f"valid-{i}", "record": r, "label": norm(r["city"])}
              for i, r in rng.sample([x for x in rows if "/" not in x[1]["phone"]], 5) + rng.sample([x for x in rows if "/" in x[1]["phone"]], 5)]
    return Workflow(name, DI_WRITER.format(examples=di_examples(examples)), DI_REF_SPEC,
                    lambda spec, it: DI_EXECUTOR.format(spec=spec, record=di_record(it["record"])), parse_city, pilot, screen)


def c2(name: str = "c2", path=C2_SAMPLE, key: str = "pilot") -> Workflow:
    sample = json.loads(path.read_text())
    screen = stratified(load_pairs("valid"), 3, 7, 0, random.Random(SCREEN_SEED))
    return Workflow(name, EM_WRITER.format(examples=em_examples(sample["writer_examples"])), EM_REF_SPEC,
                    lambda spec, it: EM_EXECUTOR.format(spec=spec, left=em_record(it["left"]), right=em_record(it["right"])),
                    parse_yes, sample[key], screen)


WORKFLOWS = {"c1": c1, "c1_k40": lambda: c1("c1_k40", "valid", 40), "c2": c2, "c2_full": lambda: c2("c2_full", C2_FULL_SAMPLE, "items")}
