"""Workflow definitions for the Specification channel pilot: C1 (UniDM-DI Restaurant imputation) and C2 (Walmart-Amazon matching).

Design and results: REPORT.md.
"""
from __future__ import annotations

import json
import random
import tarfile
from dataclasses import dataclass
from typing import Callable

from c2_setup import FULL_OUT as C2_FULL_SAMPLE, OUT as C2_SAMPLE, load_pairs, stratified
from lexical_baseline import TARBALL, load, norm
from prompts import (DI_EXECUTOR, DI_REF_SPEC, DI_WRITER, EM_EXECUTOR, EM_REF_SPEC, EM_WRITER,
                     di_examples, di_record, em_examples, em_record)

SEED = 0
SCREEN_SEED = 1
C1_K_EXAMPLES = 12


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


def c1() -> Workflow:
    with tarfile.open(TARBALL) as tar:
        train, valid, test = (load(tar, s) for s in ("train", "valid", "test"))
    examples = random.Random(SEED).sample(train, C1_K_EXAMPLES)
    pilot = [{"id": f"test-{i}", "record": r, "label": norm(r["city"])} for i, r in enumerate(test)]
    # Screen: 5 dash-phone + 5 slash-phone valid rows; the two phone formats carry different city conventions.
    rng = random.Random(SCREEN_SEED)
    rows = list(enumerate(valid))
    screen = [{"id": f"valid-{i}", "record": r, "label": norm(r["city"])}
              for i, r in rng.sample([x for x in rows if "/" not in x[1]["phone"]], 5) + rng.sample([x for x in rows if "/" in x[1]["phone"]], 5)]
    return Workflow("c1", DI_WRITER.format(examples=di_examples(examples)), DI_REF_SPEC,
                    lambda spec, it: DI_EXECUTOR.format(spec=spec, record=di_record(it["record"])), parse_city, pilot, screen)


def c2(name: str = "c2", path=C2_SAMPLE, key: str = "pilot") -> Workflow:
    sample = json.loads(path.read_text())
    screen = stratified(load_pairs("valid"), 3, 7, 0, random.Random(SCREEN_SEED))
    return Workflow(name, EM_WRITER.format(examples=em_examples(sample["writer_examples"])), EM_REF_SPEC,
                    lambda spec, it: EM_EXECUTOR.format(spec=spec, left=em_record(it["left"]), right=em_record(it["right"])),
                    parse_yes, sample[key], screen)


WORKFLOWS = {"c1": c1, "c2": c2, "c2_full": lambda: c2("c2_full", C2_FULL_SAMPLE, "items")}
