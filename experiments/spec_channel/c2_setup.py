"""C2 (Walmart-Amazon entity matching) setup: stratified pilot and full-run samples, spec-writer examples, and a context-truncation check on Purdue.

Design and results: REPORT.md.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from em_lexical_baseline import jaccard, tokens  # noqa: E402
from probe_api import PURDUE_BASE, REPO, call, load_env  # noqa: E402
from prompts import EM_EXECUTOR, EM_REF_SPEC, EM_WRITER, em_examples, em_record  # noqa: E402

DATA = REPO / "testdata" / "fm_data_tasks" / "datasets" / "entity_matching" / "structured" / "Walmart-Amazon"
OUT = REPO / "experiments" / "spec_channel" / "results" / "c2_sample.json"
FULL_OUT = REPO / "experiments" / "spec_channel" / "results" / "c2_full_sample.json"
SEED, FULL_SEED = 0, 2
N_POS, N_HARD_NEG, N_RAND_NEG = 40, 30, 10
FULL_STRATA = (75, 56, 19)  # 150 pairs at pilot proportions
K_POS, K_HARD_NEG = 4, 4
PURDUE_MODELS = ["gemma3:12b", "llama3.3:70b", "qwen2.5:72b", "mistral:latest"]


def load_pairs(split: str) -> list[dict]:
    with (DATA / "tableA.csv").open() as f:
        A = {r["id"]: r for r in csv.DictReader(f)}
    with (DATA / "tableB.csv").open() as f:
        B = {r["id"]: r for r in csv.DictReader(f)}
    with (DATA / f"{split}.csv").open() as f:
        out = []
        for r in csv.DictReader(f):
            left, right = A[r["ltable_id"]], B[r["rtable_id"]]
            out.append({"id": f"{split}-{r['ltable_id']}-{r['rtable_id']}", "left": left, "right": right,
                        "label": int(r["label"]), "jaccard": jaccard(tokens(em_record(left)), tokens(em_record(right)))})
    return out


def stratified(pairs: list[dict], n_pos: int, n_hard: int, n_rand: int, rng: random.Random) -> list[dict]:
    # Hard negatives = highest token Jaccard among non-matches (same brand / near-identical titles).
    pos = rng.sample([p for p in pairs if p["label"]], n_pos)
    neg = sorted((p for p in pairs if not p["label"]), key=lambda p: -p["jaccard"])
    hard = neg[:n_hard]
    rand = rng.sample(neg[n_hard:], n_rand)
    for p, kind in [(x, "pos") for x in pos] + [(x, "hard_neg") for x in hard] + [(x, "rand_neg") for x in rand]:
        p["stratum"] = kind
    sample = pos + hard + rand
    rng.shuffle(sample)
    return sample


def build() -> dict:
    rng = random.Random(SEED)
    pilot = stratified(load_pairs("test"), N_POS, N_HARD_NEG, N_RAND_NEG, rng)
    examples = stratified(load_pairs("train"), K_POS, K_HARD_NEG, 0, rng)
    data = {"seed": SEED, "dataset": "structured/Walmart-Amazon", "pilot": pilot, "writer_examples": examples}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1))
    longest = max(pilot, key=lambda p: len(em_record(p["left"]) + em_record(p["right"])))
    print(f"pilot={len(pilot)} (pos {N_POS}, hard_neg {N_HARD_NEG}, rand_neg {N_RAND_NEG}) writer_examples={len(examples)}")
    print(f"hard_neg jaccard range {min(p['jaccard'] for p in pilot if p['stratum'] == 'hard_neg'):.2f}-"
          f"{max(p['jaccard'] for p in pilot if p['stratum'] == 'hard_neg'):.2f}; "
          f"pos jaccard median {sorted(p['jaccard'] for p in pilot if p['label'])[N_POS // 2]:.2f}")
    print(f"writer prompt chars={len(EM_WRITER.format(examples=em_examples(examples)))}; "
          f"longest executor prompt chars (ref spec)={len(EM_EXECUTOR.format(spec=EM_REF_SPEC, left=em_record(longest['left']), right=em_record(longest['right'])))}")
    return data


def build_full() -> dict:
    # Fresh test pairs disjoint from the pilot; writer examples reused so the writer prompt is unchanged.
    pilot = json.loads(OUT.read_text())
    used = {p["id"] for p in pilot["pilot"]}
    items = stratified([p for p in load_pairs("test") if p["id"] not in used], *FULL_STRATA, random.Random(FULL_SEED))
    assert not used & {p["id"] for p in items}
    data = {"seed": FULL_SEED, "dataset": pilot["dataset"], "excludes": OUT.name, "items": items,
            "writer_examples": pilot["writer_examples"]}
    FULL_OUT.write_text(json.dumps(data, indent=1))
    counts = {k: sum(p["stratum"] == k for p in items) for k in ("pos", "hard_neg", "rand_neg")}
    hard = [p["jaccard"] for p in items if p["stratum"] == "hard_neg"]
    longest = max(items, key=lambda p: len(em_record(p["left"]) + em_record(p["right"])))
    print(f"full={len(items)} {counts} overlap_with_pilot=0 hard_neg jaccard range {min(hard):.2f}-{max(hard):.2f}")
    print(f"longest executor prompt chars (ref spec)="
          f"{len(EM_EXECUTOR.format(spec=EM_REF_SPEC, left=em_record(longest['left']), right=em_record(longest['right'])))}")
    return data


def truncation_check() -> None:
    # Secret word at the very start of a ~6k-token prompt; a backend that truncates from the front cannot recall it.
    key = load_env()["PURDUE"]
    filler = " ".join(f"Line {i}: the quick brown fox jumps over the lazy dog." for i in range(500))
    text = f"The secret word is PELICAN.\n{filler}\nWhat is the secret word stated at the very beginning? Answer with the word only."
    for model in PURDUE_MODELS:
        body = {"model": model, "messages": [{"role": "user", "content": text}], "temperature": 0}
        s, _, resp, lat = call(f"{PURDUE_BASE}/chat/completions", {"Authorization": f"Bearer {key}"}, body)
        msg = (((resp or {}).get("choices") or [{}])[0] or {}).get("message") or {}
        print(f"{model:14s} status={s} prompt_tokens={(resp or {}).get('usage', {}).get('prompt_tokens')} "
              f"latency={lat:.1f}s reply={(msg.get('content') or '')[:40]!r}")


if __name__ == "__main__":
    build_full() if "--full" in sys.argv else build()
    if "--truncation-check" in sys.argv:
        truncation_check()
