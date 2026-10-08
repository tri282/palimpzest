"""Small per-model baseline on C1 Restaurant imputation under a neutral spec: accuracy, latency, tokens, reasoning leakage.

Design and results: REPORT.md.
"""
from __future__ import annotations

import json
import random
import re
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lexical_baseline import TARBALL, load, norm  # noqa: E402
from probe_api import GEMINI_BASE, PURDUE_BASE, REPO, call, load_env  # noqa: E402

N_RECORDS = 12
SEED = 0
MIN_INTERVAL_S = 60 / 18  # ponytail: one global pacing interval, per-endpoint limiter lands in the runner
SPEC = "Fill in the missing city for this restaurant record. Answer with the city name only, nothing else."
OUT = REPO / "experiments" / "spec_channel" / "results" / "model_baseline.jsonl"

# (model id, extra request fields, suffix appended to the prompt)
PURDUE_MODELS = [
    ("gemma3:12b", {}, ""),
    ("gemma3:27b", {}, ""),
    ("llama3.1:latest", {}, ""),
    ("llama3.3:70b", {}, ""),
    ("qwen2.5:72b", {}, ""),
    ("mistral:latest", {}, ""),
    ("phi4:latest", {}, ""),
    ("qwen3:14b", {}, " /no_think"),
    ("qwen3:32b", {}, " /no_think"),
    ("gpt-oss:120b", {"reasoning_effort": "low"}, ""),
]
GEMINI_MODEL = "gemini-3.5-flash-lite"
THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def prompt_for(r: dict, suffix: str) -> str:
    rec = f"name: {r['name']}\naddr: {r['addr']}\nphone: {r['phone']}\ntype: {r['type']}\ncity: ?"
    return f"{SPEC}\n\n{rec}{suffix}"


def ask_purdue(key: str, model: str, extra: dict, text: str) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": text}], "temperature": 0, **extra}
    s, _, resp, lat = call(f"{PURDUE_BASE}/chat/completions", {"Authorization": f"Bearer {key}"}, body)
    msg = (((resp or {}).get("choices") or [{}])[0] or {}).get("message") or {}
    usage = (resp or {}).get("usage") or {}
    return {"status": s, "latency_s": lat, "content": msg.get("content"), "msg_keys": sorted(msg),
            "reasoning_chars": len(msg.get("reasoning_content") or msg.get("reasoning") or ""),
            "tokens_in": usage.get("prompt_tokens"), "tokens_out": usage.get("completion_tokens"),
            "error": None if s == 200 and msg else str(resp)[:300]}


def ask_gemini(key: str, text: str) -> dict:
    body = {"contents": [{"parts": [{"text": text}]}], "generationConfig": {"temperature": 0}}
    s, _, resp, lat = call(f"{GEMINI_BASE}/models/{GEMINI_MODEL}:generateContent", {"x-goog-api-key": key}, body)
    cand = ((resp or {}).get("candidates") or [{}])[0]
    usage = (resp or {}).get("usageMetadata") or {}
    return {"status": s, "latency_s": lat,
            "content": "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", [])),
            "msg_keys": [], "reasoning_chars": usage.get("thoughtsTokenCount", 0),
            "tokens_in": usage.get("promptTokenCount"), "tokens_out": usage.get("candidatesTokenCount"),
            "error": None if s == 200 else str(resp)[:300]}


def main() -> int:
    env = load_env()
    with tarfile.open(TARBALL) as tar:
        test = load(tar, "test")
    records = random.Random(SEED).sample(test, N_RECORDS)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    jobs = [(m, x, sfx, "purdue") for m, x, sfx in PURDUE_MODELS] + [(GEMINI_MODEL, {}, "", "gemini")]
    last = 0.0
    with OUT.open("w") as f:
        for model, extra, suffix, provider in jobs:
            # Call 0 repeats record 0 as a warm-up so model load time is excluded from latency.
            for i, r in enumerate([records[0]] + records):
                time.sleep(max(0.0, MIN_INTERVAL_S - (time.time() - last)))
                last = time.time()
                text = prompt_for(r, suffix)
                res = ask_purdue(env["PURDUE"], model, extra, text) if provider == "purdue" else ask_gemini(env["GEMINI"], text)
                raw = res["content"] or ""
                pred = norm(THINK_RE.sub("", raw).strip().strip(".").splitlines()[-1] if raw.strip() else "")
                row = {"model": model, "provider": provider, "warmup": i == 0, "record": r["name"],
                       "label": norm(r["city"]), "pred": pred, "correct": pred == norm(r["city"]),
                       "think_tag": "<think>" in raw, **{k: v for k, v in res.items() if k != "content"},
                       "raw": raw[:200]}
                f.write(json.dumps(row) + "\n")
                f.flush()
            print(f"done {model}", flush=True)
    summarize()
    return 0


def summarize() -> None:
    rows = [json.loads(l) for l in OUT.read_text().splitlines()]
    print(f"{'model':24s} acc   err  lat_med  tok_out_med  think_tag  reasoning")
    for model in dict.fromkeys(r["model"] for r in rows):
        rs = [r for r in rows if r["model"] == model and not r["warmup"]]
        ok = [r for r in rs if not r["error"]]
        lat = sorted(r["latency_s"] for r in ok) or [float("nan")]
        tout = sorted(r["tokens_out"] or 0 for r in ok) or [0]
        warm = next(r["latency_s"] for r in rows if r["model"] == model and r["warmup"])
        print(f"{model:24s} {sum(r['correct'] for r in rs) / len(rs):.2f}  {len(rs) - len(ok):3d}  "
              f"{lat[len(lat) // 2]:6.1f}s  {tout[len(tout) // 2]:11d}  {sum(r['think_tag'] for r in rs):9d}  "
              f"{max((r['reasoning_chars'] or 0) for r in rs):9d}  (warmup {warm:.0f}s)")


if __name__ == "__main__":
    sys.exit(summarize() if "--summary" in sys.argv else main())
