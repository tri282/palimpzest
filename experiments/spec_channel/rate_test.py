"""Is Purdue GenAI's rate limit per API key? Unpaced bursts on one key, then on both keys at once; counts 429s.

Usage: python -I rate_test.py [--n 30]. Writes results/rate_test.json. ~3 x n tiny calls. Design: REPORT-C2-ext.md V.2.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_api import PURDUE_BASE, REPO, call, load_env  # noqa: E402

MODEL = "gemma3:12b"  # fast, already loaded by most runs
PROMPT = "Reply with the single word: ok"
OUT = REPO / "experiments" / "spec_channel" / "results" / "rate_test.json"


def burst(key: str, n: int) -> dict:
    """n sequential unpaced calls on one key: how many succeed, how many are throttled, achieved calls per minute."""
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}], "temperature": 0}
    t0, statuses = time.time(), []
    for _ in range(n):
        s, _, _, _ = call(f"{PURDUE_BASE}/chat/completions", {"Authorization": f"Bearer {key}"}, body)
        statuses.append(s)
    elapsed = time.time() - t0
    return {"n": n, "ok": statuses.count(200), "throttled": statuses.count(429), "other": n - statuses.count(200) - statuses.count(429),
            "elapsed_s": round(elapsed, 1), "ok_per_min": round(statuses.count(200) / elapsed * 60, 1)}


def main() -> int:
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 30
    env = load_env()
    keys = {"PURDUE": env["PURDUE"], "PURDUE_2": env["PURDUE_2"]}
    single = burst(keys["PURDUE"], n)
    print(f"one key alone: {single}", flush=True)
    time.sleep(70)  # let the first key's window clear
    with ThreadPoolExecutor(2) as pool:
        both = dict(zip(keys, pool.map(lambda k: burst(k, n), keys.values())))
    for k, v in both.items():
        print(f"both keys, {k}: {v}", flush=True)
    # Per key: each key alone-rate holds while both run. Shared: the combined rate is about the single-key rate.
    combined = sum(v["ok_per_min"] for v in both.values())
    verdict = "per key" if combined > 1.5 * single["ok_per_min"] else "shared"
    print(f"combined {combined:.1f}/min vs single {single['ok_per_min']:.1f}/min: limit looks {verdict}", flush=True)
    OUT.write_text(json.dumps({"model": MODEL, "n": n, "single": single, "both": both, "verdict": verdict}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
