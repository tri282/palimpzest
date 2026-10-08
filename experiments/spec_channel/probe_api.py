"""One-call probe of Purdue GenAI and Gemini: model list, usage/cache fields, rate-limit headers.

Design and results: REPORT.md.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PURDUE_BASE = "https://genai.rcac.purdue.edu/api"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
PROMPT = "Reply with the single word: ok"
OUT = REPO / "experiments" / "spec_channel" / "results" / "probe.json"


def load_env() -> dict[str, str]:
    env = {}
    for line in (REPO / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def call(url: str, headers: dict, body: dict | None = None) -> tuple[int, dict, dict, float]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            status, hdrs, raw = r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        status, hdrs, raw = e.code, dict(e.headers), e.read()
    latency = time.time() - t0
    try:
        payload = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        payload = {"_raw": raw[:500].decode(errors="replace")}
    return status, hdrs, payload, latency


def limit_headers(hdrs: dict) -> dict:
    return {k: v for k, v in hdrs.items() if any(s in k.lower() for s in ("limit", "remaining", "retry", "reset"))}


def pick(ids: list[str], prefs: list[str]) -> str:
    for p in prefs:
        for i in ids:
            if p in i.lower():
                return i
    return ids[0]


def probe_purdue(key: str) -> dict:
    auth = {"Authorization": f"Bearer {key}"}
    s, h, models, _ = call(f"{PURDUE_BASE}/models", auth)
    ids = sorted(m["id"] for m in (models or {}).get("data", []))
    model = pick(ids, ["gemma"]) if ids else None
    out = {"models_status": s, "model_ids": ids, "probe_model": model}
    if model:
        body = {"model": model, "messages": [{"role": "user", "content": PROMPT}], "temperature": 0}
        s, h, resp, lat = call(f"{PURDUE_BASE}/chat/completions", auth, body)
        out |= {"chat_status": s, "latency_s": round(lat, 2), "limit_headers": limit_headers(h),
                "response_keys": sorted((resp or {}).keys()), "usage": (resp or {}).get("usage"),
                "reply": ((resp or {}).get("choices") or [{}])[0].get("message", {}).get("content"),
                "error": None if s == 200 else resp}
    return out


def probe_gemini(key: str) -> dict:
    auth = {"x-goog-api-key": key}
    s, h, models, _ = call(f"{GEMINI_BASE}/models?pageSize=200", auth)
    ids = sorted(m["name"].removeprefix("models/") for m in (models or {}).get("models", [])
                 if "generateContent" in m.get("supportedGenerationMethods", []))
    model = pick(ids, ["gemini-3.5-flash-lite", "flash-lite-latest", "flash"]) if ids else None
    out = {"models_status": s, "model_ids": ids, "probe_model": model}
    if model:
        body = {"contents": [{"parts": [{"text": PROMPT}]}], "generationConfig": {"temperature": 0}}
        s, h, resp, lat = call(f"{GEMINI_BASE}/models/{model}:generateContent", auth, body)
        cand = ((resp or {}).get("candidates") or [{}])[0]
        out |= {"chat_status": s, "latency_s": round(lat, 2), "limit_headers": limit_headers(h),
                "response_keys": sorted((resp or {}).keys()), "usage": (resp or {}).get("usageMetadata"),
                "reply": "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", [])),
                "error": None if s == 200 else resp}
    return out


def main() -> int:
    env = load_env()
    result = {"purdue": probe_purdue(env["PURDUE"]), "gemini": probe_gemini(env["GEMINI"])}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
