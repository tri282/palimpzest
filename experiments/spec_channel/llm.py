"""Single-call client for Purdue GenAI and Gemini with per-provider pacing, retries, and R3 cost fields.

Design and results: REPORT.md.
"""
from __future__ import annotations

import re
import time

from probe_api import GEMINI_BASE, PURDUE_BASE, call, load_env

PROVIDER = {"gemini-3.5-flash-lite": "gemini"}  # everything else is Purdue
MIN_INTERVAL_S = {"purdue": 60 / 18, "gemini": 60 / 14}
# USD per million tokens, list rate. Gemini row from the Google Gemini API pricing page
# (read 2026-08-28). Purdue models have no list price: cost stays None (unpriced), tokens are the cost unit.
RATES = {"gemini-3.5-flash-lite": {"prompt": 0.30, "completion": 2.50}}
MAX_ATTEMPTS = 6
THINK = re.compile(r"<think>(.*?)(?:</think>|$)", re.S)  # unclosed: generation stopped mid-reasoning
_last: dict[str, float] = {}
_env: dict[str, str] = {}


def provider_of(model: str) -> str:
    return PROVIDER.get(model, "purdue")


def split_reasoning(msg: dict) -> tuple[str | None, str]:
    """Answer text and reasoning text. Purdue returns reasoning in `reasoning_content`, `reasoning`, or as <think> blocks
    inside `content` (qwen3, deepseek-r1); its `completion_tokens` already include reasoning, so it is only split out here."""
    content = msg.get("content") or ""
    reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "") + "".join(THINK.findall(content))
    return THINK.sub("", content).strip() or None, reasoning


def check_split() -> None:
    assert split_reasoning({"content": "Yes"}) == ("Yes", "")
    assert split_reasoning({"content": "No", "reasoning_content": "abc"}) == ("No", "abc")
    assert split_reasoning({"content": "<think>x y</think>\n\nAnswer: No", "reasoning": ""}) == ("Answer: No", "x y")
    assert split_reasoning({"content": "<think>cut off"}) == (None, "cut off")


def _once(model: str, provider: str, text: str, temperature: float) -> dict:
    if provider == "purdue":
        body = {"model": model, "messages": [{"role": "user", "content": text}], "temperature": temperature}
        s, _, resp, lat = call(f"{PURDUE_BASE}/chat/completions", {"Authorization": f"Bearer {_env['PURDUE']}"}, body)
        msg = (((resp or {}).get("choices") or [{}])[0] or {}).get("message") or {}
        usage = (resp or {}).get("usage") or {}
        content, reasoning = split_reasoning(msg)
        out = {"content": content, "tokens_in": usage.get("prompt_tokens"), "tokens_out": usage.get("completion_tokens"),
               "cached_tokens": None, "reasoning_tokens": None, "reasoning_chars": len(reasoning),
               "reasoning_tokens_est": round(len(reasoning) / 4)}  # ~4 chars per token; Purdue reports no reasoning count
    else:
        body = {"contents": [{"parts": [{"text": text}]}], "generationConfig": {"temperature": temperature}}
        s, _, resp, lat = call(f"{GEMINI_BASE}/models/{model}:generateContent", {"x-goog-api-key": _env["GEMINI"]}, body)
        cand = ((resp or {}).get("candidates") or [{}])[0]
        usage = (resp or {}).get("usageMetadata") or {}
        out = {"content": "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", [])) or None,
               "tokens_in": usage.get("promptTokenCount"), "tokens_out": usage.get("candidatesTokenCount"),
               "cached_tokens": usage.get("cachedContentTokenCount", 0), "reasoning_tokens": usage.get("thoughtsTokenCount", 0),
               "reasoning_chars": 0, "reasoning_tokens_est": usage.get("thoughtsTokenCount", 0)}
    return {"status": s, "latency_s": round(lat, 3), "raw_response": resp, **out}


def ask(model: str, text: str, temperature: float = 0.0) -> dict:
    """One logical call: paces per provider, retries transient failures, returns final attempt plus attempt history."""
    if not _env:
        _env.update(load_env())
    provider = provider_of(model)
    attempts, wait_total = [], 0.0
    for attempt in range(MAX_ATTEMPTS):
        wait = max(0.0, MIN_INTERVAL_S[provider] - (time.time() - _last.get(provider, 0.0)))
        time.sleep(wait)
        wait_total += wait
        _last[provider] = time.time()
        r = _once(model, provider, text, temperature)
        ok = r["status"] == 200 and r["content"]
        attempts.append({"status": r["status"], "latency_s": r["latency_s"], "error": None if ok else str(r["raw_response"])[:300]})
        if ok:
            break
        # 429 / 5xx / null body (Purdue proxy quirk): back off and retry.
        backoff = 30.0 if r["status"] == 429 else min(60.0, 2.0 * 2 ** attempt)
        time.sleep(backoff)
        wait_total += backoff

    rate = RATES.get(model)
    cost = (None if rate is None or r["tokens_in"] is None or r["tokens_out"] is None
            else (r["tokens_in"] * rate["prompt"] + r["tokens_out"] * rate["completion"]) / 1e6)
    return {"model": model, "provider": provider, "temperature": temperature, **r,
            "error": attempts[-1]["error"], "retry_count": len(attempts) - 1, "attempts": attempts,
            "latency_total_s": round(sum(a["latency_s"] for a in attempts), 3), "wait_s": round(wait_total, 3),
            "cache_hit": None if provider == "purdue" else bool(r["cached_tokens"]),
            "cost_usd_list": cost}


if __name__ == "__main__":
    check_split()
    print("split_reasoning ok")
