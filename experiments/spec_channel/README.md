# Specification Channel Experiment

Measures whether an optimizer that profiles LLM operators one at a time picks the right plan when one LLM writes the instructions another LLM follows. Results and method: `REPORT.md`. Pre-registered C2 full run: `PREREG-C2.md`.

---

## I. Layout

| file | role |
|------|------|
| `runner.py` | runner: fresh specs per replicate, shuffled downstream calls, resumable JSONL logs; `--ref-executor` adds held-out references B2, B3 |
| `run_resilient.sh` | restarts `runner.py` after outages or crashes until the run completes |
| `analyze.py` | offline analysis: interaction and estimability tests, independence baseline and regret per reference executor, cost model, cache sensitivity, F1, workload sizing |
| `optimizer_policies.py` | offline comparison of optimizer estimation policies (single, averaged, matched reference; joint sampling incl. equal budget) and draft features vs executor gain; targeted estimator P7 (synthetic self-check only); needs `analysis.json` |
| `workflows.py` | workload definitions: `c1` (Restaurant city imputation), `c2` (Walmart-Amazon matching, pilot sample), `c2_full` (150-pair full-run sample), `c1_k40` (C1 redesign test: 40 stratified writer examples, valid split), `ab` (Abt-Buy matching, pilot-shaped sample) |
| `prompts.py` | writer, executor, and reference-spec templates |
| `llm.py` | single-call client for Purdue GenAI and Gemini: pacing, retries, token and cost fields |
| `screen.py` | cheap screens: spec variation across writers, executor sensitivity to spec |
| `c2_setup.py` | builds the stratified C2 pilot sample (`--full`: full-run sample disjoint from the pilot; `--abt-buy`: Abt-Buy sample) and spec-writer examples |
| `ref_screen.py` | screens candidate reference executors on C2 validation pairs |
| `lexical_baseline.py`, `em_lexical_baseline.py` | zero-API rule baselines for C1 and C2 |
| `model_baseline.py`, `probe_api.py` | per-model sanity baseline and API probe |
| `results/` | screen outputs, C2 sample, `analysis.json`, and `runs/<run_id>/` logs |

---

## II. Setup

**Install dependencies (from the repository root):**  

```bash
pip install -e .
```

**Add API keys to `.env` at the repository root:**  

```txt
PURDUE=<PURDUE_GENAI_API_KEY>
GEMINI=<GEMINI_API_KEY>
```

**Download the Walmart-Amazon data:**  

```bash
mkdir -p testdata/fm_data_tasks && cd testdata/fm_data_tasks
curl -fO https://fm-data-tasks.s3.us-west-1.amazonaws.com/datasets.tar.gz
tar -xzf datasets.tar.gz --wildcards 'datasets/entity_matching/structured/Walmart-Amazon/*'
```

* **Restaurant data:** the UniDM data-imputation benchmark, read from the SCOPE artifact's `data_workspace.tar.gz` at `testdata/scope/data_workspace.tar.gz` (path inside: `data/UNIDM-datasets/data_imputation/Restaurant/`). See the SCOPE repository (`github.com/waetr/SCOPE-LLM-optimizer`) for the artifact.
* *Note: `testdata/*/` is gitignored; datasets are never committed.*

**Verify the APIs and keys (two short calls):**  

```bash
python -I experiments/spec_channel/probe_api.py
```

Expected: `chat_status: 200` for both providers.

---

## III. Running

**Build the C2 sample (once):**  

```bash
python -I experiments/spec_channel/c2_setup.py
```

**Run a pilot:**  

```bash
python -I experiments/spec_channel/runner.py pilot-c2 --workflow c2
python -I experiments/spec_channel/runner.py pilot-c1 --workflow c1
```

* **Resume:** re-run the same command. Finished specs and calls are skipped; failed calls are retried; the seeded order is unchanged.
* **Stop on outage:** the runner exits after 5 consecutive failed calls; re-run to continue.
* **Long runs:** launch detached so the run survives the terminal.

```bash
setsid nohup python -I experiments/spec_channel/runner.py pilot-c2 --workflow c2 >> experiments/spec_channel/results/runs/pilot-c2.log 2>&1 < /dev/null &
```

**Run the C2 full experiment (unattended, ~16 h):**  

```bash
python -I experiments/spec_channel/c2_setup.py --full
setsid nohup experiments/spec_channel/run_resilient.sh full-c2 --workflow c2_full --replicates 9 --ref-executor \
  >> experiments/spec_channel/results/runs/full-c2.log 2>&1 < /dev/null &
```

* **Exit codes:** `runner.py` returns 0 when done, 2 on a transient stop (outage, failed spec call), 3 on a config mismatch. `run_resilient.sh` restarts on any non-zero code except 3, waiting `RETRY_SLEEP` seconds (default 900), at most `MAX_RESTARTS` times (default 30).

**Analyze:**  

```bash
python -I experiments/spec_channel/analyze.py pilot-c2 pilot-c1
python -I experiments/spec_channel/analyze.py full-c2
```

Writes `results/analysis.json` and prints a per-workload summary.

**Compare optimizer policies (after `analyze.py`):**  

```bash
python -I experiments/spec_channel/optimizer_policies.py full-c2
```

Writes `results/optimizer_policies.json`; asserts the single-reference policy reproduces `analysis.json`.

---

## IV. Outputs

`results/runs/<run_id>/`:

* **`config.json`:** arms, models, temperatures, seed, writer prompt, reference spec.
* **`specs.jsonl`:** one row per generated spec (writer call): full text, tokens, latency, list-price estimate.
* **`calls.jsonl`:** one row per downstream call:

```txt
run_id, workflow, seed, replicate, arm_a, arm_b, operator, item_id, stratum, spec_hash, call_index, model_switch,
label, prediction, correct, unparsed, timestamp, model, provider, temperature, status, content, raw_response,
tokens_in, tokens_out, cached_tokens, cache_hit, reasoning_tokens, reasoning_chars,
latency_s, latency_total_s, wait_s, retry_count, attempts, error, cost_usd_list
```

* **`cache_hit`:** `null` means the provider does not report cache use (Purdue GenAI).
* **`cost_usd_list`:** list-price estimate; `null` means unpriced (Purdue-hosted models have no list price).
* **Failed calls:** kept as reliability data; analysis uses the latest successful row per (replicate, cell, item).

---

## V. Rate Limits

* **Purdue GenAI:** 20 requests/min per user; the client paces at 18/min. Switching models costs a 30-50 s reload, so downstream calls are shuffled within blocks of one executor.
* **Gemini free tier (`gemini-3.5-flash-lite`):** 15 requests/min, 500/day. Gemini only writes specs (one call per replicate per workload), well under the daily cap.
