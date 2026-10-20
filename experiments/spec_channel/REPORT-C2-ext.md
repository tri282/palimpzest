# C2 Extension: Report

Goal: test whether the C2 result (rules written by an upstream LLM change which downstream LLM is best) holds across many executor and writer models, and what that means for an optimizer that profiles operators one at a time.  
*Requirements: `REPORT.md` Sections VII and IX (the full C2 run this extends). Pre-registration: `PREREG-C2-ext.md` (to be committed before the first replicate). Raw logs: `results/runs/{drift-c2, ext-easy, ext-models, ext-writers, p9-scores-ext}` (planned).*  

Status: **design, no results yet.** Design fixed 2026-10-19; runs pending.

---

## I. Summary

*Placeholder: filled after the runs.*

---

## II. Context

### 1. Question

A semantic query optimizer (for example Palimpzest's Abacus) profiles each LLM operator on its own and combines the numbers. In the Specification channel an upstream LLM (the **writer**) writes instructions that a downstream LLM (the **executor**) follows on every record, so the executor's behavior depends on what the writer wrote.

### 2. What The Full C2 Run Found (`REPORT.md` VII, IX)

* **Rules erase the strong executor's lead.** Under a one-line generic instruction, `llama3.3:70b` beats `gemma3:12b` by 12 points (0.91 vs 0.79); under either writer's rules they tie (about 0.87). Significant in all 9 replicates (p < 0.001): adding a writer interacts with the executor choice.
* **Which writer did not matter:** the which-writer x which-executor interaction was not detected (p = 0.34; about 17 replicates needed for a 0.05 effect).
* **Optimizer consequence:** profiling executors under the generic instruction overpays about 1.7x at list prices; profiling them under the best writer's rules (Abacus-style) removes the overpay but misses the quality bar more often.

### 3. Why Extend

* 4 executors and 2 writers cannot show whether the effect is general across model families and sizes, or whether the best writer differs by executor (a **crossover**, the case where a single "best upstream" choice misleads the optimizer).
* The natural-mix estimate rested on 19 easy non-matches carrying 86% of the weight.

---

## III. Pipeline

```
8 labeled training pairs (4 matches, 4 hard non-matches)
  -> WRITER (temperature 0.7): matching rules, at most 250 words     1 call per draft
  -> EXECUTOR (temperature 0): rules + record pair, "Answer Yes or No only"   1 call per pair
  -> first word parsed as Yes / No, scored against the label
Baseline (S_ref): the executor with a one-line generic instruction instead of rules
```

* **Dataset:** Walmart-Amazon (structured product matching), test split.
* **Pairs:** 250 per executor: 75 matches, 56 hard non-matches (highest token Jaccard among non-matches), 119 easy non-matches (19 from the full run, 100 new: `results/c2_easy_sample.json`, disjoint from all earlier samples, below the hard-negative Jaccard floor).
* **Natural weights** (test split: 193 matches, 86 hard, 2,049 pairs): 0.094 / 0.042 / 0.864.
* **Replicate:** one fresh draft from each writer. Rules vary from draft to draft, so effects must hold across drafts. **9 replicates, committed in advance**; no replicates are added after looking.

---

## IV. Models

### 1. Writers (6)

| id | model | drafts | source |
|---|---|---:|---|
| A0 | `gemini-3.5-flash-lite` | 9 | saved from the full run (Gemini API, 2026-10-09) |
| A1 | `qwen2.5:72b` | 9 | saved from the full run |
| A2 | `gpt-oss:120b` | 9 | new |
| A3 | `llama3.3:70b` | 9 | new |
| A4 | `gemma3:27b` | 9 | new |
| A5 | `qwen3:8b` | 9 | new |

New writers are chosen from models that are also executors, so each has a writer-on-its-own-rules cell. A writer screen (2 drafts each: on-topic rules, not a copy of the examples, length within a tolerance of the limit) runs before their drafts are frozen.

### 2. Executors (13)

| id | model | type | host |
|---|---|---|---|
| B0 | `gemma3:12b` | plain | Purdue |
| B1 | `llama3.3:70b` | plain | Purdue |
| B2 | `gpt-oss:120b` | reasoning | Purdue |
| B3 | `gemma3:27b` | plain | Purdue |
| B4 | `gemma4:26b-a4b` (MoE, 4B active) | plain | Chameleon |
| B5 | `qwen3:8b` | reasoning | Chameleon |
| B6 | `qwen3:32b` | reasoning | Purdue (failed host check) |
| B7 | `gpt-oss:latest` (= 20B, 14 GB) | reasoning | Purdue |
| B8 | `deepseek-r1:14b` | reasoning | Purdue (failed host check) |
| B9 | `mistral:latest` (= 7B, 4.4 GB) | plain | Purdue (failed host check) |
| B10 | `phi4:latest` (= 14B, 9.1 GB) | plain | Chameleon |
| B11 | `qwen2.5:72b` | plain | Purdue |
| B12 | `llama4:latest` (= Llama 4 Scout 17B-16E, w4a16) | plain | Purdue |

* **Screened** before the run (`drop_screen.py`, `results/drop_screen/`): 40 validation pairs under the generic instruction; drop if accuracy <= majority rate, one answer >= 95%, or > 10% unparsed (rule fixed in advance). Dropped: `gemma3:1b` (always Yes), `llama3.1:latest` (8B; 75% unparsed: writes an analysis instead of Yes/No). `gemma3:4b` is not served on Purdue.
* **All executors are candidate plans** in the optimizer analysis (in the full run B2 and B3 were profiling-only references).
* **B11 = writer A1:** its A1 cells are the writer applying its own rules; reported separately and excluded from the cross-executor test.
* **Hosts:** B0-B3 stay on Purdue (compared with full-run data); B7 stays on Purdue (compared with B2). Each executor uses one host for all its calls. Chameleon copies use the same Ollama tags. Planned on Chameleon: B4-B6, B8-B10; after the host check (VII.1) B6, B8 and B9 run on Purdue, as the pre-registered rule requires.

---

## V. Runs

### 1. Blocks

| block | what | calls | why this shape |
|---|---|---:|---|
| checks | per-key rate test; host check (6 Chameleon models x 40 screen pairs vs their Purdue answers); drift anchor (B0-B3 x S_ref x 150 full-run pairs, 1 replicate) | ~880 | confirm old and new data, and the two hosts, are comparable |
| A1 `ext-easy` | B0-B3 x {A0, A1, S_ref} x 100 new easy x 9 | 10,800 | B0-B3 already have the 150 full-run pairs with the same drafts (16,200 calls, reused); only the new easy pairs are missing |
| A2 `ext-models` | B4-B12 x {A0, A1, S_ref} x 250 x 9 | 60,750 | new executors on the same drafts and pairs, so only the executor changes |
| B0 writers | A2-A5 x 9 drafts | 36 | new drafts |
| B1 `p9-scores-ext` | PZ validator on the 36 new drafts x 3 samples | 108 | Abacus-style champion over 6 writers (VI.3) |
| B2 `ext-writers` | A2-A5 x all 13 executors x 250 x 9 | 117,000 | full cross: every executor under every writer, so crossovers can be tested over the whole matrix; S_ref cells are shared with block A |
| **total new** | | **~189.6k** | plus 16,200 reused |

Blocks A and B differ only in where the drafts come from (saved vs new); together they form one 6-writer x 13-executor matrix.

### 2. Order And Time

* Checks first, then replicates 0-2 for every run, then 3-8. All 9 are committed; the early look is descriptive and cannot change what runs.
* **Hosts:** Purdue about 94.8k calls (B0-B3, B7, B11, B12, new writers, judge) over two API keys (about 18 calls per minute per key; per-key limit to be confirmed), about 44 h; Chameleon about 94.7k calls (B4-B6, B8-B10) on one GPU node (Ollama, 4 parallel requests), about 48 h, of which `qwen3:32b` is about 22 h.
* **Estimated time:** about 2 days of running plus 0.5-1 day of setup. Throughput on Chameleon is an estimate until measured.
* **As run:** with B6, B8 and B9 moved to Purdue, Purdue carries about 142k calls and Chameleon about 47k (B4, B5, B10). B6 and B8 are latency-bound on Purdue (about 10 s and 7.5 s per call), so each runs as two concurrent processes on one key, still under the per-key limit; about 3.2 days in total.

---

## VI. Analysis (Fixed In `PREREG-C2-ext.md` Before Replicate 0)

### 1. Quantities

* **Gain** of executor b on draft r: g(b, r) = accuracy under the rules of draft r minus S̄_ref(b), the generic-instruction accuracy averaged over its 9 replicates (S_ref does not depend on the draft).
* **Unit:** drafts primary (9 per writer), pairs secondary (draft x pair bootstrap).
* **Primary sample:** the 250 pairs, with per-stratum results. Natural mix secondary (86% easy pairs, near ceiling for most models).
* **Unparsed answers** count as wrong; sensitivity analysis excludes them; an executor with > 10% unparsed in any rule cell moves to exploratory.

### 2. Primary Tests (Holm Over 7 Tests At Family-Wise 0.05, Smallest Effect Of Interest 0.03)

Tests (a)-(c) use block A, gains pooled over writers A0 and A1, one-sided. Tests (d1)-(d2) use the full 6 x 13 matrix, rule cells only.

| test | contrast | statistic | direction |
|---|---|---|---|
| (a) | across 12 executors (B11 excluded): generic accuracy vs gain | Spearman rho, split-half: generic accuracy on one half of the pairs, gain on the other; must also hold for the error-rate ratio (1 - acc_rules) / (1 - acc_generic) | rho < 0 (weaker models gain more) |
| (b1-b3) | size pairs: `gemma3` 12B vs 27B, `qwen3` 8B vs 32B, `gpt-oss` 20B vs 120B | paired t over 9 drafts | smaller gains more |
| (c) | B0 vs B1 on the 100 new easy pairs | paired t over 9 drafts | B0 gains more |
| (d1) | writer x executor interaction (omnibus) | split-plot linear model on accuracy (draft within writer as the whole-plot unit); writer x executor F-tested against executor x draft(writer) | any interaction |
| (d2) | cost of a single champion writer | cross-validated champion regret: per executor, accuracy under its own best writer minus under the oracle champion (best writer averaged over executors); both chosen on drafts {0, 2, 4, 6, 8}, measured on {1, 3, 5, 7}, then halves swapped and averaged; draft bootstrap (drafts resampled within writer), one-sided: with no crossover the expected regret is <= 0, since the pooled champion estimates a common best writer more precisely than any one executor's pick | regret > 0 |

*(a) uses split halves because gain = acc_rules - acc_generic contains generic accuracy and is capped at 1 - acc_generic, so a negative correlation appears even with no real effect.*

*Exact statistics, thresholds and decision rules: `PREREG-C2-ext.md`.*

*(d1) alone can be significant from scale differences with the same writer ranking everywhere; (d2) asks whether the ranking actually changes enough to cost quality. (d2) is cross-validated because the best of 6 noisy writer means beats any single writer even with no real effect.*

* **Own-model cells** (a writer executing its own rules: A1 x B11, A2 x B2, A3 x B1, A4 x B3, A5 x B5) are excluded from (d1) and (d2): a writer that helps its own model most is a crossover by construction. Reported separately, with (d1)-(d2) recomputed including them as a what-if.

### 3. Optimizer Analysis (Offline, On The Full Matrix)

Abacus is **modelled, not run**: Palimpzest cannot yet optimize an LLM operator that follows an aggregate, and the writer aggregates its examples. The full matrix gives the true best plan; policies are simulated on it:

* **P0:** profile executors under the generic instruction (the overpaying policy in the full run).
* **P9:** Abacus-style. Pick a champion writer with Palimpzest's default validator (`MAP_VALIDATOR_PROMPT`, writer modelled as a `sem_map` with input field `examples` and output field `rules`, messages from `PromptFactory`, no temperature sent; `gpt-oss:120b` stands in for o4-mini), then profile executors under the champion's rules. **Sample 0 of 3 is the champion** (Abacus judges once); samples 1-2 measure how often a single judgment flips. The 18 saved drafts were judged the same way (`results/p9_scores_pz.json`).
* **Champions:** the **oracle** champion (best writer on the data, averaged over executors) is the reference, since the measured matrix is the ground truth; the **judge** champion is Abacus's route. Reported side by side, including whether the judge misses the oracle champion.
* Outcomes: best plan picked, meets the quality bar, overpay at list prices (VI.4), and how often each champion is the wrong writer for some executor.

### 4. Prices (List, Per 1M Tokens In / Out, Read 2026-10-19)

Purdue and Chameleon are free; dollar figures are what the same calls would cost at a provider's list price. Rule: DeepInfra where listed (its last list price if delisted); Nscale where DeepInfra never listed it; sensitivity analysis at the lowest and highest price found for each model. Reasoning tokens are priced as output. Tokens (input, output, estimated reasoning) are reported next to every dollar figure, since price does not track model size (older or scarcer models can cost more).

| model | role | provider | in | out | other prices found |
|---|---|---|---:|---:|---|
| `gemini-3.5-flash-lite` | A0 | Google (rechecked 2026-10-19) | 0.30 | 2.50 | |
| `qwen2.5:72b` | A1, B11 | DeepInfra (Qwen2.5-72B-Instruct) | 0.36 | 0.40 | |
| `gpt-oss:120b` | A2, B2 | DeepInfra | 0.037 | 0.17 | |
| `llama3.3:70b` | A3, B1 | DeepInfra (Llama-3.3-70B-Instruct-Turbo) | 0.10 | 0.32 | |
| `gemma3:27b` | A4, B3 | DeepInfra (gemma-3-27b-it) | 0.08 | 0.16 | |
| `qwen3:8b` | A5, B5 | Nscale | 0.07 | 0.18 | OpenRouter (deprecated) / Alibaba 0.12 / 0.46 |
| `gemma3:12b` | B0 | DeepInfra | 0.05 | 0.15 | |
| `gemma4:26b-a4b` | B4 | DeepInfra (gemma-4-26b-a4b-it) | 0.07 | 0.34 | |
| `qwen3:32b` | B6 | DeepInfra | 0.08 | 0.28 | |
| `gpt-oss:latest` (20B) | B7 | DeepInfra (gpt-oss-20b) | 0.03 | 0.14 | |
| `deepseek-r1:14b` | B8 | Nscale (R1-Distill-Qwen-14B) | 0.07 | 0.07 | Novita 0.15 / 0.15; Alibaba 0.144 / 0.431; Together 1.6 / 1.6 |
| `mistral:latest` (7B v0.3) | B9 | DeepInfra (Mistral-7B-Instruct-v0.3, last list price before delisting) | 0.03 | 0.06 | Together 0.2 / 0.2 |
| `phi4:latest` (14B) | B10 | DeepInfra (phi-4) | 0.07 | 0.14 | |
| `llama4:latest` (Scout 17B-16E) | B12 | DeepInfra (Llama-4-Scout-17B-16E-Instruct) | 0.10 | 0.30 | |

*Provider builds differ from the ones measured (e.g. DeepInfra's Turbo is an FP8 build; the measured `llama3.3:70b` is presumed Q4): prices are for the model, not the exact build.*

---

## VII. Results

### 1. Checks

**Rate test** (`results/rate_test.json`): 18.7 successful calls per minute on one key, 46 on two at once, so the limit is per key.

**Host check** (`results/drop_screen_chameleon/host_check.json`): each self-hosted model answers the 40 screen pairs under `S_ref`; it stays self-hosted at >= 38 of 40 answers equal to Purdue's.

| id | model | agree with Purdue | host |
|---|---|---:|---|
| B4 | `gemma4:26b-a4b` | 39 / 40 (rerun, plain mode) | Chameleon |
| B5 | `qwen3:8b` | 40 / 40 | Chameleon |
| B6 | `qwen3:32b` | 37 / 40 | Purdue |
| B8 | `deepseek-r1:14b` | 36 / 40 | Purdue |
| B9 | `mistral:latest` | 36 / 40 | Purdue |
| B10 | `phi4:latest` | 40 / 40 | Chameleon |

* **gemma4 ran in a different mode at first.** The self-hosted Ollama (0.40.6) turns gemma4 thinking on by default; Purdue serves it plain. Same request on both hosts, yet the first check gave a median of 350 output tokens and about 1,100 reasoning characters on Chameleon against 2 tokens and none on Purdue, while answers still agreed 38/40. Self-hosted gemma4 calls now send `reasoning_effort: "none"` (a deviation, VIII.8); the rerun gives 2 tokens (median and maximum), no reasoning text and 39/40. The first check is kept in `results/drop_screen_chameleon-think/`.
* **The B6, B8, B9 gaps are systematic.** The rerun repeats every self-hosted answer and token count exactly (40/40 for all five other models; digests unchanged), and Purdue repeats itself at temperature 0 (`S_ref` across 9 full-run replicates: 40/40 for B0 and B1, 39.4/40 for B2). No call was truncated or unparsed on Chameleon. Prompt token counts differ by 2-6 per call between hosts, which points to a different chat template or build behind the same tag. B6's 3 flips are all Yes on Purdue, No on Chameleon (all true matches); B9's 4 lean the same way (3 Yes to No, 1 unparsed to No); B8's 4 go both ways.

*Placeholders: block A, block B, tests (a)-(d), optimizer analysis, reliability.*

---

## VIII. Limitations

1. **Model identity.** Purdue's API exposes only the model tag; its model list (https://docs.rcac.purdue.edu/services/genai/models/) links each tag to its source. Resolved there: `gpt-oss:latest` = 20B (14 GB), `mistral:latest` = 7B (4.4 GB), `phi4:latest` = 14B (9.1 GB), `llama3.1:latest` = 8B, `llama4:latest` = `RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16` (4-bit weights, 16-bit activations; a Hugging Face build, so not every model is served the same way). Quantization is stated only for some tags (e.g. `llama3.1:70b-instruct-q4_K_M`); others are presumed to be Ollama's default 4-bit builds, which the file sizes are consistent with. No digests are exposed, and the library page may not match the build Purdue pulled. Chameleon copies use the same tags and are checked against Purdue answers, which shows agreement on these pairs only.
2. **Cost is not measured in dollars.** Dollar figures use list prices (DeepInfra), not what was paid; Purdue is free. Reasoning tokens are estimated: Purdue hides reasoning text but bills it in completion tokens, so they are completion tokens minus the visible answer.
3. **Abacus is modelled, not run** (VI.3), and the judge is a stand-in model. Palimpzest's validator sends no temperature, so the judge runs at the provider's default and is not deterministic: on the 18 full-run drafts, 8 got different verdicts across 3 judgments, so a single judgment (as Abacus makes) picks its champion partly by chance. Kept as is to stay faithful; sample 0 is the pick and samples 1-2 measure the flips.
4. **Scope.** One dataset, one task type (product matching), one generic instruction as the stand-in.
5. **Writers are mixed in provenance:** A0 is an API model whose drafts date from the full run; A2-A5 are Purdue models drafted now.
6. **Statistics.** Test (a) has 12 executors, so it only detects a strong relation; the natural mix is near ceiling; temperature 0 is not fully deterministic for reasoning models.
7. **Time.** B0-B3's 150-pair data is about 10 days older than the rest; the drift anchor checks this but does not prove it.
8. **Deviation from the freeze.** One code change after the freeze commit: self-hosted gemma4 calls send `reasoning_effort: "none"` (VII.1), so B4 runs plain as on Purdue. It changes request fields only, not prompts, parser, pairs, drafts or tests; it was made before any B4 data existed, and the host check was rerun under it.
9. **The host check compares answers only.** It did not notice gemma4 thinking on one host and not the other, since the answers still agreed. Mode (output tokens, reasoning text) is now compared by hand for every self-hosted model (VII.1) but is not part of the pre-registered rule.
10. **Size pair (b2) crosses hosts.** B5 (`qwen3:8b`) runs on Chameleon and B6 (`qwen3:32b`) on Purdue. B5 agrees 40/40 with its Purdue answers, but the pair compares two sizes on two serving builds.

---

## IX. Reliability And Latency

Every call logs latency, wait time, retries, host and model tag; a `/models` snapshot is taken at the start and end of each block. **Latency is reported descriptively, within each host only**: on Purdue it mixes shared load, queueing, model reloads and our own pacing, and Chameleon runs on different hardware. Tokens (input, output, estimated reasoning) are the cost unit.
