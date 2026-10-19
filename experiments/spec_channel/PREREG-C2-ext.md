# C2 Extension: Pre-Registered Analysis

Fixed before replicate 0 of any extension run starts; the commit containing this file is the freeze point. Design and context: `REPORT-C2-ext.md`. Full run this extends: `PREREG-C2.md`, `REPORT.md` VII.

---

## I. Design

* **Workload:** Walmart-Amazon entity matching, C2 writer and executor prompts unchanged (`prompts.py`: `EM_WRITER`, `EM_EXECUTOR`, `EM_REF_SPEC`), parser `parse_yes` (first word).
* **Pairs:** 250 test pairs: the 150 of `results/c2_full_sample.json` (75 matches, 56 hard, 19 easy) plus the 100 easy of `results/c2_easy_sample.json` (seed 3, disjoint from all earlier samples). Workflows `c2_easy` (100) and `c2_union` (250).
* **Writers (temperature 0.7, same 8 labeled training pairs):** A0 `gemini-3.5-flash-lite`, A1 `qwen2.5:72b` (saved `full-c2` drafts, replicates 0-8, copied with `--specs-from full-c2`); A2 `gpt-oss:120b`, A3 `llama3.3:70b`, A4 `gemma3:27b`, A5 `qwen3:8b` (new drafts, after the writer screen, II.1).
* **Executors (temperature 0, no max_tokens):** B0 `gemma3:12b`, B1 `llama3.3:70b`, B2 `gpt-oss:120b`, B3 `gemma3:27b`, B4 `gemma4:26b-a4b`, B5 `qwen3:8b`, B6 `qwen3:32b`, B7 `gpt-oss:latest`, B8 `deepseek-r1:14b`, B9 `mistral:latest`, B10 `phi4:latest`, B11 `qwen2.5:72b`, B12 `llama4:latest`. Selected by the drop-rule screen (`results/drop_screen/verdicts.json`). All are candidate plans.
* **Cells:** every writer and `S_ref` (the one-line generic instruction) with every executor.
* **Replicates:** 9. A replicate r is draft r of each writer; every cell runs in every replicate. **No extension:** results at 9 replicates are reported as they are; replicates are not added after looking.
* **Runs:**

| run | executors | pairs | instructions | calls |
|---|---|---:|---|---:|
| `drift-c2` | B0-B3 | 150 | `S_ref` | 600 (1 replicate) |
| `ext-easy` | B0-B3 | 100 easy | A0, A1, `S_ref` | 10,800 |
| `ext-models` | B4-B12 | 250 | A0, A1, `S_ref` | 60,750 |
| `ext-writers` | B0-B12 | 250 | A2-A5 | 117,000 |
| `p9-scores-ext` | judge `gpt-oss:120b` | - | 36 new drafts x 3 samples | 108 |

* B0-B3 on the 150 full-run pairs reuse `full-c2` (same drafts); their 250-pair numbers are `full-c2` + `ext-easy`.
* **Order:** checks (II), then replicates 0-2 of every run, then 3-8. Replicate order and executor order shuffled; within a replicate each executor's calls are contiguous (model reloads cost 30-50 s). Each host or key stream is its own run id (e.g. `ext-models-chameleon`); runs are merged by (writer, replicate, executor, pair) for analysis.
* **Hosts:** B4, B5, B6, B8, B9, B10 on a self-hosted Ollama server (same tags); all others on Purdue GenAI. Each executor uses one host for all its calls; host logged per call.

---

## II. Checks And Their Rules

1. **Writer screen** (2 drafts per new writer, not used in the run): a writer passes if both drafts are non-empty, at most 375 words (1.5x the 250-word limit; the saved A1 drafts reach 330), contain no example record copied verbatim, and mention at least 3 of {model number, brand, variant (color / size / capacity), bundle or accessory, missing or noisy field} (keyword match). A failing writer is replaced, in order, by `phi4:latest`, then `gemma4:26b-a4b`. Writer drafts for the run are generated after the screen and frozen before any executor call.
2. **Host check:** each self-hosted executor answers the 40 screen pairs (`drop_screen.py` sample) under `S_ref`; it stays self-hosted if at least 38 of 40 answers equal its Purdue answers, otherwise it runs on Purdue.
3. **Drift anchor:** `drift-c2` answers vs the per-pair modal `S_ref` answer of the same executor in `full-c2`. Agreement below 95% for an executor flags its old-vs-new comparisons ((a), (b1), (b3), (c), and its block A cells), which are then reported as exploratory.
4. **Re-pulls:** `/models` is recorded at the start and end of each block. If an executor's `updated_at` changes within a block, that executor's block is rerun under the new version and flagged.
5. **Unparsed answers:** count as wrong. Sensitivity analysis excludes them. An executor with more than 10% unparsed in any rule cell moves to exploratory for every test that uses it.
6. **Own-model cells** (A1 x B11, A2 x B2, A3 x B1, A4 x B3, A5 x B5) are excluded from (a) and (d1)-(d2) and analyzed separately (IV.2).

---

## III. Primary Tests

Seven tests, Holm correction at family-wise alpha 0.05. A test **supports** its claim only if it is Holm-significant **and** its effect reaches the smallest effect of interest.

**Notation.** acc(w, b, r): accuracy of executor b on the 250 pairs under draft r of writer w (100 easy pairs for (c)). S̄(b): `S_ref` accuracy of b averaged over its 9 replicates. Pooled gain g(b, r) = mean over w in {A0, A1} of acc(w, b, r) - S̄(b).

### 1. (a) Weaker Executors Gain More

* **Executors:** B0-B10 and B12 (n = 12; B11 excluded as A1's own model).
* **Split:** the 250 pairs are split once into halves H1 and H2, stratified by stratum, seed 0. Generic accuracy S̄(b) is computed on H1; pooled gain (mean over the 9 replicates) on H2.
* **Statistic:** Spearman rho between H1 generic accuracy and H2 gain; one-sided, H1: rho < 0; exact permutation p-value.
* **Ceiling check (required for support):** error ratio on H2, ER(b) = (rule errors + 0.5) / (generic errors + 0.5), pooled over A0, A1 and replicates; Spearman rho between H1 generic accuracy and ER must be > 0 with one-sided p < 0.05 (weaker models have more of their error removed).
* **Smallest effect:** abs(rho) >= 0.6.

### 2. (b1)-(b3) Within A Family, The Smaller Model Gains More

* **Pairs:** (b1) B0 `gemma3:12b` vs B3 `gemma3:27b`; (b2) B5 `qwen3:8b` vs B6 `qwen3:32b`; (b3) B7 `gpt-oss:latest` (20B) vs B2 `gpt-oss:120b`.
* **Statistic:** d(r) = g(small, r) - g(large, r) over the 9 replicates (both executors read the same drafts); one-sided paired t-test (df 8), H1: mean d > 0.
* **Secondary:** draft x pair bootstrap (10,000 resamples) of mean d.
* **Smallest effect:** mean d >= 0.03.

### 3. (c) The Full-Run Result On New Easy Pairs

* **Data:** `ext-easy`, the 100 new easy pairs only; S̄ computed on the same 100 pairs.
* **Statistic:** d(r) = g(B0, r) - g(B1, r); one-sided paired t-test (df 8), H1: mean d > 0.
* **Smallest effect:** mean d >= 0.03.

### 4. (d1) Writer x Executor Interaction

* **Data:** rule cells of all 6 writers x 13 executors, own-model cells excluded; one value per (writer, draft, executor): acc(w, b, r) on the 250 pairs.
* **Model:** split-plot linear model fitted by least squares, acc ~ draft(writer) + executor + writer:executor, with one fixed effect per draft within writer (the whole-plot unit). The residual is executor x draft-within-writer variation, so draft-to-draft variation is the error for the interaction, as in a mixed model with a random draft intercept; the least-squares form handles the missing own-model cells exactly.
* **Statistic:** F test for writer:executor by nested-model comparison (full model vs the same model without writer:executor), numerator df = rank difference, denominator df = residual df of the full model.
* **Smallest effect:** none; (d1) alone does not establish a crossover (it can arise from scale differences with the same writer ranking for every executor).

### 5. (d2) Cost Of A Single Champion Writer

* **Data:** as (d1).
* **Statistic:** cross-validated champion regret. On drafts {0, 2, 4, 6, 8}: for each executor b pick its best writer w*(b) (highest mean accuracy), and the oracle champion w_c (highest mean accuracy averaged over executors, own-model cells excluded). On drafts {1, 3, 5, 7}: regret(b) = acc(w*(b), b) - acc(w_c, b). Swap the roles of the halves (choose on {1, 3, 5, 7}, measure on {0, 2, 4, 6, 8}) and average. R = mean over executors.
* **Test:** one-sided, H1: R > 0. Draft bootstrap: resample the 9 drafts of each writer with replacement (10,000 resamples), keeping each draft's 13 executor values together, recompute R; p = share of resamples with R <= 0. With no crossover the expected R is <= 0, so the test is conservative.
* **Smallest effect:** R >= 0.03.
* **Claim:** "crossover" is stated only if (d2) is supported.

---

## IV. Secondary Analyses (Reported, Not Tested For The Claims)

1. Natural mix (weights 0.094 / 0.042 / 0.864) and per-stratum results for every cell.
2. **Own-model cells:** each writer on its own model vs the same model under the other writers; (d1) and (d2) recomputed with own-model cells included, as a what-if.
3. Per-writer gains for (a)-(c) (A0 and A1 separately).
4. **Optimizer analysis on the full matrix** (Abacus modelled, not run): P0 (executors profiled under `S_ref`) and P9 (champion writer, executors profiled under its rules), with two champions: oracle (as in (d2), on all 9 drafts) and judge (Palimpzest `MAP_VALIDATOR_PROMPT`, `p9_scores.py --style pz`, `gpt-oss:120b`, no temperature; **sample 0 of 3 is the champion**, samples 1-2 measure single-judgment flips; the 18 saved drafts use `results/p9_scores_pz.json`). Outcomes: best plan picked, meets the quality bar, overpay at list prices (`REPORT-C2-ext.md` VI.4, with lowest / highest price sensitivity), whether the judge's champion differs from the oracle's.
5. Tokens (input, output, estimated reasoning) per cell; latency per host, descriptive only.

---

## V. Freeze

* The code commit is recorded with every run invocation. Results of replicates 0-2 may be inspected descriptively before replicates 3-8 finish; all 9 run regardless. Any change to prompts, parser, pair sets, writer drafts or the tests above after that look demotes the affected primary tests to exploratory, and the change is reported.
* Analysis code is committed before the analysis is run on replicates 3-8.
