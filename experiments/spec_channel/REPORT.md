# Specification Channel: Report

Goal: test whether an optimizer that profiles LLM operators one at a time picks the right plan when one LLM writes the instructions another LLM follows.  
*Requirements: none to read it. How to reproduce: `README.md`. Raw logs: `results/runs/pilot-c1`, `results/runs/pilot-c2`, `results/runs/full-c2`, `results/runs/c1-k40`. Full-run pre-registration: `PREREG-C2.md`. Analysis output: `results/analysis.json`.*  

Pilot run: 2026-10-07. Reanalysis: 2026-10-09 (Section IV.7). Full C2 run and C1 retest: 2026-10-10 (Sections VII, VIII). Framing of H1/H2 and reference results corrected after review: 2026-10-12. Follow-up analysis (real prices, equal-budget estimates, per-stratum quality, Abt-Buy screen): 2026-10-13 (Section IX).  

---

## I. Summary

* **Question:** a semantic query optimizer (for example Palimpzest's Abacus) usually profiles each LLM operator on its own and combines the numbers. When an upstream LLM writes the instructions a downstream LLM follows, is that still safe?
* **Main finding:** no. Under a generic instruction the strong downstream model looks clearly better than the weak one. Under the instructions a real upstream model writes, the two are about equal. An optimizer that profiles the downstream model under a generic instruction pays more for the strong model when the cheap one would have met the quality bar (5-6x under the size proxy, ~1.7x at list prices). Profiling the downstream model under a writer's actual rules, as Abacus does with the best upstream output, avoids this overpay on C2 whichever writer is used, at the cost of missing the quality bar more often (Section IX).
* **Quality vs cost:** the optimizer's pick still meets the quality bar. The mistake shows up as cost.
* **Cost composition:** generated instructions make every downstream call 2.6-6.5x longer. Adding per-operator costs underestimates downstream cost by 67-81%; scaling by instruction length is within about 3%.
* **Two workloads, two behaviors:** in product matching (C2) the interaction keeps its direction in all 3 replicates but is not significant once variation between drafts of the rules is counted (Section IV.7). In city imputation (C1) the generated instructions hurt every plan, and the largest source of variation is which *writing* of the instructions you get, not which writer.
* **Full C2 run (Section VII):** 150 new pairs, 9 replicates, two held-out reference executors. Treating the generic instruction as the cheapest writer, adding a writer removes llama's lead over gemma: an upstream x downstream interaction, large and significant (p < 0.001; pre-registered as "estimability"). The interaction between *which* writer and which executor is not detected (p = 0.34). When a writer is required, the optimizer overpays ~5.6x (size proxy; ~1.7x at list prices) under three of the four reference executors with an oracle quality bar, and under all four with a bar from its own estimate; when it is optional, the reference decides between the correct pick and lost quality (up to 0.15).
* **C1 retest (Section VIII):** 40 stratified writer examples did not help. No writer stated the phone-format convention; every writer plan scores ~0.52 against the simple rule's 0.840. C1 is dropped as a workload.
* **Follow-up (Section IX):** under the test split's natural record mix the rules still shrink llama's lead (0.189 to 0.095-0.121) but llama stays ahead, so the tie under rules depends on the enriched sample. At equal profiling budget, profiling the executor under one writer's rules (Abacus's approach) does not overpay but misses the quality bar in roughly 10-30% of draws. Whether generic + llama is the cheapest acceptable plan depends on llama's price. A second matching dataset (Abt-Buy) was at ceiling and is not pursued.

---

## II. Background

### 1. Inter-Operator Interaction

An optimizer builds a pipeline from LLM operators and picks an implementation (model, prompt, call pattern) for each. Profiling operators separately and combining the scores is only safe if operators are independent. An interaction means the best choice for one operator depends on what an earlier operator did.

### 2. The Specification Channel

Usually an upstream operator hands the downstream operator data. Here it hands over *instructions*: the prompt, rules, or rubric the downstream operator follows. The downstream operator is effectively doing a different task depending on who wrote its instructions, so its quality under a generic instruction may not predict its quality in the real pipeline. The instructions are also pasted into every downstream call, so their length multiplies downstream cost.

---

## III. Setup

### 1. Pipeline

```txt
labeled examples --> [ writer LLM ] --> rules text --> [ executor LLM, once per record ] --> answers
```

The writer runs once per replicate. The executor runs once per record with the writer's rules at the top of its prompt.

### 2. Workloads

| | C1: city imputation | C2: product matching |
|---|---|---|
| data | UniDM data-imputation benchmark, Restaurant, 86 test records | Walmart-Amazon entity matching (HazyResearch `fm_data_tasks`), 80 test pairs |
| task | fill in the missing city | do two listings describe the same product? |
| writer sees | 12 random labeled training records | 8 labeled training pairs (4 matches, 4 hard non-matches) |
| hard part | two sources use different city names (`new york` vs `new york city`, LA neighborhoods); the phone format reveals the source | 30 of 80 pairs are near-identical non-matches (different capacity, color, model number) |
| simple-rule baseline | 0.895 accuracy (area code + phone format) | accuracy 0.74, F1 0.72 on the same 80 pairs (model-number overlap) |

* **Why these two:** both passed four screens before the pilot: the task is not solved by a simple rule, the two writers produce materially different rules, the executor's answers change with the rules, and every record has a label.
* **Dropped:** the UniDM Buy dataset (the missing manufacturer is the first word of the product name in 98% of rows).
* *Note: C1 uses the same data as the SCOPE study but its own two-step pipeline and prompts; it does not reproduce SCOPE's three-module pipeline.*

### 3. Arms

* **Writers:** `gemini-3.5-flash-lite` (Gemini API free tier) and `qwen2.5:72b` (Purdue GenAI).
* **Executors:** `gemma3:12b` (weak) and `llama3.3:70b` (strong), both Purdue GenAI.
* **Generic instruction (`S_ref`):** one line, e.g. "Fill in the missing city for this restaurant record."
* **Cells:** 2 writers x 2 executors, plus each executor under the generic instruction. Six cells.
* **Replicates:** 3, each with freshly written rules. Writers at temperature 0.7, executors at 0.
* **Size:** 1,440 calls (C2) + 1,548 calls (C1), plus 12 writer calls.
* **Logged per call:** full response, input and output tokens, cache fields where reported, latency, retries, wait time, list-price estimate.

### 4. Independence Baseline

Mimics an optimizer that profiles operators one at a time:

* **Executor profile:** accuracy and tokens under the generic instruction.
* **Writer profile:** accuracy of a fixed reference executor (`llama3.3:70b`) using that writer's rules.
* **Combined estimate:** writer profile x (executor's generic score / reference executor's generic score).
* **Policies:** highest estimated quality; or cheapest plan with estimated quality at least 95% of the best.
* **Regret:** the picked plan's real quality and cost vs the real best plan.

---

## IV. Results

### 1. C2: Product Matching (accuracy across 3 replicates)

| | gemma (weak) | llama (strong) |
|---|---|---|
| generic instruction | 0.81 | 0.94-0.95 |
| Gemini's rules | 0.85-0.90 | 0.83-0.88 |
| qwen's rules | 0.78-0.90 | 0.89-0.90 |

* **Rules helped the weak model and hurt the strong one.** gemma learned to reject near-identical non-matches (63% right to 76-89%). llama became too strict and missed real matches (found 93% before, 72-84% after).
* **Interaction (on error rate):** -0.038, -0.138, -0.013 across replicates; pooled -0.063, 95% interval over records [-0.121, -0.004]. Same direction every time, but not significant at the replicate level (Section IV.7).

### 2. C1: City Imputation (accuracy across 3 replicates)

| | gemma (weak) | llama (strong) |
|---|---|---|
| generic instruction | 0.77 | 0.81-0.83 |
| Gemini's rules | 0.72-0.76 | 0.72-0.76 |
| qwen's rules | 0.71-0.79 | 0.56-0.74 |

* **Rules hurt every cell.** No writer discovered the phone-format convention; every LLM plan is below the simple-rule baseline.
* **One writing dominated.** One qwen spec (294 words) pushed llama to answer `new york city` for every New York record (0.56).
* **Interaction:** -0.035, +0.233, +0.023 across replicates. The direction changes; variation between writings is larger than the writer-by-executor effect.

### 3. Executor Ranking

Strong-minus-weak executor accuracy:

| | C2 | C1 |
|---|---|---|
| generic instruction | +0.13 to +0.14 | +0.05 to +0.06 |
| Gemini's rules | -0.03 to +0.01 | -0.02 to 0.00 |
| qwen's rules | -0.01 to +0.13 | -0.23 to +0.04 |

In both workloads the generic-instruction profile overstates the strong executor's advantage in the real pipeline.

### 4. Optimizer Regret

* **Writer optional:** the optimizer picks generic instruction + llama, the real best plan in both workloads. Regret 0. This holds with llama as the reference executor, not with gemma (Section IV.7).
* **Writer required, max quality:** regret 0 to 0.013 accuracy (C2), 0 to 0.070 (C1).
* **Writer required, cheapest above the bar:** in 5 of 6 replicates the optimizer picks llama when gemma would have qualified, at 5.3-6.3x the cost. Every pick met the bar.
* *Note: cost is a proxy (tokens x model parameter count), because Purdue-hosted models have no list price. The direction holds whenever the 70B model costs more per token than the 12B model; the multiplier depends on real prices.*

### 5. Cost Composition

| downstream input tokens per call | C2 | C1 |
|---|---|---|
| generic instruction | 147-162 | 70-81 |
| generated rules | 422-519 | 366-461 |
| naive estimate error (rules ignored) | -67% | -81% |
| composed estimate error (scaled by rule length) | 2.8% | 3.3% |

The composed estimate scales the executor's generic token count by the rules-to-generic ratio measured once on the reference executor.

### 6. Reliability And Latency

* 0 failed calls, 0 retries, 0 unreadable answers across ~3,000 calls.
* Median latency 1.1-1.4 s per call.

### 7. Reanalysis (2026-10-09, same logs, no new calls)

**Replicate-level tests.** A replicate is a fresh draft of the rules, so replicates are the unit of the test (df = 2):

| | C2 | C1 |
|---|---|---|
| interaction (accuracy): mean / t / p | -0.063 / -1.64 / 0.24 | +0.074 / 0.91 / 0.46 |
| interaction, CI95 resampling drafts then records | [-0.150, +0.013] | [-0.039, +0.221] |
| executor gap shrink under Gemini's rules: mean / p | 0.154 / 0.0007 | 0.058 / 0.013 |
| executor gap shrink under qwen's rules: mean / p | 0.092 / 0.16 | 0.132 / 0.23 |

* **Interaction is not significant yet** in either workload once draft-to-draft variation is counted. The record-only interval reported earlier understated the uncertainty.
* **The executor ranking change is significant** for the Gemini writer in both workloads. This is an estimability effect (isolated profiles mispredict the in-pipeline executor gap), reported separately from interaction.

**Reference executor.** The writer profile used `llama3.3:70b` as the reference executor, so plans with llama were estimated exactly by construction. With `gemma3:12b` as reference instead:

* **Writer optional:** the optimizer picks a writer + llama plan instead of generic + llama in all 3 C2 replicates (quality regret 0.05-0.09, below the 95% bar).
* **Writer required:** cost regret disappears (it picks gemma) but quality regret appears (up to 0.038 in C2, 0.233 in C1).
* **Reading:** the reference executor decides whether the error shows up as cost or as quality. The full run adds a third executor used only as reference, so no candidate plan is estimated exactly.

**F1 alongside accuracy (C2).** LLM cells score F1 0.79-0.95; the F1 interaction (-0.043 / -0.120 / -0.012, CI95 [-0.146, +0.021]) matches the accuracy picture. The simple-rule baseline on the same 80 pairs scores accuracy 0.74 / F1 0.72 (model-number overlap). The earlier F1 0.58 was measured on the full Walmart-Amazon test split (2,049 pairs, 9% matches), not on the stratified pilot pairs.

**Prompt caching.** The rules are the shared prefix of every executor call and account for 54-70% (C2) and 65-83% (C1) of executor cost, so prefix caching would cut absolute cost substantially. The cost-regret ratio barely moves (C2: 5.5-6.3x uncached vs 5.3-6.0x with rule tokens at 10% price), because the discount applies to both executors. Purdue GenAI does not report cache use, so the actual discount is unknown.

---

## V. Limits

* **Two tasks, one dataset each.**
* **Cost proxy, not dollars** (Section IV.4).
* **Execution order:** switching executor models on Purdue GenAI costs a 30-50 s reload, so calls were shuffled within executor blocks rather than fully. Quality comparisons are unaffected; latency comparisons between executors are less reliable.
* **C1 headroom:** the simple-rule baseline leaves about 9 of 86 test records for any rules to win.
* **Pilot size:** 80-86 records and 3 replicates estimate the noise; they do not settle effect sizes.

---

## VI. Full Experiment Size

Sized so the observed interaction is detectable (5% significance, 80% power).

*Note: an earlier version proposed running the generic-instruction cells once. llama's generic cells vary across replicates at temperature 0 (0.94-0.95 in C2, 0.81-0.83 in C1), so they run every replicate. The planned C2 run is 150 new pairs x 9 replicates with a held-out reference executor, about 12,150 calls. C1 is to be redesigned before scaling. Both done: Sections VII and VIII.*

| | C2 | C1 |
|---|---|---|
| records | ~150 | ~142 |
| replicates | 9 | 29 |
| calls | ~5,700 | ~16,750 |
| time at ~18 calls/min | ~5.3 h | ~15.7 h |

C1's replicate count comes from the large variation between writings of the rules.

---

## VII. Full C2 Run (2026-10-10)

Pre-registered in `PREREG-C2.md` before launch. Analysis: `analyze.py pilot-c2 pilot-c1 full-c2 c1-k40`.

### 1. Changes From The Pilot

* **Records:** 150 new Walmart-Amazon test pairs (none reused), same mix as the pilot: half matches, ~37% hard non-matches, ~13% easy non-matches.
* **Replicates:** 9, fixed in advance; no extension after looking.
* **Held-out reference executors:** `gpt-oss:120b` (B2) and `gemma3:27b` (B3) are used only to profile writers, never as candidate plans, so no plan is estimated exactly by construction. Chosen by a screen on 24 validation pairs (`ref_screen.py`); four other candidates were rejected (always "yes", unreadable answers, or identical answers under every instruction).
* **Cells per replicate:** {Gemini's rules, qwen's rules, generic instruction} x {gemma, llama, B2, B3} = 12.
* **Tests:** replicate-level t-tests (df = 8) and a two-level bootstrap (drafts, then records). Accuracy and F1; simple-rule baseline on the same pairs.
* **Size:** 16,200 calls, ~16.9 h. 5 calls failed during a provider outage and succeeded on resume; 0 unreadable answers.

### 2. Quality (accuracy, mean over 9 replicates; F1)

| | gemma (weak) | llama (strong) |
|---|---|---|
| generic instruction | 0.793 / F1 0.827 | 0.913 / F1 0.914 |
| Gemini's rules | 0.877 (0.847-0.913) / F1 0.872 | 0.870 (0.807-0.893) / F1 0.853 |
| qwen's rules | 0.838 (0.800-0.887) / F1 0.843 | 0.856 (0.767-0.907) / F1 0.838 |

* **Same pattern as the pilot:** rules lift gemma and pull llama down. Generic instruction + llama is the best plan in 8 of 9 replicates.
* **Simple rule on the same 150 pairs** (model-number overlap): accuracy 0.820 / F1 0.824. Every LLM plan except generic + gemma is above it.

### 3. Tests

| | mean | t / p (df 8) | CI95 (two-level) |
|---|---|---|---|
| H1 interaction (accuracy) | -0.025 | -1.01 / 0.34 | [-0.076, +0.025] |
| H2 executor-gap shrink, Gemini's rules | 0.127 | 13.96 / < 0.001 | |
| H2 executor-gap shrink, qwen's rules | 0.102 | 5.08 / 0.001 | |

* **H1, writer x executor interaction: not significant.** Per-replicate contrasts range from -0.127 to +0.107 (6 of 9 negative); the replicate SD (0.075) is three times the mean. The F1 contrast agrees (CI95 [-0.068, +0.044]). The pilot's -0.063 does not hold at full size.
* **H2, writer present x executor interaction: significant for both writers.** Under the generic instruction llama leads gemma by 0.120; under real rules the gap is -0.007 (Gemini) and +0.018 (qwen). Treating the generic instruction as the cheapest writer (a fixed template), adding a writer removes the downstream ranking: an interaction between the upstream choice and the downstream choice. `PREREG-C2.md` names H2 "estimability"; the test is unchanged. For the optimizer, it means the executor profile measured under the generic instruction does not predict the executor's behavior in the pipeline.
* **What H2 does not separate:** the generic instruction is shorter and less specific than any writer's rules, so H2 shows that detailed rules erase llama's lead, not that a particular writer does. H1 is the test of the latter.

### 4. Optimizer Regret By Reference Executor (H3, 9 replicates each)

| reference | writer optional: true best picked | quality regret mean / max | writer required: true best picked | writer required, cheapest above bar: median cost ratio |
|---|---|---|---|---|
| llama (candidate) | 8/9 | 0.000 / 0.000 | 6/9 | 5.57x |
| `gpt-oss:120b` (held out) | 7/9 | 0.001 / 0.007 | 3/9 | 5.57x |
| gemma (candidate) | 0/9 | 0.045 / 0.107 | 2/9 | 1.00x |
| `gemma3:27b` (held out) | 0/9 | 0.058 / 0.147 | 2/9 | 5.57x |

* **Writer required: overpay under three of four references.** llama, `gpt-oss:120b` and `gemma3:27b` all give a median 5.57x. Only gemma as its own reference avoids it, because it then measures the cheap plan directly.
* **Writer optional: the reference decides.** References that react to rules like llama (no gain) make writers look weak, so the optimizer keeps the generic instruction, which is correct. References that react like gemma (gain) make writers look strong: the optimizer adds a writer even when skipping it is better, and composes that gain onto llama, which actually loses from rules (quality regret up to 0.15).
*Cost ratios here use the size proxy and the true best plan for the bar; Section IX re-prices them, sets the bar from the estimate, and profiles on samples.*

* **Reading:** the reference executor is a hidden design choice. It decides between the correct pick and lost quality when the writer is optional; when the writer is required, the overpay holds for every reference except the cheap candidate itself. The pilot's "writer-optional pick is correct" holds only with a llama-like reference.

### 5. Cost Composition And Caching

* **Naive estimate** (executor profiled under the generic instruction): mean absolute error 67.6%.
* **Composed estimate** (scaled by the rules-to-generic token ratio on the reference): 2.6-2.7% via llama, gemma, `gemma3:27b`; 16.3% via `gpt-oss:120b`, whose chat template adds ~190 tokens per call, so its ratio does not transfer.
* **Caching (priced, not measured):** rules are 59-75% of executor cost. Median overpay with llama as reference: 5.57x uncached, 5.33x with rule tokens at 10% price.

### 6. Reading

* **Established for C2:** adding a writer interacts with the executor choice, so executor profiles under the generic instruction mispredict in-pipeline behavior (H2); when a writer is required the optimizer overpays under three of four references, and when it is optional the reference decides between the correct pick and lost quality (H3); downstream cost composes as rules length x calls.
* **Not established:** an interaction between which writer and which executor (H1). Detecting a 0.05 effect at the observed replicate SD would need about 17 replicates (post-hoc, not used).

---

## VIII. C1 Retest: More Writer Examples (2026-10-10)

### 1. Design

* **Change:** the writer sees 40 training records instead of 12, stratified so every (phone format, city) pair with at least 5 training rows appears 3 times. Both New York conventions (slash phone with `new york`, dash phone with `new york city`) are shown equally.
* **Records:** the 156-record validation split (the test split is kept unused). 3 replicates, 2,808 calls, ~2.7 h.
* **Pass criteria, fixed before running:** (P1) the best writer x executor plan beats the simple rule on the same records (0.840) in at least 2 of 3 replicates; (P2) at least 3 of 6 rules drafts state the phone-format convention.

### 2. Results

| | gemma (weak) | llama (strong) |
|---|---|---|
| generic instruction | 0.731 | 0.784 (0.776-0.788) |
| Gemini's rules | 0.504 (0.487-0.519) | 0.528 (0.506-0.538) |
| qwen's rules | 0.521 (0.506-0.532) | 0.521 (0.513-0.532) |

* **P1: fail, 0 of 3.** Best plan minus rule on the same records: -0.312, CI95 [-0.400, -0.229].
* **P2: fail, 0 of 6.** No draft mentions the phone format.

### 3. Why

| records | n | generic: gemma / llama | rules: gemma / llama |
|---|---|---|---|
| `new york` (slash phone) | 45 | 1.000 / 0.978 | 0.000 / 0.000 |
| `new york city` (dash phone) | 17 | 0.059 / 0.314 | 1.000 / 1.000 |
| all other cities | 94 | 0.723 / 0.777 | 0.67 / 0.69 |

* **Every draft maps area code 212 to `new york city`.** Gemini's drafts call the two spellings "interchangeable"; qwen's name only `New York City`. The executors follow, so all 45 slash-phone records are wrong.
* **The writers noticed both spellings but did not explain them.** More examples made drafts consistent (interaction SD across replicates 0.020, pilot 0.141), not correct.
* **Side observations (3 replicates, not tested):** the executor gap shrinks under rules (Gemini 0.030, p = 0.07; qwen 0.053, p = 0.006). With a writer required, the optimizer picks llama when gemma would do in 3 of 3 replicates (3.4-5.5x cost). Naive cost error 84%, composed 3.2-3.4%.

### 4. Decision

C1 is dropped as a workload: a simple rule beats every LLM plan, and more examples do not let these writers find the convention. The next check of whether the C2 results generalize was a second product-matching dataset (Abt-Buy); it was at ceiling (Section IX.6).

---

## IX. Follow-Up Analysis (2026-10-11 to 2026-10-13)

Offline on the full C2 logs, no new calls, except the Abt-Buy screen (IX.6). Code: `optimizer_policies.py`, `optimizer_budget.py`. Output: `results/optimizer_policies.json`, `results/optimizer_budget.json`.

### 1. List Prices

USD per 1M tokens (input / output), checked 2026-10-12/13: gemma 0.05 / 0.15, llama 0.10 / 0.32 (DeepInfra Turbo FP8), qwen 0.36 / 0.40 (DeepInfra), Gemini 3.5 Flash-Lite 0.30 / 2.50 (Google). llama is ~2x gemma per token, not 5.8x as under the size proxy. The writer-required overpay of Section VII.4 becomes 1.69-1.70x; direction and reference pattern unchanged. Purdue serves different builds, so prices are a what-if on the measured token counts.

### 2. Three Ways To Profile The Executor

| estimate | executor profiled on | gemma | llama |
|---|---|---|---|
| separate profile (Section III.4) | the generic instruction | 0.793 | 0.913 |
| averaged over upstream outputs | both writers' rules | 0.858 | 0.863 |
| best upstream (P9) | one writer's rules (the champion) | 0.861-0.879 | 0.856-0.889 |

*Abacus feeds each downstream operator the output of the highest-quality upstream operator during sampling, scored on the upstream's own output (labels or an LLM validator), then composes per-operator estimates assuming independence. P9ref picks the champion by its score on the reference executor, which knows more than Abacus would; P9A0 / P9A1 fix the champion to Gemini / qwen, bounding whichever writer an upstream-only score picks. Row: P9ref, means over 9 replicates, ranges over the four reference executors (champion is Gemini in 5-8 of 9 replicates).*

* **P9 models what Abacus would do; Palimpzest does not run it today.** The writer runs once over many records, an aggregate in Palimpzest, and its sampler cannot yet optimize LLM operators after an aggregate (`src/palimpzest/query/execution/mab_execution_strategy.py`, `get_frontier_op_inputs`). Value-channel pipelines with a per-record upstream are inside what it optimizes.

* **Averaged over upstream outputs** with two writers is half the candidate plan's own cell. Leaving that writer out makes the picks worse than the separate profile under llama-like references.

### 3. Equal Profiling Budget

Every estimate is profiled on k sampled records, with k = budget / cells it profiles; 500 draws per replicate; truth on all 150 records. llama at the Turbo price, bar = 95% of the true best.

| estimate (reference) | budget (calls) | writer optional: best plan picked | writer required: meets bar | writer required: median overpay |
|---|---|---|---|---|
| separate profile (llama) | 300 / 600 | 0.62 / 0.89 | 0.95 / 1.00 | 1.70x |
| separate profile (`gpt-oss:120b`) | 300 / 600 | 0.58 / 0.73 | 0.80 / 0.78 | 1.70x |
| separate profile (gemma, `gemma3:27b`) | 300 / 600 | 0.01-0.08 / 0.00-0.01 | 0.79-0.89 | 1.00-1.41x |
| averaged, own writer left out (4 refs) | 300 / 600 | 0.26-0.39 / 0.33-0.53 | 0.54-0.82 | 1.00x |
| P9ref, champion via reference (4 refs) | 300 / 600 | 0.56-0.64 / 0.76-0.89 | 0.79-0.93 | 1.00x |
| P9, champion fixed to either writer (2 x 4 refs) | 300 / 600 | 0.42-0.60 / 0.46-0.87 | 0.72-0.94 | 1.00x |
| joint sampling of every plan | 300 / 600 | 0.48 / 0.73 | 0.86 / 0.90 | 1.00x |

* **With the bar from the optimizer's own estimated best,** the separate profile overpays under all four references (1.70-1.71x). The other estimates stay at 1.00-1.04x.
* **Profiling the executor under a writer's rules removes the overpay whichever writer is the champion** (median 1.00x for both fixed champions, every reference, price and bar). The overpay in Section VII.4 comes from profiling the executor under a generic stand-in.
* **It trades the overpay for bar misses.** With the bar from the estimated best at 600 calls, P9ref meets the bar in 79-92% of draws and the fixed champions in 70-93%, against 100% for the separate profile with llama as reference (which overpays 1.7x) and 89% for joint sampling.
* **P9ref's accuracy on the writer-optional decision partly comes from choosing the champion through the reference executor.** With a fixed champion it is at or below joint sampling. A faithful version scores the writer on its own output; not run yet.
* **Targeted checking** (sample only the plans that could change the decision) ends up verifying most writer plans here and performs like joint sampling. It was designed after seeing C2, so this is not a clean test.

### 4. llama Price Range

* At a higher llama price (Together, 1.04 / 1.04, ~10x Turbo input) the separate-profile overpay rises to ~15x.
* **Writer optional, cheapest plan above the bar:** generic + llama in 9 of 9 replicates at the Turbo price; from 2x the Turbo price up, a writer + gemma plan in 6 of 9 (Gemini + gemma costs 0.93x generic + llama at 2x, 0.19x at 10x). The break-even lies between 1x and 2x Turbo.

### 5. Per Stratum And Natural Mix

Test-split mix: matches 9.4%, hard non-matches 4.2% (the top-Jaccard non-matches, as sampled), easy non-matches 86.4%. Sample: 75 / 56 / 19.

| accuracy | matches | hard non-matches | easy non-matches | sample mix | natural mix |
|---|---|---|---|---|---|
| generic + gemma | 0.987 | 0.554 | 0.737 | 0.793 | 0.753 |
| generic + llama | 0.920 | 0.893 | 0.947 | 0.913 | 0.942 |
| Gemini + gemma | 0.843 | 0.927 | 0.865 | 0.877 | 0.866 |
| Gemini + llama | 0.772 | 0.962 | 0.982 | 0.870 | 0.961 |
| qwen + gemma | 0.883 | 0.786 | 0.819 | 0.839 | 0.824 |
| qwen + llama | 0.770 | 0.935 | 0.965 | 0.856 | 0.945 |

* **Mechanism:** under the generic instruction the weaker executors say "yes" too often (gemma 0.69, `gemma3:27b` 0.61; true rate 0.50), and the rules fix that. The stronger ones are balanced (llama 0.51, `gpt-oss:120b` 0.52) and the rules make them stricter (llama recall on matches 0.92 to 0.77), likely because the writers saw 4 matches and 4 hard non-matches.
* **llama minus gemma, generic / Gemini's / qwen's rules:** sample mix 0.120 / -0.007 / 0.017; natural mix 0.189 / 0.095 / 0.121. The rules still shrink llama's lead under the natural mix, but llama stays ahead; the tie in Section VII.2 depends on the enriched sample. The best plan under the natural mix is Gemini + llama (0.961).
* **Caveat:** the easy stratum carries 86% of the natural weight on 19 records. Optimizer decisions were not recomputed under natural weights.

### 6. Abt-Buy Screen (2026-10-11)

Same design on a second matching dataset (`workflows.py ab`, `results/ab_screen.json`, `results/runs/pilot-ab`). Writers wrote different rules and gemma changed 3 of 10 answers under them, but on a 40-pair, 1-replicate smoke run every plan scored 0.90-1.00 (llama generic 0.950, gemma 0.900), failing the pre-set ceiling criterion. With no executor gap the rules have nothing to change; not pursued.

### 7. Reading

* **Established for C2, narrowed:** the writer-present x executor interaction holds in direction under both record mixes. An optimizer that profiles the executor under a generic instruction overpays when a writer is required (~1.7x at list prices, more at higher llama prices) under every reference once the bar is estimated. Profiling the executor under a writer's rules (Abacus's best-upstream approach) avoids it on C2 whichever writer is used, but misses the bar more often.
* **Scope:** one task type (yes/no matching), one enriched record mix, open-weight executors from four families, one price source per model. The results show the effect exists in this setting, not that it holds across Specification workloads or model families.
* **Open:** cases where the best upstream output is not representative for the cheap executor, which is where Abacus-style profiling would fail; more workloads with other task types and natural record mixes; GPT and Gemini executors.
