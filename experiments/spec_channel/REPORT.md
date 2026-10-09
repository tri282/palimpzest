# Specification Channel Pilot: Report

Goal: test whether an optimizer that profiles LLM operators one at a time picks the right plan when one LLM writes the instructions another LLM follows.  
*Requirements: none to read it. How to reproduce: `README.md`. Raw logs: `results/runs/pilot-c1`, `results/runs/pilot-c2`. Analysis output: `results/analysis.json`.*  

Pilot run: 2026-10-07. Reanalysis: 2026-10-09 (Section IV.7).  

---

## I. Summary

* **Question:** a semantic query optimizer (for example Palimpzest's Abacus) usually profiles each LLM operator on its own and combines the numbers. When an upstream LLM writes the instructions a downstream LLM follows, is that still safe?
* **Main finding:** no. Under a generic instruction the strong downstream model looks clearly better than the weak one. Under the instructions a real upstream model writes, the two are about equal. An optimizer trusting separate profiles pays 5-6x more for the strong model when the cheap one would have met the quality bar.
* **Quality vs cost:** the optimizer's pick still meets the quality bar. The mistake shows up as cost.
* **Cost composition:** generated instructions make every downstream call 2.6-6.5x longer. Adding per-operator costs underestimates downstream cost by 67-81%; scaling by instruction length is within about 3%.
* **Two workloads, two behaviors:** in product matching (C2) the interaction keeps its direction in all 3 replicates but is not significant once variation between drafts of the rules is counted (Section IV.7). In city imputation (C1) the generated instructions hurt every plan, and the largest source of variation is which *writing* of the instructions you get, not which writer.

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

*Note: an earlier version proposed running the generic-instruction cells once. llama's generic cells vary across replicates at temperature 0 (0.94-0.95 in C2, 0.81-0.83 in C1), so they run every replicate. The planned C2 run is 150 new pairs x 9 replicates with a held-out reference executor, about 12,150 calls. C1 is to be redesigned before scaling.*

| | C2 | C1 |
|---|---|---|
| records | ~150 | ~142 |
| replicates | 9 | 29 |
| calls | ~5,700 | ~16,750 |
| time at ~18 calls/min | ~5.3 h | ~15.7 h |

C1's replicate count comes from the large variation between writings of the rules.
