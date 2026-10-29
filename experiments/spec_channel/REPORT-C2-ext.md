# C2 Extension: Report

Goal: test whether the best downstream LLM depends on which upstream LLM wrote its instructions, across 6 writers and 13 executors, and what that costs an optimizer that picks them one at a time.  
*Requirements: `REPORT.md` VII and IX (the full C2 run). Pre-registration: `PREREG-C2-ext.md`. Analysis: `analyze_ext.py`, output `results/analysis_ext.json`. Raw logs: `results/runs/{drift-c2, ext-easy, ext-models-*, ext-writers-*, p9-scores-ext}`.*  

Status: **primary analysis done 2026-10-29** on 8 executors. B8 and B9 still running, reported later as a supplement; B6 stopped (VI.1).

---

## I. Summary

* **The best executor does depend on the writer.** The interaction is clear: each executor has its own best writer, and four different writers each come out on top for some executor (V.1).
* **For accuracy, ignoring it costs almost nothing.** Picking one champion writer for every executor loses far less than the smallest effect we pre-registered as meaningful (V.2).
* **For cost, it can matter.** The cheapest executor needs its own best writer to clear a tight quality bar; with the shared champion, an Abacus-style pick moves to a pricier executor and pays about 1.5x (V.3).
* **Abacus picks a near-best plan; separate profiling can miss.** Abacus fixes the best writer, then profiles executors under its rules, and lands next to the best plan. Profiling each operator alone picks the executor that looks best without rules, and that executor gets worse under them (V.3).
* **Abacus's remaining loss is its judge.** The judge ties several writers at the top score; the tie rule costs more than the interaction does (V.3).
* **Output format, not the parser, decides which models count.** Two executors reason before answering under rules and fail the strict parser; a lenient parser brings one back, and the conclusions do not change (V.4).
* **Writer vs no writer is a prompt choice, not an operator interaction.** Rules help the weakest executor and hurt the strongest ones; these pre-registered tests are reported, but they belong to prompt search, not to this question (V.2, II.1).

---

## II. Context

### 1. Question

* **Setup:** an upstream LLM (the **writer**) turns 8 labeled examples into matching rules; a downstream LLM (the **executor**) follows them on every record pair.
* **H1:** does the best executor depend on which writer wrote the rules? If yes, an optimizer that profiles the executor under one writer's rules can pick the wrong plan.
* **Not the question:** writer vs no writer. A one-line generic instruction is not a writer implementation, so comparing against it is a prompt choice, which an optimizer handles by profiling each executor under the prompt it will run with (group review, 2026-10-29).

### 2. What The Full C2 Run Found (`REPORT.md` VII, IX)

* Rules erased the strong executor's lead: `llama3.3:70b` beat `gemma3:12b` by 12 points under the generic instruction (0.91 vs 0.79) and tied it under rules (about 0.87).
* H1 was not detected with 2 writers and 4 executors (p = 0.34).

### 3. Why Extend

* 2 writers and 4 executors cannot show a crossover. This run uses 6 writers and 13 executors.

---

## III. Analysis

Fixed in `PREREG-C2-ext.md` before the first replicate; code committed before it touched the data.

### 1. Tests

| test | question | how | supports H1 if |
|---|---|---|---|
| **(d1)** | is there a writer x executor interaction? | split-plot F on accuracy, drafts within writer as the noise term | significant |
| **(d2)** | does it cost accuracy to ignore? | regret of one shared writer vs each executor's own best writer (below) | regret > 0.03 and significant |
| (a), (b1-b3), (c) | do weaker executors gain more from rules? | rank correlation; paired t on size pairs and on easy pairs | (writer present vs absent, II.1) |

* **Decision:** Holm over the 7 tests at 0.05; smallest effect of interest 0.03 accuracy.
* **Why (d2) too:** (d1) only says rankings differ somewhere, even by a hair; (d2) asks whether they differ enough to matter.

**(d2) in plain terms:**  

* **Shortcut:** pick the one writer that is best on average over all executors (the champion), then give every executor that writer's rules.
* **Full search:** pick each executor's own best writer.
* **Regret R:** how much accuracy the shortcut loses against the full search, averaged over executors.
* **No luck allowed:** writers are chosen on drafts 0, 2, 4, 6, 8 and scored on drafts 1, 3, 5, 7 (then swapped). Without this, the full search always wins by picking whichever writer drew a lucky draft.
* **Reading:** R near 0 means the shortcut is safe; R >= 0.03 means an optimizer must search writer and executor together.

### 3. Optimizers Modelled

Neither runs here: Palimpzest cannot yet optimize an operator that follows an aggregate (the writer turns all examples into one output, so its output cannot be traced to single records for sampling). Both are replayed on the measured matrix (`optimizer_picks_ext.py`; descriptive, not pre-registered) under two goals: the highest-accuracy plan, and the cheapest plan within a quality bar.

| optimizer | how it scores the writer | how it scores the executor | then picks |
|---|---|---|---|
| **Separate profiles** (one operator at a time) | accuracy of a fixed reference executor under each writer's rules | accuracy under the one-line generic instruction | best writer + best executor |
| **Abacus** (best upstream) | its validator (an LLM judge) scores each writer's rules, no executor involved | accuracy under the champion writer's rules | champion + best executor under it |
| **Full search** (reference) | every writer x executor pair measured | | best pair |

* **Abacus judge:** Palimpzest's default validator, one score per draft. It often ties several writers at the top score; the tie is broken toward the best, the first (operator order) or the worst writer (pre-registered amendment 1).

### 2. Rules Of The Analysis

* **Executors:** primary if complete (all writers, 9 replicates, 250 pairs) and at most 10% unparsed in every rule cell; computed from the data.
* **Parser:** first word Yes / No (pre-registered, primary). A lenient parser (also reads a final `Answer:` line or `\boxed{}`) is a sensitivity analysis only.
* **Own-model cells** (a writer executing its own rules) are left out of (d1), (d2) and the champion: a model helping itself most is a crossover by construction.
* **Failed calls** count as wrong.

---

## IV. Pipeline And Models

### 1. Pipeline

```txt
8 labeled pairs (4 matches, 4 hard non-matches)
  -> WRITER (temperature 0.7): matching rules, <= 250 words       1 draft per replicate
  -> EXECUTOR (temperature 0): rules + record pair, "Answer Yes or No only"
  -> parsed Yes / No, scored against the label
Baseline S_ref: the executor with a one-line generic instruction
```

* **Data:** Walmart-Amazon test split, 250 pairs per executor (75 matches, 56 hard non-matches, 119 easy non-matches).
* **Replicates:** 9 fresh drafts per writer, fixed in advance.

### 2. Models

| id | writer | executor | model | type | host | in primary analysis |
|---|---|---|---|---|---|---|
| A0 | yes | | `gemini-3.5-flash-lite` | | Gemini API | |
| A1 / B11 | yes | yes | `qwen2.5:72b` | plain | Purdue | **yes** |
| A2 / B2 | yes | yes | `gpt-oss:120b` | reasoning | Purdue | **yes** |
| A3 / B1 | yes | yes | `llama3.3:70b` | plain | Purdue | **yes** |
| A4 / B3 | yes | yes | `gemma3:27b` | plain | Purdue | **yes** |
| A5 / B5 | yes | yes | `qwen3:8b` | reasoning | Chameleon | **yes** |
| B0 | | yes | `gemma3:12b` | plain | Purdue | **yes** |
| B4 | | yes | `gemma4:26b-a4b` | plain | Chameleon | **yes** |
| B7 | | yes | `gpt-oss:latest` (20B) | reasoning | Purdue | **yes** |
| B10 | | yes | `phi4:latest` (14B) | plain | Chameleon | no: format (VI.2) |
| B12 | | yes | `llama4:latest` (Scout) | plain | Purdue | no: format (VI.2) |
| B6 | | yes | `qwen3:32b` | reasoning | Purdue | no: stopped (VI.1) |
| B8 | | yes | `deepseek-r1:14b` | reasoning | Purdue | later: supplement |
| B9 | | yes | `mistral:latest` (7B) | plain | Purdue | later: supplement |

*Note: Chameleon = self-hosted Ollama on one GPU node; Purdue = Purdue's GenAI API. B6, B8 and B9 were planned on Chameleon and moved to Purdue by the pre-registered host check (VI.3).*

---

## V. Results

Primary parser, 8 executors.

### 1. H1: Best Writer Per Executor

Mean accuracy over 9 drafts. **Bold:** each executor's best writer. †: own-model cell (excluded).

| writer | B0 | B1 | B2 | B3 | B4 | B5 | B7 | B11 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 0.905 | 0.915 | 0.930 | **0.914** | **0.929** | **0.884** | 0.908 | 0.934 | **0.915** |
| A1 | 0.856 | 0.905 | 0.912 | 0.875 | 0.922 | 0.861 | 0.890 | 0.923† | 0.889 |
| A2 | 0.800 | **0.919** | 0.871† | 0.884 | 0.926 | 0.868 | 0.872 | 0.922 | 0.884 |
| A3 | 0.911 | 0.891† | **0.932** | 0.895 | 0.926 | 0.882 | **0.914** | **0.937** | 0.914 |
| A4 | **0.922** | 0.905 | 0.920 | 0.902† | 0.927 | 0.877 | 0.904 | 0.930 | 0.912 |
| A5 | 0.898 | 0.898 | 0.907 | 0.893 | 0.927 | 0.883† | 0.899 | 0.926 | 0.907 |
| `S_ref` | 0.788 | 0.924 | 0.952 | 0.864 | 0.924 | 0.932 | 0.932 | 0.931 | |

* **Crossover is real but shallow.** Four writers are each best somewhere; each executor's best writer beats the champion (A0) by at most 0.017 (B0).
* **Rules hurt the strongest executors:** B1, B2, B5 and B7 do best under `S_ref` (writer vs no writer, II.1).

### 2. Tests

| test | result | Holm p | supported |
|---|---|---:|---|
| **(d1) interaction** | **F = 5.37** (df 30 / 296) | **< 0.001** | **yes** |
| **(d2) cost of one champion** | **R = 0.0005** (95% CI -0.005 to 0.010) | 0.72 | **no** |
| (a) weaker executors gain more | rho = -0.63 (7 executors) | 0.28 | no |
| (b1) `gemma3` 12B vs 27B | +0.061 | < 0.001 | yes |
| (b2) `qwen3` 8B vs 32B | not computable (B6 stopped) | 1 | no |
| (b3) `gpt-oss` 20B vs 120B | -0.002 | 1 | no |
| (c) B0 vs B1, easy pairs | +0.092 | < 0.001 | yes |

* **H1: rankings differ, but not enough to matter.** (d1) finds the interaction; (d2) finds a shared writer loses 0.0005, 60x below the 0.03 threshold. Same with own-model cells included.
* **Weak executors need rules, strong ones do not.** `gemma3:12b` gains 0.061 more than its 27B sibling (b1) and 0.092 more than `llama3.3:70b` on easy pairs (c). Across all executors (a) the trend is there (rho = -0.63) but 7 executors are too few to confirm it.

### 3. What The Optimizers Pick

Plans scored by mean accuracy over 9 drafts; regret = best plan minus picked plan (III.3).

| optimizer | picks (writer x executor) | accuracy | regret |
|---|---|---:|---:|
| full search | **A3 x B11** (`llama3.3:70b` writes, `qwen2.5:72b` executes) | **0.937** | 0 |
| Abacus, true champion (A0) | A0 x B11 | 0.934 | **0.003** |
| Abacus, judge (tie best / first / worst)* | varies: A0, A2, A3 or A5 with B1, B2, B4 or B11 | | 0.003 / 0.012 / 0.017 |
| separate profiles, writer scored on B3, B4 or B5 | A0 x B2 | 0.930 | 0.007 |
| separate profiles, writer scored on B2, B7 or B11 | A3 x B2 | 0.932 | 0.005 |
| separate profiles, writer scored on B0 | A4 x B2 | 0.920 | 0.017 |
| **separate profiles, writer scored on B1** | **A2 x B2** (`gpt-oss:120b` runs its own rules) | **0.871** | **0.066** |

*\*The judge scores each replicate's drafts separately, so its pick and regret are per replicate (against that replicate's best plan), averaged over 9.*

* **The top is flat.** The 5 best plans are within 0.007 of each other; 3 of them use B11.
* **Abacus is near-optimal because one writer is good for everyone.** A0 is within 0.017 of every executor's own best writer (V.1), so fixing it first costs 0.003. This is what (d2) measures, and why it is not supported.
* **Abacus's loss comes from its judge, not from the interaction.** The judge cannot separate the top writers, so the tie rule decides; the worst tie costs 6x the true champion.
* **Separate profiles always pick B2 (`gpt-oss:120b`)**, the best executor under the generic instruction (0.952). Under rules B2 drops to 0.871-0.932, so its pick depends on which writer comes with it: up to 0.066 lost. This is the writer-present effect (II.1), not H1: separate profiling never sees the executor under real rules.

**With cost:** cheapest plan whose estimated accuracy is within 95% (the full run's bar) or 98% of the best estimate. List prices on measured tokens, USD per 1k records; one writer draft per 1k records. Every pick below clears its bar on true accuracy.

| optimizer | 95% bar (0.890) | cost | 98% bar (0.918) | cost |
|---|---|---:|---|---:|
| full search | A5 x B0 | $0.020 | **A4 x B0** | **$0.021** |
| Abacus, true champion (A0) | A0 x B0 | 1.14x | **A0 x B4** | **1.49x** |
| separate profiles (any reference) | A4 x B4 | 1.47x | A4 x B2 | 1.78x |

* **Cost is where the interaction shows.** The cheapest executor, B0 (`gemma3:12b`), has the largest crossover: 0.922 under its own best writer (A4), 0.905 under the champion (A0). At the tight bar A0's rules push B0 below it, and Abacus pays 1.49x for B4.
* **The writer sets the executor's bill.** Rules are part of every executor prompt, so long rules cost on every record: the same executor costs up to 1.7x more under A2 (longest rules) than under A5 (B0: $0.035 vs $0.020).
* **Separate profiles always overpay (1.47-1.78x)**, close to the full run's 1.7x: under the generic instruction B0 looks too weak (0.788) and every executor looks cheap, so the cheap-and-good plan is never considered.

### 4. Parser Sensitivity

| | primary (8 executors) | lenient (+ B10, B12) |
|---|---|---|
| (d1) | F = 5.37 | F = 3.93 |
| (d2) | R = 0.0005 | R = 0.0015 |
| champion (best writer on average) | A0 | A3 |

* Same conclusion. Neither parser is Palimpzest's (VII.7).

---

## VI. Other Results

### 1. Runs And Outages

| date | event | effect |
|---|---|---|
| 2026-10-20 | runs start: Purdue (2 keys) + Chameleon GPU node | |
| 2026-10-21 | Purdue gateway errors, about half a day | runners resumed on their own |
| 2026-10-22 | Chameleon executors done (B4, B5, B10) | |
| 2026-10-24 to 26 | Purdue shared queue full (`server busy`) | most models unreachable 2+ days |
| 2026-10-26 | first CPU node reclaimed at lease end | resumed on a new node from a log copy |
| 2026-10-29 | Purdue stalls again; scope cut to H1 | **B6 stopped** (about 0.2 calls/min, days left), B9 started, B8 continues |

### 2. Model Failures

| executor | what failed | calls | treatment |
|---|---|---:|---|
| **B10** `phi4` | reasons first under rules, ends `**Answer: Yes**`; up to 32% unparsed per cell | about 2.2k of 13,500 | exploratory; primary under lenient parser |
| **B12** `llama4` | narrates steps, ends `\boxed{Yes}`; 92% unparsed under A2 | about 2.7k of 13,500 | exploratory under both parsers |
| B5 `qwen3:8b` | reasoning loops past the 120 s timeout (same pair, "identical model numbers" rule) | 8 of 15,750 | counted wrong; not retried (non-deterministic) |
| B7 `gpt-oss:latest` | Purdue `CUDA error: the function failed to launch` on the same requests (4), timeout (1) | 5 of 15,750 | counted wrong; retried once, failed again |

* B10 and B12 answer one word under `S_ref`; only rules trigger the format.

### 3. Checks

| check | result |
|---|---|
| rate limit | per key: 18.7 calls/min on one key, 46 on two |
| host check (Chameleon vs Purdue, 40 pairs) | B4 39, B5 40, B10 40: self-hosted; **B6 37, B8 36, B9 36: moved to Purdue** |
| `gemma4` mode | thought by default on Chameleon (350 vs 2 tokens); fixed by `reasoning_effort: "none"`, rerun 39 / 40 |
| merge | 250 pairs x 9 in every primary cell; re-scoring reproduces logged scores |

---

## VII. Limitations

1. **Fewer executors than planned:** 8 of 13 primary. (a) has little power; (b2) is lost.
2. **Primary analysis ran before B8 and B9 finished.** Their data is a supplement and never replaces it.
3. **Not blind:** a descriptive look at about 75% of the data (2026-10-26) preceded the committed analysis code; nothing was changed after it.
4. **Optimizer replay is in-sample.** Profiles use all 9 drafts; the bar is set from the true best plan; dollar costs use list prices, not what was paid (Purdue and Chameleon are free).
5. **Abacus is modelled, not run.** The judge is a stand-in (`gpt-oss:120b`) at the provider's default temperature, so a single judgment flips often (8 of 18 full-run drafts).
6. **Mixed hosts:** a host is part of an executor's identity; Purdue exposes tags, not builds or digests.
7. **Parser:** this experiment asks for Yes / No and reads the first word; Palimpzest's filter asks for TRUE / FALSE after `ANSWER:`.
8. **Code deviations from the freeze:** `gemma4` request field (before any B4 data) and a Purdue lock for large models (orchestration only). The first CPU node ran from a modified checkout, contents lost with the node; re-scoring with the committed parser matches every stored score.
9. **Scope:** one dataset, one task, one generic instruction; B0-B3's full-run pairs are about 10 days older than the rest (drift check passed).
