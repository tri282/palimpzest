# C2 Full Run: Pre-Registered Analysis

Fixed before run `full-c2` starts; the commit containing this file is the freeze point. Pilot results and method: `REPORT.md`.

---

## I. Design

* **Workload:** Walmart-Amazon entity matching, workflow `c2_full`.
* **Records:** 150 test pairs disjoint from the pilot (`results/c2_full_sample.json`, seed 2): 75 matches, 56 hard non-matches (highest token Jaccard among non-matches), 19 random non-matches.
* **Writers (temperature 0.7, 8 fixed labeled training pairs):** A0 `gemini-3.5-flash-lite`, A1 `qwen2.5:72b`.
* **Candidate executors (temperature 0):** B0 `gemma3:12b`, B1 `llama3.3:70b`.
* **Held-out reference executors (profiling only, never candidate plans):** B2 `gpt-oss:120b`, B3 `gemma3:27b`, chosen by a screen on 24 validation pairs (`ref_screen.py`, `results/ref_screen.jsonl`).
* **Cells per replicate:** {A0, A1, `S_ref`} x {B0, B1, B2, B3} = 12. `S_ref` is the one-line generic instruction.
* **Replicates:** 9. A replicate is a fresh spec draw from each writer; every cell runs in every replicate.
* **Size:** 150 x 12 x 9 = 16,200 executor calls plus 18 writer calls.
* **No extension:** if results are inconclusive at 9 replicates, they are reported as such; replicates are not added after looking at results.

---

## II. Tests

Loss = 1 - accuracy. Alpha 0.05, two-sided.

### 1. H1: Interaction

* **Statistic:** per-replicate `I_r = L(A0,B0) - L(A0,B1) - L(A1,B0) + L(A1,B1)`.
* **Test:** one-sample t-test on `I_1..I_9` against 0.
* **CI95:** two-level bootstrap (resample replicates, then records within each).
* **Secondary:** sign agreement across replicates; the same contrast on 1 - F1 with its two-level bootstrap CI, not tested.

### 2. H2: Estimability

* **Statistic:** executor gap `acc(·, B1) - acc(·, B0)` under `S_ref` minus the gap under each writer's spec, per replicate.
* **Test:** one-sample t-test on the per-replicate difference against 0, separately for A0 and A1.

### 3. H3: Regret Of The Independence Baseline

* **Estimate:** `acc(a, b) ≈ acc(a, ref) x acc(S_ref, b) / acc(S_ref, ref)` for each reference ref in {B1, B0, B2, B3}; executor cost profiled under `S_ref`.
* **Pools:** all six candidate plans (writer optional), and the four factorial plans (writer required). Writer-optional reported first.
* **Policies:** max estimated quality; min estimated cost with estimated quality >= 95% of the pool's best true quality.
* **Reported:** quality regret and cost regret per replicate and reference.

### 4. Cost Composition And Cache Sensitivity

* **Composition:** downstream input tokens per call, naive (`S_ref` profile) vs composed (scaled by the spec-to-`S_ref` ratio on each reference).
* **Cache sensitivity:** cost and cost regret with spec tokens priced at k x the input rate, k in {1, 0.5, 0.25, 0.1}. Cost unit: tokens x model parameter count (relative proxy).

### 5. Metrics And Baseline

* Accuracy and F1 for every cell, and for the lexical baselines (`em_lexical_baseline.py` rules) on the same 150 pairs.

---

## III. Analysis Code

`analyze.py full-c2` at the freeze commit produces every quantity above. Changes after the freeze are listed in the results with their reason.
