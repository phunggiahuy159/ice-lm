# ICE (In-Context Engram): the complete guide

This guide explains the whole project, from the problem to the current results and open questions. It is written so
that someone with no prior context, a person or a model, can answer questions about it. Every number is from
held-out data unless marked otherwise, and each section names the file that holds the details.

Status as of 27 Oct 2026. Repo: `github.com/huypg-zenai/ice-lm` (private). Local copy: `proposal_ice/`.

---

## 1. The problem

**Recurrent language models** (here: Gated DeltaNet, "GDN") keep a fixed-size state instead of attention's KV cache.
Memory and per-token cost stay constant however long the context is. Their weakness is **exact recall**: they blur
details seen long ago, such as a variable name or a config value from 2,000 tokens back. Jelassi et al., *Repeat After
Me* (2402.01032), shows that recurrent models fail at copying. Zoology (Arora et al., 2312.04927) finds that 82% of
the quality gap between attention and attention-free models comes from in-context recall.

**The usual fix is a hybrid:** mostly GDN layers plus a few full-attention layers (here 7 GDN : 1 attention, i.e. 2
attention layers out of 16). Attention restores recall, but its KV cache grows with every token: 2,048 bytes per
token for our hybrid.

**The idea behind ICE:** much of that recall is *exact copying*. If the last n tokens appeared earlier in the
context, the token that followed them then is a strong guess for the next token now. A hash table over token IDs
finds that in O(1), with no learned vectors and no GPU memory. So an exact lookup might replace some or all of the
attention.

### The proposal and the reviewers
- **Original proposal** (`proposal.tex/.pdf`): add ICE to hybrids so they need fewer attention layers. Arms A (3:1),
  B (7:1), C (7:1 + ICE), F (15:1 + ICE). Kill criterion: C does not beat B on copy tasks beyond seed noise.
- **Reviewer feedback** (`feedback.md`; GStar academic team, score 5, "Meet expectation") asked for four things:
  1. **Misleading lookups:** the same phrase can have different continuations. Can the model pick the right
     occurrence and ignore a wrong one?
  2. **Over-reliance:** better copying is not better reasoning, especially when a value is revised later.
  3. **Real cost:** total memory *including the lookup table*, and speed, against the quality change.
  4. **One concrete use case**, e.g. identifiers and config values in code, including conflicting values,
     later corrections, and cases where copying is wrong.

---

## 2. The lookup itself (unchanged across all ICE versions)

Code: `ice_lm/ice_lm/ice_features.py` (training, vectorised NumPy) and `ice_lm/ice_lm/inference.py`
(`ICEIndex`, incremental, for generation).

At every position t (predicting token t+1), for each **order** n in {1, 2, 3, 4, 6, 8}:
- Find the most recent earlier position p < t whose last n tokens equal the last n tokens at t.
- Return the **successor** x[p+1], the **true match length** (how far back the match extends, up to 32), and the
  **distance** t − p (log2 bucket).

The **longest-match successor** is called s*. The lookup works on token IDs only and is causal (p + 1 ≤ t, so it
never sees the future). Matches never cross document boundaries, because features reset at EOS.

Example: the context ends with `config.get(`, and earlier it contained `config.get("lr")`. The lookup proposes `"lr`.

**Day-0 statistic** (real code, no model): s* is right for 46.7% of tokens. It is right 76.8% of the time when the two
most recent occurrences agree, and 20.3% when they disagree. So the lookup is often right, but it is wrong in a
structured way when the context holds conflicting continuations.

---

## 3. Version history: how the model uses the lookup

### 3.1 Old ICE (the reviewed version; arms C, G_C)
The model *reads* the lookup. Formulas from the `ice_lm/ice_lm/model.py` docstring:
- **Input side:** after a middle block, h ← h + Σ_n g_n f_n, where f_n = W_n E[s_n] + L[len_n] + D[dist_n] (+ trust
  embeddings), and g_n = sigmoid(similarity between h and f_n) is a gate.
- **Output side:** a learned bonus on the longest-match token, z[s*] += softplus(wᵀRMSNorm(h) + b[len*] + …).
- **Trust signals:** `agree` (did the two most recent occurrences have the same successor?) and `rel` (how often
  this order was right so far in the document).
- **ICE-dropout:** some training sequences get no ICE, to fight "laziness" (the backbone leaving copying to the
  lookup).

Explainer for this version only: `ice_explained.tex/.pdf`, written 5 Oct and now out of date.

**What happened:**
- **In the 7:1 hybrid (B vs C), ICE's gain vanished with training.** The code-loss gap C − B went from −0.27 early to
  −0.01 at 360M tokens: B's attention learns to copy by itself. The trust signals did nothing (C ≈ C_notrust).
- **In the attention-free model (G_C), ICE helped a lot,** but:
  - where the lookup was wrong it was **0.19 worse than no lookup** (the bonus pushes a wrong token up);
  - with ICE switched off it was **0.43 worse** than a model trained without ICE (strong over-reliance).

**Pivot:** test ICE where it matters, in a model with **no attention at all** (G), and change how the lookup is used.

### 3.2 ICE-V (arm G_V): the lookup proposes, the model verifies
Design: `research/decision.md`, `research/impl_notes.md`. Code: `ICEPointer` in `model.py`.
- **The backbone never sees the lookup.** It produces its own logits z, so p_LM = softmax(z).
- **The lookup proposes K = 12 candidate slots:** 6 orders × {last occurrence, second-to-last occurrence}. Each slot
  has a token c_k, its match length, a distance bucket and a fixed source id.
- **A log-space pointer-sentinel mixture** (after Merity et al., *Pointer Sentinel Mixture Models*, 1609.07843)
  combines them:
  ```
  l      = log_softmax(z)                          (detached inside the scores)
  s_k    = b[src_k] + L[src_k, len_k] + D[dist_k] + γ[src_k] · clip(l[c_k], −20, 0)    candidate k
  s_0    = wᵀ RMSNorm(h_last) + β · max_v l + c0                                       "sentinel": trust the model
  log π  = log_softmax([s_0, s_1..s_K])            (empty slots: −inf)
  p(y)   = π_0 · p_LM(y) + Σ_{k : c_k = y} π_k
  loss   = −log p(y) + 0.5 · CE(z, y)              (the aux CE keeps the backbone usable alone)
  ```
- **The γ·l term is the "verification":** a candidate that the model itself finds implausible gets a low weight.
  A wrong candidate is *down-weighted*, not forbidden.
- With no candidates (or ICE off), p(y) is exactly p_LM(y). ICE-dropout is 0 for the V arms.

### 3.3 ICE-VR (arm G_VR): the current main model
The same verifier with **K = 34 slots**:
- the 12 V slots;
- **deep suffix chain:** occurrences 3–8 of the longest-matching order, each with its own match length;
- **identifier anchors:** the 2 most recent alphabetic tokens in the last 8 tokens. For each, find its latest
  earlier occurrence (order-1 and order-2 keys) and propose the next 4 tokens there (`set_alpha_mask` defines
  "alphabetic"). Slots that would point at the label itself are masked, and this is asserted.

The verifier has 2.7K parameters in total. The compact table measures 82–90 B/token (§5.4).

### 3.4 ICE-V2 (arms G_VR2, `fit_*_v2`): candidates compared with each other (being tested on Kaggle)
Spec: `research/round2/decision2.md` §2 and `research/round2/impl_v2_notes.md`.

**Why:** VR scores each candidate on its own, so when candidates conflict it often picks the wrong one. Where s* is
wrong but the right token is among the candidates, the backbone's log-prob alone ranks the right one first **64.5%**
of the time, but VR's verifier only **40.6%** (§5.6).

**What it adds**, all int8 and 0 extra cache bytes, with tables per slot group g ∈ {last, second, chain, anchor}:

| Feature | Meaning |
|---|---|
| Δ_k | match-length lead over the best *competing* (different-token) occurrence slot, clipped ±8 |
| n_k | how many slots propose the same token (1–6) |
| r_k | recency rank of the token's newest source among the distinct tokens ("latest write wins", generalised) |
| q_k | rank of the backbone's log-prob of c_k among the distinct slot tokens (0–2): a non-linear use of l |

```
s_k = b + L[len] + D[dist] + softplus(γ̃)·l[c_k] + R_g[Δ_k] + A_g[n_k] + N_g[r_k] + Q_g[q_k]
s_0 = wᵀ RMSNorm(h) + c0                        (β dropped; the new tables start at 0, so V2 starts as VR)
loss += 0.25 · 1[y among ≥ 2 distinct slot tokens] · (−log Σ_{c_k=y} softmax_slots(s)_k)    (listwise term)
```

**How it is tested:** *post hoc*. Only the scorer (`ice_ptr`) is fitted on a **frozen** backbone, for 20M tokens
(`ice_lm/scripts/fit_scorer.py --v2 --listwise 0.25 --unseen`). `--unseen` uses only training windows that none
of the backbones saw. Joint training (arm G_VR2) is implemented but not run.

---

## 4. Experimental setup

| | |
|---|---|
| Model | 140M preset: 16 layers, d = 768, 12 heads, MLP 2048. Parameters matched by widening the MLP (B 145M, G_VR 148M) |
| Arms | **G** = 16 GDN, no attention (0 B/token cache). **B** = 7:1 hybrid (2,048 B/token KV). **G_C / G_V / G_VR** = G + old ICE / ICE-V / ICE-VR. `*_off` = the same model evaluated with ICE switched off. **G_MQA** = 15 GDN + 1 one-KV-head attention layer (256 B/token), the equal-memory baseline, not yet trained. All arms in `ice_lm/ice_lm/config.py` (`ARMS`) |
| Data | FineWeb-Edu + 30% code (codeparrot). TinyLlama tokenizer (32K vocabulary) |
| Training | 200M tokens (≈ 1.4 tokens per parameter, short), 4K context, batch 131K tokens (1,526 steps), lr 1e-3, 20M-token warmup, cosine to 10%. H100 on Modal |
| Evaluation | 500 held-out windows of 4K tokens per split (code, prose); `ice_lm/scripts/evaluate.py` |
| Seeds | B and G_VR: 2 seeds. Others: 1 |

Main results: `results_t200/README.md`, `results_t200/results_t200.json`. Timeline of earlier runs: `HANDOFF.md` §3–4.

### Metric glossary (`ice_lm/ice_lm/evals.py`)
The token groups are defined by the lookup alone, from the text, so **every arm is scored on the same tokens**.
- **copyable:** the last ≥ 4 tokens occurred earlier in the window (37.5% of code tokens). The continuation may still
  differ from the earlier one.
- **copy > 512 back** (`copy_far`): copyable, with the matched occurrence more than 512 tokens back (7% of code).
- **lk_right / lk_wrong / lk_none:** s* equals the true next token (47.6% of code) / s* exists but is wrong / no
  match at all.
- **follow wrong lookup:** on lk_wrong tokens, how often the model's top-1 prediction is the wrong s*.
- **top-1 accuracy:** argmax equals the true token. Loss is teacher-forced cross-entropy in nats; perplexity = e^loss.
- **CodeRecall** (`ice_lm/ice_lm/coderecall.py`): template code embedded in real code; the score is accuracy on the
  deciding digit. Categories: copy_same, copy_diff (asked with other wording), conflict_g{1,2,5} (same key under
  several objects), override_same / override_diff (value changed later), increment (copying is wrong).

---

## 5. Results

### 5.1 Main table (code unless noted; lower is better except accuracies)

| | G (no attn) | G_C (old ICE) | G_V | **G_VR** | B (hybrid) |
|---|---|---|---|---|---|
| Code loss | 1.899 | 1.762 | 1.703 | **1.690** | 1.730 |
| Prose loss | 3.444 | 3.379 | 3.339 | **3.324** | 3.384 |
| Copyable | 1.374 | 1.033 | 0.996 | **0.989** | 1.061 |
| Copy > 512 back | 1.680 | 0.966 | **0.926** | 0.933 | 1.132 |
| lk_wrong | 3.215 | 3.408 | 3.251 | 3.211 | **3.172** |
| lk_none | 2.560 | 2.579 | 2.512 | **2.506** | 2.535 |
| Follow wrong lookup | 0.208 | 0.315 | 0.340 | 0.340 | 0.239 |
| ICE off, code loss | – | 2.328 | 2.037 | 2.042 | – |
| Top-1, all code | 62.4% | | | **66.5%** | 66.1% |
| Top-1, copy > 512 back | 64.6% | | | **80.8%** | 76.2% |

**How to read it:**
- **The headline:** with no attention at all, G_VR beats the hybrid at equal training. Code −0.040 (−4%
  perplexity), prose −0.060 (−6%). Of the G→B gap (0.169), G_C closes 81%, G_V 116% and G_VR 124%.
- **Two seeds:** G_VR − B is −0.040 / −0.045 on code and −0.060 / −0.061 on prose. Code loss varies ≤ 0.005 between
  seeds.
- **The lead shrinks during training:** −0.117 / −0.058 / −0.039 at 131M / 164M / 197M tokens (8-window
  training-time evals). The learning rate anneals, so this cannot be extrapolated. G_VR at 164M tokens matches B at
  197M.
- **Harm from wrong lookups** (lk_wrong vs G): old ICE +0.19, V +0.04, VR ≈ 0. The hybrid is still 0.04 better than
  G_VR on these tokens.
- **Follow rate:** the V arms still put top-1 on the wrong lookup 34% of the time (B 24%). The loss there is fine
  because the model *hedges* (spreads probability) rather than refusing.
- **Over-reliance** (ICE off vs G): old ICE +0.43, V/VR +0.14 (0.14–0.17 over the two seeds). Reduced, not removed.
  On no-match tokens, the ICE-trained backbone with ICE off (2.511) beats both G (2.560) and B (2.535). A likely
  cause is loss weighting (the aux CE weights those tokens 1.5×), so treat it as Q&A material, not a claim.

### 5.2 What attention adds is mostly exact copying (the key insight)
Split B − G (attention's gain over pure GDN) by lookup bucket. This is exact, because the buckets are the same tokens
for both models:
- **88% of the code gain (0.149 of 0.169) and 90% of the prose gain fall on lk_right tokens**, which make up 47.6% of
  code tokens.
- **Null:** G's own gain from ordinary training (steps 1000→1500) puts only 45% there, about the base rate.

So attention, added to a recurrent model, mostly buys exact copying, which a 65–90 B/token lookup can supply.
Not claimable: "ICE recovers 133%" (tautological), and "the whole prose gain is copying" (56% of B → G_VR on prose is
elsewhere).

### 5.3 CodeRecall (deciding-digit accuracy, difference vs B, 95% CI)

| Category | G | G_C | G_V | G_VR |
|---|---|---|---|---|
| copy_same | −0.06 | **+0.34** | +0.15 | +0.17 |
| conflict_g1 | −0.01 | **+0.90** | +0.16 | +0.23 |
| override_same | −0.24 | −0.04 | **+0.15** | +0.09 |
| copy_diff | −0.01 | −0.03 | −0.03 | −0.03 |
| conflict_g2 | +0.02 | −0.05 | −0.04 | −0.07 |
| conflict_g5 | −0.02 | −0.10 | −0.10 | −0.10 |
| override_diff | −0.03 | −0.07 | −0.08 | −0.09 |
| increment | +0.01 | +0.86 | 0.00 | 0.00 |

- **Lookup right** (copy_same, conflict_g1): ICE helps. The verifier gave up some of G_C's raw copying power.
- **Lookup wrong in a structured way** (stale values, "which object", other wordings): every ICE arm is ≤ B.
- VR's candidates *contain* the answer (coverage: override_diff 100%, copy_diff 91%, conflict_g5 87%), but the
  verifier does not pick it.
- The pre-registered test VR − V ≥ +0.10 on one structured category **failed**.
- G_C's +0.86 on increment is unexplained and not cited.

### 5.4 Cost (`results_t200/benchmark_t200.json`, `research/round3/index_memory.json`)

| | B | G | G_VR |
|---|---|---|---|
| Memory per sequence at 4K | 10.8 MiB | 3.2 MiB | **3.8 MiB (2.8× less than B)** |
| Memory per sequence at 32K | 66.8 MiB | 3.2 MiB | **6.3 MiB (11× less)** |
| Growth per token | 2,048 B (KV) | 0 | 82–90 B (measured), plus a fixed 250 KiB order-1 table |
| Decode tok/s at 4K / 32K (H100, batch 1) | 87.2 / 86.4 | 84.2 / 85.2 | 80.1 / 78.4 (9% slower at 32K) |

- **The 82–90 B/token is measured:** `ice_lm/scripts/index_memory.py` builds the compact layout as real numpy arrays
  (power-of-two open-addressing hash tables) and checks every lookup against the Python index (0 mismatches). The
  earlier "64.7 B/token" was counted at an ideal load factor.
- The Python dict index actually used in this repo takes 520–640 B/token. That is an implementation artefact; a C or
  GPU index would use the compact layout.
- The slowdown is the Python index on the CPU, about 0.35 ms per decode step.
- Quality is measured at 4K, so **2.8× is the honest headline**; 11× is at 32K, where quality has not been measured.

### 5.5 Exact recall checks (`ice_lm/scripts/recall_check.py`, CPU, n = 10 per cell)

| Repeat task, token accuracy | B | G_VR |
|---|---|---|
| random tokens, full vocabulary (the old synthetic task) | 0.04–0.05 | 0.98–1.00 |
| random frequent tokens (top 5K) | 0.44–0.62 | 0.98–1.00 |
| **real prose repeated verbatim** | **0.58–0.62** | **0.94–0.97** |
| MQAR (Zoology's task), frequent tokens | 0.09–0.19 | 0.61–0.73 |

The hybrid's "0" on the synthetic task is a rare-token artefact: their embeddings barely trained, and attention has to
match through embeddings, while ICE matches IDs. **The fair number is about 60% vs 97% on real text.**

### 5.6 Why conflicts fail (`research/round3/results_cpu3.md` §4.4; fresh windows, pre-registered)
At 96,148 positions where s* is wrong but the right token is among the 34 candidates, the rank-1 rate of the right
candidate is:

| Score | Rank-1 rate |
|---|---|
| backbone log-prob alone | **0.645** |
| G_VR's verifier | 0.406 |
| uniform random | 0.140 |

On ordinary ambiguous positions, the verifier is better (0.868 vs 0.821). A context-dependent mix should beat both.
That is V2's q feature and softplus γ.

### 5.7 Realistic code use case: a stale file version (`ice_lm/scripts/stale_edit.py`, 100 held-out files)
- **Setup:** v1 = the file with 10% of its identifiers renamed (prefix `old_`, `prev_`, …) and 10% of its numbers
  changed (first digit). The model reads v1, then v2 (the original), and v2 is scored.

| v2 tokens | B loss / top-1 | G_VR loss / top-1 |
|---|---|---|
| unchanged code (85,258 tokens) | 0.894 / 80.0% | **0.391 / 92.1%** |
| changed code (3,851 tokens) | **2.520 / 57.1%** | 2.716 / 48.6% |
| first token of a changed name: predicts the *old* name | **13.3%** | 65.0% |

- Both pre-registered checks passed: unchanged better by ≥ 0.10 (−0.503), changed ≤ B + 0.20 (+0.196, just under).
- This is the clearest single example of the trade-off: ICE copies far better, but copies stale values where the
  file changed.

### 5.8 Earlier results (history, `HANDOFF.md` §4)
- **RTX 3090, 80M tokens, B vs C:** C's code loss was 2.231 vs 2.885, with large CodeRecall gains. This was early
  training, and the gain did not survive longer training.
- **H100, 400M tokens, B vs C:** the gap fell from −0.27 to −0.01. Those models were lost (spend limit, corrupted
  downloads); the curves are in `results_h100/`.

---

## 6. Answers to the reviewers, in one place

| Reviewer asked | Answer now | Still missing |
|---|---|---|
| Misleading lookups | Harm vs no lookup 0.19 → ≈ 0 (verifier). Conflicts: unsolved; ICE copies the wrong occurrence (CodeRecall ≤ B; stale-file test copies the old name 65% vs 13%) | V2 (Kaggle) |
| Over-reliance | ICE off: 0.14 worse than G (old ICE 0.43) | Frozen-G + fitted scorer (Kaggle run 3): ICE off is then exactly G |
| Real cost | 3.8 vs 10.8 MiB at 4K, 82–90 vs 2,048 B/token measured; decode 9% slower | An equal-memory attention baseline (G_MQA) |
| Use case | Code identifiers and values: CodeRecall + the stale-file test | Line-completion exact match (Kaggle); a public benchmark (RepoBench-P / LCC) |

---

## 7. What is pending, and the decision rules

Plan and pre-registered predictions: `research/round3/decision3.md`.

### 7.1 Kaggle run (2× T4; the user runs `ice_lm/notebooks/ice_eval_kaggle.ipynb`)
- **Inputs:** the notebook downloads the models and data from the Modal volume `ice-lm`, which needs a Modal token
  (cell 0 or Kaggle secrets).
- **Outputs:** results are uploaded after each job to `kaggle_results/latest` on that volume, with a `SUMMARY.md`
  showing pass/fail for each prediction.
- **Run time:** about 8–11 h.

| # | Job | Prediction / rule |
|---|---|---|
| 1–3 | Re-evaluate G_VR, B, G on T4 | each within 0.006 of H100; if off by > 0.02, it is a bug |
| 4–6 | Line completion (greedy exact match) B, G_VR, G | G_VR − B ≥ +1 pt overall, ≥ +5 on repeated lines; on the slides only if the CI excludes 0 |
| 7 | Gate: refit the old VR scorer on frozen G_VR | within ±0.005 of G_VR, else the fitter is broken |
| 8 | **Run 3:** V2 fitted on frozen G (backbone never saw ICE) | code − B in [−0.025, +0.005]. ≤ −0.02 "beats"; ±0.01 "matches"; > +0.01 needs co-training |
| 9–10 | **Run 2:** V2 on frozen G_VR, 2 seeds | code −0.003 to −0.010, prose ±0.003, follow ≤ 0.32 |
| 11 | **Run 4:** V2 on frozen B | −0.005 to −0.02; otherwise ICE is "attention-free only" |
| 12 | Recall check, n = 50 | B real text 0.55–0.65, G_VR ≥ 0.94 |
| 13 | Run 5: V2 + 2% synthetic conflict items (`syn_conflicts.py`) | off the slides ("task-supervised") |

### 7.2 Modal (workspace hit its spend limit; usable again after 1 Nov, ≤ $30)
- **M1, G_MQA** (≈ $3.5): prediction code 1.74–1.78. If ≤ 1.680, the "beats attention at equal memory" headline is
  dead.
- **M2, B and G_VR to 600M tokens** with a WSD schedule (constant, then decay), with decay branches at 200M / 400M /
  600M (≈ $23). At 600M: ≤ −0.02 "beats", (−0.02, +0.01) "matches", ≥ +0.01 → the fallback claim.
- **M3** (optional): G with CE ×1.5 on no-match tokens, to test the loss-weighting explanation.
- **Next round:** a bolt-on to real pretrained models (Mamba2-370M vs Pythia-410M).

### 7.3 The two possible final claims (decision3 §6)
- **If M1 and M2 work:** "Up to 600M tokens, an attention-free model with an exact lookup and a 2.7K-parameter
  verifier beats both the hybrid and an attention layer with 4× its bytes. Most of what attention adds to a
  recurrent model is exact copying, which the lookup supplies. Stale values remain unsolved by both."
- **Fallback:** "At short training it matches or beats the hybrid with a third of the memory, because 88% of what
  attention adds is exact copying. With more tokens the hybrid catches up: ICE substitutes for copying, not for
  attention."

---

## 8. Limitations (say these when asked)
1. **Short training:** 200M tokens on 140M parameters, and the lead is shrinking. The 600M run decides it.
2. **One scale, one tokenizer, no pretrained model.**
3. **No equal-memory attention baseline yet** (G_MQA).
4. **Conflicts and stale values are unsolved.** ICE confidently copies the wrong occurrence.
5. **Over-reliance is reduced, not removed** (ICE off +0.14).
6. **Mostly teacher-forced loss metrics.** Generation exact match is pending (Kaggle). The public code benchmarks the
   proposal promised (RepoBench-P, LCC, paraphrased needles, retrieval-head formation) are not run.
7. **Speed numbers use a Python index.** Memory uses the measured compact layout, not the Python dicts.
8. **Only B and G_VR have 2 seeds.** CodeRecall varies more across seeds (e.g. G_VR conflict_g1 0.31 vs 0.18).

## 9. What comes from prior work, and what is new
- **From earlier papers:**
  - loss and perplexity, top-1, KV-cache accounting;
  - repeat/copy tasks (Jelassi et al.) and MQAR (Zoology);
  - line-level exact match (as in CodeXGLUE / RepoBench);
  - the pointer-sentinel mixture (Merity et al.).
- **Adapted:**
  - the copyable-token split, in the spirit of Zoology's "AR hits";
  - the 88% decomposition.
- **Designed here:**
  - the lookup-bucket split, harm-from-wrong-lookups and follow rate;
  - the ICE-off test;
  - CodeRecall, the synthetic conflict families, and the stale-file test;
  - the V/VR/V2 candidate sets and their features.
- **Closest related methods:**
  - Akyürek et al., n-gram heads (2401.12973), which still need a per-token vector cache;
  - Engram (2601.07372), a static n-gram table learned from the corpus, where ICE builds its table from the
    current context;
  - Infini-gram (2401.17377), kNN-LM / adaptive kNN-LM (2109.04212), and prompt-lookup / LLMA (2304.04487)
    drafting;
  - HOLA (2607.02303), bounded exact KV for delta-rule models;
  - Retrieval heads (Wu et al., 2404.15574), DuoAttention (2410.10819).
- **Novelty claim (proposal):** to our knowledge, no prior work trains an attention-free, token-ID-only in-context
  n-gram lookup into a recurrent model to replace attention.

## 10. Map of the repo

| Need | File |
|---|---|
| The lookup / candidate slots | `ice_lm/ice_lm/ice_features.py` (docstring) |
| How the model uses it (all formulas) | `ice_lm/ice_lm/model.py` (docstring, `ICEPointer`) |
| Incremental index for generation | `ice_lm/ice_lm/inference.py` (`ICEIndex`) |
| Arms and settings | `ice_lm/ice_lm/config.py` |
| Training (cosine or WSD with branches) | `ice_lm/scripts/train.py` |
| Evaluation | `ice_lm/scripts/evaluate.py`, `ice_lm/ice_lm/evals.py`, `coderecall.py`, `syn_conflicts.py` |
| Scorer fit on a frozen backbone | `ice_lm/scripts/fit_scorer.py` |
| Line completion, recall check, stale-file test, index memory | `ice_lm/scripts/{line_completion,recall_check,stale_edit,index_memory}.py` |
| Main results | `results_t200/README.md`, `results_t200/results_t200.json` |
| Round-3 CPU results | `research/round3/results_cpu3.md` |
| Decisions and pre-registrations | `research/decision.md` (V/VR), `research/round2/decision2.md` (V2), `research/round3/decision3.md` (current plan) |
| Project log, infrastructure, pitfalls | `HANDOFF.md` |
| Slides and the 2-minute script | `slides/ice_gstar.html`, `slides/talk_notes.md` |
| Proposal and reviewer feedback | `proposal.tex/.pdf`, `feedback.md`, `feedback_response.md` |
| Kaggle notebook (and its builder) | `ice_lm/notebooks/ice_eval_kaggle.ipynb` (`build_eval_notebook.py`) |
| Tests (170, CPU) | `ice_lm/tests/` (`ICE_CPU_ONLY=1 python -m pytest -q tests` from `ice_lm/`) |

## 11. Working rules on this project
- **Pre-register a prediction and a decision rule before every run.** Report a fail as a fail.
- **New options must leave old arms bit-identical** (tested).
- **Run CPU tests before GPU runs.**
- **Never write credentials into files.** Modal profile `run3` = workspace tien-phat140205, shared with other
  projects: never touch their volumes or secrets.
- **Ask the user before paid or long runs.** About $30 per round is authorised.
