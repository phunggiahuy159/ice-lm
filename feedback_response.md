# Response to the proposal feedback (NTI review, Oct 14)

Score 5, "meets expectation". The reviewers' four requests and what changed in the code (`ice_lm/`) to answer each:

| # | Feedback | What we do now | Where |
|---|---|---|---|
| 1 | The same phrase can appear with **different continuations**; how reliably does the model pick the right occurrence and **ignore a misleading lookup**? | Two cheap, causal **trust signals** are now given to the model with every lookup: **agree**, whether the two most recent earlier occurrences had the same next token (ambiguous key or not), and **rel**, how often this n-gram order has been right so far in the current document (last 32 predictions). They feed the input gate and the copy bonus, so the model can learn to switch ICE off for unreliable keys. Measured with a held-out loss split by *lookup right / lookup misleading / no match*, plus the **follow rate** P(top-1 = the wrong lookup token). Ablation arm `C_notrust` = the original ICE. | `ice_features.py`, `model.py` (ICEInput/ICEOutput), `evals.py` |
| 2 | **Over-reliance on copying**; recall ≠ reasoning (combining info, revising a value) | (a) Every ICE model is also evaluated with ICE **switched off** (`C_off`): if C_off is worse than B, the model became lazy. (b) Loss on non-copyable tokens is reported separately. (c) CodeRecall has an `override_diff` case (a value is revised with a different syntax, and the lookup points at the stale value), an `increment` case where copying is wrong, and a tiny `track` (state-tracking) probe. | `evals.py`, `coderecall.py`, `scripts/evaluate.py` |
| 3 | Report **actual total memory and speed, including the lookup table** | A real decode path (KV cache, GDN state, conv state) plus an **incremental ICE index**, tested to produce exactly the training features. `scripts/benchmark.py` measures prefill and decode speed, KV / GDN-state / ICE-table bytes, the ICE CPU time per token, and the memory model `M(batch, T) = W + batch·(fixed + per_token·T)` → how many 32K-token sequences fit in 16 / 24 / 80 GB. ICE bytes are reported for a compact layout (2 B/token ids + 16 B per distinct n-gram per order). | `model.py` (`allocate_cache`, `forward_cached`), `inference.py`, `scripts/benchmark.py` |
| 4 | Pick **one concrete use case**: identifiers / config values in code, with **conflicting values, later corrections, and cases where copying is wrong** | **CodeRecall**: template code embedded in held-out real code, answers are 4-digit values with distinct leading digits within an item (scored on the deciding digit), per category and distance (256 / 1K / 3K filler tokens, so items stay within the 4K training length). There is also a realistic-workload split: loss on code tokens whose match is more than 512 tokens back (`copy_far`). | `coderecall.py` |

## CodeRecall categories

| Category | Example | What the exact lookup does | Expected |
|---|---|---|---|
| `copy_same` | `settings = {"db_port": 5432, …}` … `override = {"db_port":` → `5432` | right | ICE helps |
| `copy_diff` | same definition … `assert settings.db_port ==` | no useful match | needs attention: the boundary of ICE |
| `conflict_g{1,2,5}` | `Server(name="dev", region="eu", port=8000)`, `Server(name="prod", region="eu", port=443)` … `Server(name="dev", region="eu", port=` | g=1: the name is within the 8-token lookup, so it is right. g≥2: the name is out of reach, so the lookup returns another server's port (checked with the real tokenizer: 100% right vs 100% wrong) | shows exactly where ICE stops working |
| `override_same` | `TIMEOUT = 30` … `TIMEOUT = 60  # raised` … `TIMEOUT =` | right (recency) | ICE helps |
| `override_diff` | `config["timeout"] = 30` … `config.update(timeout=60)` … `config["timeout"] =` | **stale value**, wrong | the model must ignore ICE |
| `increment` | `Migration(id=1041)`, `(id=1042)`, … `Migration(id=` | wrong on the last digit | copying is wrong |
| `track` | `x = 3`, `x = x + 4`, `print(x)  # prints` | none | tiny reasoning probe |

## Controls that keep the comparison fair

C differs from B in four ways at once: the lookup information, about 2.5% extra parameters (mostly the six
d×d projections W_n), the trust signals, and ICE-dropout. Each has its own control:

| Row | What it isolates |
|---|---|
| A (3:1) | upper reference: twice the attention |
| B (7:1) | reference |
| **B_pm**: B with a wider MLP, same parameter count as C | extra capacity, not information |
| C_notrust | the lookup alone (the original ICE) |
| **C** (pre-registered headline arm) | lookup + trust signals |
| C_off: C evaluated with ICE switched off | laziness (should not be worse than B) |
| F (15:1 + ICE), stretch | memory frontier |

CodeRecall items and all text-task items stay within the training length (4K), so attention is never asked to
extrapolate while ICE is not. Lengths above 4K are reported separately, as extrapolation.

## Run plan (as planned for Kaggle 2×T4; the runs moved to an RTX 3090 and Modal H100, see "Results so far")

0.6B does not fit the Kaggle budget: the first run measured 1.4K tok/s, about 59 h for 300M tokens. The main
scale is therefore **140M**, with the same token budget for every arm (about 300–500M, chosen after measuring
throughput in the first session) and at most 2 sessions per arm:

* **Recipe (default of the `140m` preset):** 400M tokens per arm (each window seen once), batch 131K tokens
  (about 3K steps), lr 1e-3 with 20M warmup tokens, cosine decay to 10%. The same seeds for every arm (paired).
  The final evaluation uses the full 2M-token validation sets.
* **Core:** B ×2 seeds, C ×2 seeds, A, C_notrust, B_pm (7 runs). C_off is free: it is an evaluation of C.
* **Stretch:** F; one 0.6B confirmation (A, B, C) on an H100 if available.
* **Free:** the inference benchmark at 0.6B uses random weights, so it does not need trained 0.6B models.
* **Dropped:** D/E (head-level hybrids) and C_in/C_out/C_nodrop. These are a separate study.
* **Before any training (Day 0):** coverage and precision of the lookup by distance and by trust bucket. This bounds
  the best possible gain and shows that the trust signals predict when a lookup is wrong.

## Result tables to show

1. **Pareto plot:** code loss vs memory per 32K-token sequence (KV + ICE tables) for A, B, C (+ F), with seed
   error bars.
2. **CodeRecall:** deciding-digit accuracy per category for B, B_pm, C_notrust and C, with paired bootstrap CIs
   against B. Categories are ordered from "lookup right" to "lookup wrong", so the boundary of the method is visible.
3. **Harm and cost:** follow rate on misleading lookups (excess over B), C_off vs B, loss on lk_right / lk_wrong /
   lk_none tokens, ICE bytes/token next to KV bytes/token, ICE CPU µs per token.

**Claim if it works:** "With half of A's attention layers, ICE matches A's code loss within X, recovers Y% of B's
deficit on long-range copying, follows misleading lookups no more often than B, and costs Z bytes/token against
the 2 KB/token of KV it saves."

**Honest fallback:** "An exact lookup reliably helps verbatim recall but does not replace an attention layer at this
scale. Here is where it helps, where it misleads, and what it costs."

## Results so far

**Day 0 (no training, real held-out data).** On code, the longest match's next token is right for 46.7% of
tokens. The trust signals separate good lookups from bad ones: 76.8% right when the two most recent occurrences
agree vs 20.3% when they disagree, and 29.6% → 83.2% from the lowest to the highest recent-hit-rate bucket
(prose: 61.1% vs 8.3% for agree/disagree).

**First training run (RTX 3090): 140M, 80M tokens per arm (one fifth of the recipe), one seed, B vs C.**

| | B | C | C_off (C, ICE switched off) |
|---|---|---|---|
| code loss, all tokens | 2.885 | **2.231** | 3.299 |
| code loss, copyable > 512 tokens back | 2.889 | **1.339** | 3.273 |
| code loss where the lookup is wrong | **3.970** | 4.259 | 4.104 |
| prose loss, all tokens | 4.095 | **3.872** | 4.174 |
| follow rate on wrong lookups, code | 0.150 | 0.407 | 0.111 |

* CodeRecall, deciding-digit accuracy vs B (paired bootstrap 95% CI): copy_same +0.44 [+0.36, +0.51],
  conflict_g1 +0.66 [+0.58, +0.72], override_same +0.23 [+0.17, +0.29]; override_diff −0.06 [−0.09, −0.02]
  (the stale-value case: C is misled), conflict_g5 −0.03 [−0.07, +0.01].
* Cost: ICE tables 41 B/token vs 2,048 B/token KV (2%); decode 43.4 vs 46.6 tok/s at 32K (the Python index is
  0.33 ms/step).
* **Two findings against the method.** C follows misleading lookups much more than B, and C_off is worse than B,
  so the model leans on ICE (the over-reliance the review asked about). With 80M tokens, B has not yet learned to
  copy (0% on the repeat task), so the size of the win is probably inflated.

**Running (Modal H100, full recipe: 400M tokens):** B, C and C_notrust, same seed and data order. This answers
whether the gain survives a baseline that has learned to copy, and whether the trust signals reduce following
wrong lookups (early signal at 26M tokens: follow rate 0.47 with trust vs 0.56 without). Next, if over-reliance
persists: a higher ICE-dropout rate (the `ice_dropout` knob; C_nodrop exists as the opposite control).

## Expected Q&A

* *"copy_same is trivial for any copy head; what is the non-trivial win?"* The `copy_far` loss on real code, the
  frontier plot (quality at lower KV), and `override_same` at long distance. Above all: no loss on the categories
  where the lookup is wrong, compared with B.
* *"Is it just extra parameters?"* B_pm.
* *"How many parameters does ICE add?"* About 3.6M at 140M (≈2.5%) and about 10M at 0.6B (≈1.7%).
* *"What does the lookup cost?"* Bytes/token and µs/token from `scripts/benchmark.py`. The index is a Python
  reference implementation, so decode speed is also reported without its CPU time.
* `track` (state tracking) is near chance for every arm at this scale; it is not evidence about reasoning either way.

## Honest limits

* Paraphrased requests (`copy_diff`, paraphrased needles) are outside what an exact lookup can do. ICE can only free
  up attention for them, not solve them.
* At batch 1, weights dominate memory. The saving shows in batch size and long context, so it is reported that way;
  the max-batch numbers are cache-only upper bounds.
* Quality is measured up to 4K tokens (the training length); 32K numbers are memory and speed only.
