# Brief: ICE (In-Context Engram) — state of the project, 26 Oct 2026

## The method
A hybrid LM (Gated DeltaNet linear-recurrent layers + a few full-attention layers; arm B = 7 GDN : 1 attention,
arm A = 3:1) loses exact long-range recall. ICE adds an **exact in-context n-gram lookup on token IDs**: for each
position t and each order n in {1,2,3,4,6,8}, find the most recent earlier position p where the last n tokens
match, and return the successor token x[p+1] ("what came next last time"). No attention, no KV: a hash table of
token IDs on the CPU (~41 B/token compact vs 2,048 B/token KV at 140M).

How it enters the model (arm C):
* input side (after layer 2): per order, embedding of the successor token + match-length bucket + distance bucket,
  gated, added to the residual stream;
* output side: a learned bonus added to the logit of the longest-match successor s* (bias depends on match length
  and trust buckets);
* **trust signals** (added after reviewer feedback): `agree` (do the two most recent occurrences of the n-gram have
  the same successor: none/single/agree/disagree) and `rel` (online hit rate of that order over the last 32
  predictions in the document, 5 buckets);
* ICE-dropout: 20% of training sequences see "no match" everywhere.
Ablations available in code: C_notrust (no agree/rel), C_in / C_out (one side only), C_nodrop, B_pm (B with C's
parameter count), F (15:1 + ICE).

## Reviewer feedback (score 5/10 "meets expectation") asked for
1. robustness to the same phrase with different continuations / misleading lookups;
2. over-reliance on copying; recall is not reasoning;
3. real memory + speed including the table;
4. one concrete use case (identifiers/config values in code, with conflicts, corrections, cases where copying is wrong).

## Evidence so far
Day 0 (no training, real code): longest-match successor right on 46.7% of tokens; 76.8% when the two latest
occurrences agree vs 20.3% when they disagree; by recent hit rate 29.6% → 83.2%. Prose: 15% right overall.

3090 run, 140M, 80M tokens (1/5 recipe), 1 seed — B vs C:
| | B | C | C_off (C with ICE switched off at eval) |
| code loss all | 2.885 | 2.231 | 3.299 |
| code loss, copyable >512 tokens back | 2.889 | 1.339 | 3.273 |
| code loss where lookup is wrong | 3.970 | 4.259 | 4.104 |
| P(top-1 = wrong lookup token), code | 0.150 | 0.407 | 0.111 |
CodeRecall (deciding digit) vs B: copy_same +0.44, conflict_g1 +0.66, override_same +0.23, override_diff −0.06
(stale value: misled), conflict_g5 −0.03 (lookup returns another object's value).

H100 runs (full recipe 400M tokens; B, C, C_notrust, same seed), held-out at ~105M tokens:
| | B | C | C_notrust |
| code loss all | 2.425 | 2.157 | 2.166 |
| code copyable | 1.918 | 1.299 | 1.317 |
| code lookup-wrong loss | 3.786 | 4.014 | 4.035 |
| follow wrong lookup | 0.196 | 0.356 | 0.370 |

## Takeaways
* ICE clearly helps where the lookup is right, at 2% memory; the gap to B shrinks as B learns to copy (0.65 → 0.27).
* **Trust signals barely help** (C ≈ C_notrust).
* **Over-reliance**: C follows wrong lookups ~2× as often as B, is worse than B where the lookup is wrong, and C with
  ICE off is worse than B (the model leans on ICE).
* An exact lookup cannot do paraphrase, cross-wording reference, or "which object's value" beyond the n-gram span.

## Constraints for any new method
* Codebase: PyTorch, ice_lm/ (features computed on CPU in the data loader, model.py ICEInput/ICEOutput, GDN via
  fla kernels). 140M preset; ~1.3 H100-hours per 400M-token arm; budget roughly 5–10 more arm-runs.
* Must keep the core selling point: much less memory than KV (bytes/token), causal, works at 32K+ context.
* Must produce a clean, falsifiable comparison against B, C and A within ~1–2 days of work.
* Student project presented to faculty; novelty + clear story matter.
