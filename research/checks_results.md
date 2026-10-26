# Checks D1 and D3: results (2026-10-26)

**Scripts:** `scripts/coverage_gate.py` and `scripts/long_context_eval.py`. They were written and tested in the frozen copy and then copied to `/Users/phunghuy/code/gstar/proposal_ice/ice_lm/scripts/`. No library file was changed.

**Outputs:**
- On the Modal volume: `/vol/coverage_gate.json` and `/vol/long_ctx_r3090.json`.
- Local copies: `/private/tmp/claude-501/-Users-phunghuy-code-gstar/a1fcb2a4-ae79-48f6-b6e8-350e5f8c37f0/scratchpad/checks_out/`.
- Full logs: `/private/tmp/claude-501/-Users-phunghuy-code-gstar/a1fcb2a4-ae79-48f6-b6e8-350e5f8c37f0/scratchpad/cov_full.log` and `/private/tmp/claude-501/-Users-phunghuy-code-gstar/a1fcb2a4-ae79-48f6-b6e8-350e5f8c37f0/scratchpad/lc_full.log`.

## D3: VR coverage gate (CPU, no model)

**Scope:**
- CodeRecall items are exactly the ones `run_coderecall` builds: same seeds, the TinyLlama tokenizer, gaps 256/1024/3072, n=100 per (category, gap). `increment` and `track` use the first gap only.
- The scored position is the deciding token.
- Held-out code uses 300 packed 4K windows of val_code (1.23M positions), with documents separated at EOS boundaries as in `eval_loss_splits`.

**Columns:**
- Each cell is the share of positions where the correct token is in the set.
- |·| is the mean number of distinct candidates.
- dg· is the mean number of distinct **digit** candidates. This shows how close a set comes to covering a deciding digit by chance.

| category | s* | V | VR | VR−V | V+chain | V+anchor | \|V\| | \|VR\| | dgV | dgVR |
|---|---|---|---|---|---|---|---|---|---|---|
| copy_same | .897 | .993 | 1.000 | +.007 | 1.000 | .993 | 3.96 | 8.79 | 3.5 | 4.2 |
| **copy_diff** | .030 | .053 | .913 | **+.860** | .167 | .907 | 1.53 | 8.95 | 0.6 | 3.0 |
| conflict_g1 | 1.000 | 1.000 | 1.000 | 0 | 1.000 | 1.000 | 3.86 | 8.31 | 3.0 | 4.4 |
| conflict_g2 | 0 | .220 | .937 | +.717 | .917 | .383 | 3.46 | 12.88 | 2.9 | 8.6 |
| **conflict_g5** | 0 | .183 | .873 | **+.690** | .853 | .313 | 3.78 | 12.46 | 2.8 | 9.3 |
| override_same | 1.000 | 1.000 | 1.000 | 0 | 1.000 | 1.000 | 4.12 | 8.18 | 3.3 | 4.0 |
| **override_diff** | 0 | .067 | .997 | **+.930** | .067 | .997 | 4.46 | 10.83 | 3.6 | 5.6 |
| increment | 0 | .010 | .250 | +.240 | .010 | .250 | 2.29 | 8.76 | 2.1 | 5.6 |
| track | 0 | .200 | .230 | +.030 | .230 | .200 | 2.00 | 8.07 | 1.0 | 2.6 |

These numbers are stable across gaps: VR−V on each of the three gate categories changes by at most ±0.06 between gaps.

| held-out code (all positions) | s* | V | VR | VR−V | \|V\| | \|VR\| |
|---|---|---|---|---|---|---|
| all (1.23M) | .477 | .544 | .609 | +.064 | 1.55 | 6.60 |
| copyable (match ≥ 4) | .774 | .843 | .864 | +.021 | 2.01 | 7.63 |
| lk_wrong (s\* wrong) | 0 | .189 | .334 | +.145 | 2.06 | 7.90 |
| lk_none | 0 | 0 | .077 | +.077 | 0 | 2.05 |

**Gate: PASS.** VR − V exceeds 0.30 on all three of override_diff (+0.93), copy_diff (+0.86) and conflict_g5 (+0.69).

**Caveats for D3:**
- **The conflict_g5 pass is close to vacuous.** The deep chain walks through every server line, so VR holds about 9.3 distinct digit tokens at the deciding position, and there are only 10 digit tokens. The selection step does all the work there. The gate still passes without it: override_diff (dgVR 5.6) and copy_diff (dgVR 3.0) are the meaningful passes.
- **Which component does the work:**
  - Anchors carry copy_diff and override_diff (V+anchor = .907 and .997).
  - The chain carries conflict_g2 and conflict_g5.
  - For override_diff, VR also contains the stale value, so the scorer has to choose between them. Coverage is not selection.
- **On real code the gain is modest:** +6.4 points over all positions and +14.5 on lk_wrong positions. Expect only a small effect on held-out loss.
- **Alphabetic mask:** I used my own definition, a stripped piece matching `[A-Za-z_][A-Za-z0-9_]*` that contains a letter (24,068 of 32,000 tokens). The main tree's new `set_alpha_mask` may define it differently, so check that they agree before trusting G_VR against this gate.
- **Sanity asserts that passed on every sequence:**
  - V(last) == `ice_succ`.
  - The highest matching order's successor == `ice_s_star`.
  - Every source index is ≤ t.

## D1: long context (GPU)

**Models:** `r3090_B` and `r3090_C`, both **at 80.1M tokens**, not the final 400M. The decision requires final checkpoints, so rerun on those. These are also the only checkpoints on the volume.

**Data:**
- **These are genuine single documents.** val_code (2.29M tokens, 768 documents) has **61 documents longer than 8K tokens**, 939K tokens in total.
- 18 of those documents are longer than 16K, 2 are longer than 32K, and the longest is 52,840 tokens.
- Each document is scored on its own and truncated at 32K, which gives 918,754 scored positions.
- Positions per bucket: 0–4K 250K; 4–8K 250K; 8–16K 277K; 16–32K 142K (from the 18 longer documents only).

**Clamp:** `--clamp_dist` clamps `ice_dist` to ≤ 12, the largest bucket reachable at the 4096 training length. It does this by wrapping `model.forward`, so `score_answers` and CodeRecall are clamped too. There is no "star" distance bucket among the model's inputs: `ICEOutput` reads s\*/len\*/agree\*/rel\* only, and `ice_dist_star` is a raw distance used only for the copy_far split. So `ice_dist` is the only feature that needed clamping.

**(a) Held-out code loss in nats, all 61 documents:**

| bucket | B all | C_clamp all | C all | B copyable | C_clamp copyable | B lk_wrong | C_clamp lk_wrong |
|---|---|---|---|---|---|---|---|
| 0–4K | 2.866 | 2.164 | 2.164 | 2.584 | 1.284 | 4.046 | 4.370 |
| 4–8K | 2.854 | 1.904 | 1.904 | 2.647 | 1.221 | 4.075 | 4.454 |
| 8–16K | 2.810 | 1.863 | 1.864 | 2.617 | 1.263 | 3.994 | 4.397 |
| 16–32K* | 2.556 | 1.848 | 1.850 | 2.308 | 1.345 | 3.759 | 4.147 |

\*The 16–32K row comes from 18 documents only. Restricted to those same 18 documents, the losses are:

| bucket | B all | C all | B copyable | C copyable |
|---|---|---|---|---|
| 0–4K | 2.893 | 2.228 | 2.517 | 1.251 |
| 4–8K | 2.748 | 1.879 | 2.498 | 1.162 |
| 8–16K | 2.721 | 1.729 | 2.532 | 1.166 |
| 16–32K | 2.556 | 1.848 | 2.308 | 1.345 |

Clamping changes long-document loss by ≤ 0.004 at every bucket.

**(b) CodeRecall deciding-digit accuracy at gaps 6144/12288/24576 (n=100 per gap, mean over gaps):**

| category | B | C_clamp | C (unclamped) | C at gaps ≤ 3072 (reference) |
|---|---|---|---|---|
| copy_same | .083 | .357 | .273 | .522 |
| conflict_g1 | .000 | .767 | .317 | .661 |
| override_same | .117 | .193 | .210 | .350 |
| increment | .090 | .420 | .410 | .250 |
| copy_diff, conflict_g2, conflict_g5, override_diff, track | .02–.07 | .03–.06 | .03–.07 | .02–.07 |

The reference column is from the existing `check_r3090_C.json`.

**Reading D1:**
- **The pre-registered rule says drop the long-context claim, at least at 80M tokens.** Beyond 8K, B degrades by **−0.056** on all tokens and **+0.034** on copyable tokens relative to its own 0–4K value. Both are below the 0.10 threshold. B's copyable loss inside 4K (2.58) is already far above C's (1.28), so this B does little attention copying and has little to lose.
- **Hard to interpret:** loss falls with position, because later positions in a long document are easier. The proper control is the same positions scored with a truncated 4K sliding context. I did not run it.
- **ICE does extrapolate:**
  - C keeps its advantage over B at every bucket: −0.70, −0.95, −0.95 and −0.71 nats on all tokens.
  - On CodeRecall at long gaps, B sits at chance while C_clamp keeps conflict_g1 at .77 and copy_same at .36.
  - **The clamp matters for recall but not for loss:** on conflict_g1 it gives .77 against .32 unclamped. The untrained distance buckets 13–17 hurt on these tasks.
- **Possible ICE degradation on copyable tokens beyond 16K:** in the 18-document view, C's copyable loss rises by +0.09 at 16–32K while B's falls by 0.21. This rests on 18 documents and one seed.

**Not done / limits:**
- C_notrust was not evaluated (it is not in the requested run list).
- CodeRecall used n=100 per gap (300 per category) instead of the 600 per category in decision.md. Pass `--n_coderecall 200` to get there.
- The clamp wrapper was generalised after the full run: it now accepts `*args, **kw` and also clamps `ice_v_dist` when present, for the coming ICE-V/VR arms. That version passed a syntax check and a stub test but was **not re-run on Modal**, because the Modal workspace hit its spend limit at the end. Logic and results are unchanged for B and C.
