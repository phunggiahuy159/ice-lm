# Rebuttal 3: ICE-Draft, conceded, cut down, and folded into a combined plan

## Conceded
- **Partial accepts cost a second forward.** The realistic ceiling is about 1.4×, not ≥1.5×, before Python overhead. Mitigation: fold the re-run into the next verify call, or use input replay (ReplaySSM) instead of snapshot/restore.
- **Batch-16 speculative decoding is impossible** with the single `pos` in `InferenceCache`. Dropped. Speculative decoding is reported at batch 1 only.
- **"Lossless" is wrong.** The chunked fla path and `gated_delta_rule_step` differ numerically, so I will report a token-agreement rate with plain decode instead.
- **32K free-running generation is out of training length (4K) and invites loops**, which ICE drafts perfectly. Revised setup: 3–4K code prompts, 256 generated tokens, and a reported repetition rate (fraction of generated 8-grams already in the prompt or output).
- **The run-length trust signal is probably null.** 1[r≥1] = 1[s*=y], which the softplus gate already receives through the LM loss, and the head reads an h_last that has already absorbed the successor. Withdrawn as a trust fix.
- **Novelty is 3/10 for speed.** LLMA (2304.04487) is the ancestor of PLD. Speculative decoding on SSMs/hybrids exists (Mamba-in-the-Llama, RAD 2505.22135, ReplaySSM). I claim no rollback novelty.
- **Lossless speculative decoding rewards blind copying.** Speed is orthogonal to the reviewer's main concern (concerns 1–2). It cannot be the headline.

## What survives (as components of the combined plan)
1. **The A and F arms at the full recipe.** The faculty critic is right that they are the identity test: does an attention-free exact copy path let a hybrid drop attention layers? No trust fix matters unless C/F closes most of the B→A gap at the matched 400M-token step. All comparisons are made at matched steps, since the brief's B/C numbers are at ~105M tokens.
2. **Measured cost, not modelled cost.** `benchmark.py` already exists. Run it on trained A, B, C, V and F at 4K/32K and batch 1/16, reporting KV, GDN state, table bytes, tokens/s and max batch per 80 GB. This answers reviewer item 3 almost for free.
3. **B+PLD as a half-day "deployment" slide, not a claim.** It is cheap, nearly deterministic and honest.
4. **A sharper speculative-decoding twist that fits P1.** In ICE-V the backbone never sees ICE, so drafts can be verified by the **ICE-blind backbone z**. The resulting acceptance rate cannot be inflated by over-reliance by construction. It is a measured speed number that is also evidence *against* blind copying. It is optional and only runs if time is left.
5. **The `ice_run` label** as one extra sentinel feature in P1 (a few lines; it must be kept out of `ICE_KEYS`).

## Revised combined plan (about 4 arm-runs)
- **Core (P1):** ICE-V and V_nover.
- **Identity (P3):** A, and F_V (15:1 + ICE-V output).
- **Cost (P3):** the measured benchmark table.
- **"Where exact fails" (P2):** the CPU coverage diagnostic only.
- **Headline figure:** x = measured bytes/token (log scale), y = CodeRecall / copyable loss, with points A, B, C, V and F_V, plus a follow-wrong-rate inset. The decode speed of each point goes in the table, not the figure.

## Fair attacks on the rivals
- **P1's weakest point is circularity, and its own step-0 gate does not test it.** On far-back identifiers the backbone is least able to judge the lookup, and that is where ICE's value lies (copyable >512: B 1.918 vs C 1.299). The gate must therefore be **stratified on the copyable->512 split**, not pooled. A pooled AUROC gain driven by short-range, easy tokens would pass a method that fails exactly where it matters. P1 also never ran C_in/C_out. If C's gain is on the input side, output-only V gives most of it back. Without an A arm, P1 cannot say whether ICE-V matters at all.
- **P2's weakest point is scope.** It has a label-leakage trap (anchor spans past t), three silent evaluation breakages and 3–4 days of work. A stored 64-d vector per token, read by softmax over gathered slots, is small sparse attention at ~190 B/token. That blurs the attention-free identity the project is graded on. Its +0.10–0.15 thresholds against SE ≈ 0.04 at one seed will likely come out ambiguous. Its coverage gate is excellent and should run regardless.
