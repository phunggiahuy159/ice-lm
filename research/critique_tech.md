# Technical critique: correctness and feasibility in ice_lm/

Checked against: model.py (`forward`, `_loss_sum`, `LOSS_CHUNK=1024`, `forward_cached`, `InferenceCache`), ice_features.py, data.py (`ICE_KEYS`, `make_example`), inference.py (`ICEIndex`), evals.py (`eval_loss_splits`, `score_answers`), gdn.py, scripts/benchmark.py. Preset 140m: micro_batch 4, seq_len 4096, V=32000, 14 GDN + 2 attention layers.

**Applies to all three.** The brief's H100 B/C/C_notrust numbers are at **~105M tokens**, but every prediction is written against them as if they were final. Compare at a matched step, or re-evaluate C at 400M. Single seed: C vs C_notrust differs by 0.009 on loss and 0.014 on follow rate, so treat that as noise. Any threshold inside about 0.02 cannot be measured.

## Proposal 1: ICE-V (pointer over 12 candidates, ICE-free backbone)

Causality is fine. `succ2 = x[p2+1]` with p2 < p < t. `ICEIndex.step` already holds `second`, so the table costs 0 B/token, as claimed.

- **Major: numerics.** `log(π0·softmax(z) + Σπk·onehot)` in probability space underflows: π0 near 1e-30 times p_LM near 1e-20 gives log 0, so CE is inf and the run NaNs. *Fix:* in `_loss_sum`, compute `logaddexp(log π0 + logsoftmax(z)[y], log Σk πk·1[ck=y])` from a gather plus `logsumexp`. Build the full log-mixture only in `_logits`, for the eval paths.
- **Major: memory understated.** The author says "one [1024,V] tensor". It is [B=4,1024,V] fp32, 524 MB each. The recomputed chunk then holds z, log_softmax, softmax (for H), p_mix, log p_mix, CE's internal log_softmax and the aux-CE log_softmax: about 3.5 GB versus about 1 GB today. That is fine on an H100 and tight on the 3090. The log-space fix brings it back to about 2 tensors.
- **Minor.** Backbone gradient still flows through π0·softmax(z), so the backbone can still become lazy on copyable tokens. α=0.5 limits this, so C_off ≤ B+0.03 is a real test, not true by construction.
- **Minor.** Occurrence o=2 has no distance feature (`ice_dist` stores only the last occurrence). Add `ice_dist2`, or the score is blind to how stale the value is.
- **Minor.** ICE-dropout is pointless when the backbone never sees ICE: it only removes 20% of the pointer's training signal. Keeping it is defensible for parity, but say so.
- **Minor.** `_logits` must return the mixture **and** expose z for the aux loss. That means refactoring `_logits`/`_loss_sum` signatures, `OUT_KEYS`, `FEATURE_KEYS_N` and `step_features_to_tensors`. The eval paths (`eval_loss_splits`, `score_answers`, benchmark) then work unchanged, because CE on normalized log-probs is the identity. I verified this.
- Predictions are measurable. Follow rate 0.356 → ≤0.27 and the V-vs-V_nover gap of ≥0.03 are well above noise. The "within 0.01" kill test sits at the noise floor.

Effort is realistic: about 1 day. **Feasibility 8/10.**

## Proposal 2: ICE-Route (32 indexed slots, 64-d stored vectors, pointer)

- **Major (fatal if implemented as written): label leakage.** Anchor spans q+1..q+4 can pass t. If the anchor sits at t and its earlier occurrence is at q = t−2, the slots cover t−1..t+2, and **x[t+1] is the label**. The text claims "every slot c ≤ t", but the construction does not enforce it. A gather-based pointer would find the leak at once, giving a spectacular train loss and garbage decoding. *Fix:* mask `c > t` in the CPU builder and add a unit test against brute force.
- **Major: eval paths silently break.**
  - The mixture is specified only inside chunked `_loss_sum`. `eval_loss_splits` and `score_answers` call `_logits` and would score p_LM alone, giving wrong CE, argmax and follow rate.
  - `score_answers` and CodeRecall call `compute_ice_features(seq, None, orders)` with no frequency-rank file, so anchors would be absent or crash.
  - C_off (`ice_keep=0`) only reaches modules that read `keep`. ICERoute must also gate both reads on it.
- **Major: anchor threshold.** CodeRecall keys (`db_port`, `timeout_ms`, `PORT`) split into subwords such as `port`, `timeout`, `_`, `ms`. With a 30% code mix these are plausibly inside the top-512. If so, the anchors drop exactly the identifiers that matter. The day-0 coverage gate catches this, which is good, but expect to iterate.
- **Minor: cost.**
  - Bytes/token come to 41 + 24 (prev, 6 orders) + 4 (unigram chain for anchors) + 128 = **~197 B**, slightly above the stated range. It is still 10× below KV.
  - Gathering `W_e E[x_c]` naively is [4,4096,32,768] bf16 = 805 MB. Project E to 64-d first.
  - The other activations (several [4,4096,32,64] tensors, 67 MB each, read twice) are fine.
- **Minor.** The kill rule "R−C < 0.10 (CI excludes 0)" contradicts itself: it kills a significant positive result. Restate it.
- Predictions (≥+0.15 with paired SE ≈0.04 on 300 items) are measurable.

Effort: CPU builder, rank file, module with two reads, mixture in both `_logits` and `_loss_sum`, rank plumbing through evals and CodeRecall, inference store. Realistically **3–4 days**, over the 1–2 day budget. **Feasibility 4/10.**

## Proposal 3: ICE-Draft (run-length head + speculative decoding)

The training side is trivial. The risk is entirely in decoding.

- **Major: a partial accept costs a second forward.**
  - The fla chunk kernel returns only the final state, so a partial accept needs restore plus a re-run.
  - With mean accept around 1.8, that gives about 2.8 tokens per ~2 forwards, roughly 1.4×, before the Python ICE overhead.
  - The ≥1.5× target is optimistic. *Fix:* fold the re-run into the next verify call (restore, then feed accepted + bonus + new draft together).
- **Major: batch-16 speculative decoding is not possible as is.** `InferenceCache.pos` is a single int for the whole batch, so rows with different accept lengths misalign the KV writes. Either accept the minimum over the batch (which kills the speedup) or add per-row positions.
- **Major: "lossless" and the 32K setting.**
  - Verification of T=k+1 goes through the fla chunk kernel, while plain decode uses `gated_delta_rule_step` (torch). In bf16/fp16 these differ, so argmax ties flip. Report the agreement rate instead of claiming identical output.
  - The model was trained at seq_len 4096, so free-running greedy decoding at 32K is out of training length for the RoPE attention layers.
  - Greedy decoding with a 140M model degenerates into loops, which ICE drafts perfectly and which inflates acceptance. Report a repetition rate, or measure at ≤4K on prompts.
- **Major: the trust claim is probably null.**
  - The label 1[r≥1] equals 1[s*=y], which is exactly the gradient the existing softplus gate already gets from the LM loss.
  - The head reads the same h_last that has already absorbed the injected successor.
  - Prediction 4 and kill (iii) will likely fire, exactly as agree/rel did.
- **Minor: ICE undo.** `hist` is a `deque(maxlen=32)`, so `append` evicts the oldest entry and an undo log cannot restore it. Snapshot the deques, `pending` and `seg_start` instead.
- **Minor: the `ice_run` label.**
  - It must be kept out of `ICE_KEYS`/`FEATURE_KEYS`: it is future information, and `ICEIndex` cannot produce it at decode.
  - Mask the last K positions of each window, where the LCP is truncated.
- **Minor: unmeasurable prediction.** Prediction 5 ("F_run within 2 points of A") is an equivalence claim under SE ≈4 points. It also needs an A arm.

Effort is about 0.5 day for training and 2–3 days for correct speculative decoding plus the benchmark. **Feasibility 5/10.**

## Ranking (technical soundness × feasibility within budget)

1. **P1 ICE-V.** Smallest diff, no causality risk, eval paths survive. Its two real bugs (prob-space log, memory) have one-line fixes. The step-0 AUROC gate is cheap and decisive.
2. **P3 ICE-Draft.** Training is safe and cheap. The headline speedup is threatened by the second forward per partial accept, batch-shared `pos`, numerics and degenerate generations. The trust component is likely a repeat of agree/rel.
3. **P2 ICE-Route.** Highest upside on conflict_g5/override_diff, but it carries a real leakage trap, three silent eval breakages and twice the time budget. Run its CPU coverage gate first, regardless of which proposal is chosen.
