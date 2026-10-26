# Pointer diagnostics, t200 runs

Script: `ice_lm/scripts/diagnose_pointer.py`. One H100 run took 11.6 min: 2,070 CodeRecall items and 300 val_code windows of 4K tokens (1.23M tokens). Outputs are `/vol/diag/diag_t200.json`, with a local copy and the full log (`diag_full.log`) in this folder.

The CodeRecall items are exactly the evaluation items: same seeds, 100 per gap. The script reproduces the table accuracies (G_VR override_diff 0.007, conflict_g2 0.000, G_C increment 0.85 vs 0.86).

`_bb` means the pointer arm's own backbone, log_softmax(z), which is the same as `_off`. G_C_in and G_C_out are G_C with only its input side or only its output side switched on.

## 1. Why G_VR misses the answer even when it is in the candidate set

The numbers are at the deciding digit, pooled over 300 items per category.

| G_VR | override_diff | conflict_g2 | conflict_g5 | copy_diff |
|---|---|---|---|---|
| answer in the set | 1.00 | 0.94 | 0.87 | 0.91 |
| π sentinel | 0.54 | 0.15 | 0.16 | **0.93** |
| π on correct slots | **0.010** | 0.058 | 0.047 | 0.009 |
| π on wrong slots (of which s*) | 0.45 (0.30) | 0.80 (0.44) | 0.79 (0.43) | 0.06 (0.03) |
| top slot = s* | 1.00 | 1.00 | 1.00 | 0.51 |
| rank of the best correct slot, median (rank 1 share) | 11 (0.00) | 13 (0.00) | 12 (0.00) | 4 (0.22) |
| where the answer is | anchor 1.00 (anc0_k2_+2) | chain 0.77, anchor 0.27 | chain 0.74 | anchor 0.90 |
| where the wrong s* is | last 1.00 (o8_last) | last 1.00, anchor 1.00 | last 1.00, anchor 0.99 | last 0.81 |
| verifier γ·ℓ, correct vs top wrong | −1.45 vs **+0.55** | −0.80 vs +0.54 | −0.78 vs +0.49 | −1.80 vs −1.32 |
| backbone ℓ, correct vs top wrong | −3.57 vs −3.44 | −3.27 vs −3.24 | −3.03 vs −2.96 | −4.04 vs −4.10 |
| backbone: P[ℓ(y) > ℓ(s*)] | 0.46 | 0.48 | 0.47 | 0.61 |
| match length, correct vs top wrong | 3.2 vs 9.2 | **18.0 vs 13.0** | **31.6 vs 28.0** | 3.5 vs 1.7 |
| "longest exact match wins" rule | 0.00 | **0.98** | **0.97** | 0.00 |

- **The backbone carries no information on these items.** p_bb(y) is 0.05–0.07, about chance over 10 digits, and ℓ(correct) ≈ ℓ(wrong). Reranking the candidates by backbone log-prob alone is right 9–36% of the time, so the verifier has nothing to verify with.
- **With ℓ flat, the "verifier" term only encodes slot type, and it works against the new slots.**
  - γ is 0.28–0.61 on chain and anchor slots and −0.17 on o8_last. With ℓ ≈ −3, the correct slot gets about −1 and the wrong slot about +0.5.
  - Together with the learned priors, the correct slot starts about 3.5 nats behind (override_diff) or about 2 nats behind (conflicts).
- **The informative feature exists but the scorer cannot use it.** In conflict_g2/g5 the correct chain occurrence has the longest match, and a "longest exact match wins" rule would be right 97–98% of the time.
  - The pointer cannot express this rule: b+L is per slot and absolute, and the chain L is flat for lengths ≥4 (−2.10 at length 4, −2.19 at 32).
  - For override_diff the correct value is the more recent one (distance bucket 9.6 vs 10.7). No feature says "this is the latest write to this identifier".
- **copy_diff is a sentinel problem.** π0 is 0.93 and the slots barely matter.

## 2. Why G_V/G_VR follow wrong lookups more often yet lose less

These are val_code positions where the lookup is wrong (35% of tokens).

| | B | G | G_C | G_C_in | G_V | G_V_bb | G_VR |
|---|---|---|---|---|---|---|---|
| mean p(wrong s*) | 0.157 | 0.135 | 0.188 | 0.092 | 0.206 | 0.105 | 0.209 |
| mean p(target) | 0.294 | 0.287 | 0.262 | 0.277 | 0.276 | 0.290 | 0.276 |
| top-1 = s* | 0.240 | 0.210 | 0.316 | 0.153 | 0.338 | 0.166 | 0.339 |
| NLL | 3.189 | 3.236 | 3.432 | 3.213 | 3.271 | 3.172 | 3.230 |

- The mixture puts mass on the wrong token without removing much from the target. p(y) is 0.276 against 0.290 for its own backbone.
  - π0 averages 0.82 (V) and 0.77 (VR), which costs a mean −log π0 of 0.28 / 0.35 nats.
  - This is partly recovered when the target is among the candidates: 19% (V) and 33% (VR) of these positions, where NLL is 1.17 / 1.42.
- G_C's harm comes from its additive output bonus. G_C_out scores 3.454, while G_C_in scores 3.213, which beats G.

Calibration of the copy mass 1−π0. The AUROC is computed against "some slot = y".

| copy-mass bin | 0–.1 | .2–.3 | .5–.6 | .8–.9 | .9–1 |
|---|---|---|---|---|---|
| share of positions (VR) | 0.29 | 0.08 | 0.06 | 0.09 | 0.10 |
| P(some slot = y) | 0.38 | 0.64 | 0.83 | 0.95 | 0.98 |
| mass on y | 0.011 | 0.11 | 0.37 | 0.73 | 0.89 |

- AUROC is 0.825 for V and 0.810 for VR. ECE of copy mass against mass on y is 0.083 / 0.098.
- The pointer under-copies when it is unsure: in the bottom bin, 38% of positions have y in the set but only 1.1% of the mass goes there.
- Its split within the copy is poor. In the middle bins only 50–67% of the copy mass lands on y.
- VR dilutes the copy. When y is only in a "last" slot, the mass on y is 0.37 for V but 0.18 for VR. When y is only in a chain slot it is 0.056, and only in an anchor slot 0.046. Those two cases cover 2.7% and 3.9% of positions.

## 3. Laziness is confined to tokens the lookup gets right

The table gives NLL on val_code.

| group (share) | G | G_V_bb | Δ | G_VR_bb | G_C_off |
|---|---|---|---|---|---|
| all | 1.907 | 2.049 | **+0.142** | 2.053 | 2.344 |
| lookup right (0.48) | 0.687 | 1.047 | **+0.360** | 1.049 | 1.479 |
| copyable, lookup right (0.29) | 0.724 | 1.143 | +0.419 | 1.137 | 1.626 |
| lookup wrong (0.35) | 3.236 | 3.172 | −0.064 | 3.182 | 3.366 |
| no match (0.17) | 2.573 | 2.527 | −0.046 | 2.526 | 2.647 |

- By copy distance (<64 / 64–512 / 512–2K / ≥2K) the copyable gap is +0.21 / +0.39 / +0.25 / +0.13, so it is not specific to far copies.
- The lookup-right tokens account for +0.172 of the +0.142. Elsewhere the backbone is better than G (−0.03), so capacity has been moved, not lost.
- The cause is the gradient path: the mixture's p_bb(y) term is not detached. Wherever a slot already carries y, the backbone's gradient on that token is scaled by the sentinel responsibility, and the 0.5 aux CE does not make up for it.

## 4. G_C's +0.86 on `increment` is real, but it is a heuristic, not recall

- **The evaluation is correct.** Digits are separate tokens: `Migration ( id = 4 5 8` → y=`9`. The deciding token is the last digit, and s* is the previous id's last digit (`8`).
- **It comes entirely from the input side.** G_C scores 0.85, G_C_in 0.89, G_C_out 0.02, and G_C_off 0.04.
- **It is a "predict s*+1" rule, not tracking.** Control variants, 100 items each, accuracy (and the share of predictions equal to s*+1):

| G_C | +1 (eval) | step +2 | decreasing | constant id | random ids |
|---|---|---|---|---|---|
| accuracy | 0.85 | 0.00 | 0.00 | 1.00 | 0.09 |
| predicts s*+1 | 0.85 | 0.74 | **0.99** | 0.00 | 0.78 |

- **The rule is learned from real data.** On val_code positions with a digit and a wrong lookup (n = 11,992), y = s*+1 in 32% of cases. Top-1 = s*+1 is 0.26 for G_C_in, against 0.12 for G and B.
  - W_n·E[s_n] lets the network map the embedding of s* to the next digit.
  - A pointer can only copy, so V/VR cannot learn this.
  - The mixture also suppresses the 14% that G_V_bb gets on its own: π0 is 0.19, so p(y) falls from 0.144 to 0.027.
- **Do not cite it as recall.** It is a template prior that happens to coincide with this category.

## 5. Concrete fixes

1. **Relative candidate features.** Add a per-slot "match length minus the maximum over slots" (or a longest-match flag) and a "most recent source for this identifier" flag. The pointer cannot express "longest match wins", which is right 97–98% of the time on conflict_g2/g5.
2. **Make the verifier verify.**
   - γ is small (0.1–0.3 on the order slots) and negative on o8 (−0.10 / −0.17). The mixture objective rewards copying exactly where the backbone is unsure.
   - The score uses ℓ weakly: on real code, ℓ alone separates hit from miss slots with AUROC 0.89 (last), 0.95 (chain) and 0.98 (anchor), against 0.83–0.88 for the full score on the order slots.
   - Options: constrain γ ≥ 0, add a listwise CE over slots ("which candidate is y"), or give the scorer h·E[c_k].
3. **Recalibrate the VR slot priors.** b+L sits at −3.4 to −4.3 for anchors and is flat in length. Anchor precision is 0.11 (0.26–0.32 at offset +1), so the model learned to ignore them. One option is per-offset features plus the "identifier fully matched" flag from fix 1. Another is to upweight structured-conflict examples (rare in training).
4. **Sentinel.** It is not saturated: π0 > 0.99 at 9% of positions and < 0.01 at 0.3%. But β·max ℓ is inert (mean −0.15, std 0.16); w·h does all the work (std 1.66). On copy_diff that term drives π0 to 0.93.
5. **Laziness.** Detach the backbone term inside the mixture, train the backbone on the aux CE alone (weight 1), or use ICE-dropout for the V arms.
