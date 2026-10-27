# Round-3 CPU results (27 Oct; decision3 §4)

All on the real t200 checkpoints (B, G_VR) on the Mac CPU, fp32, pre-registered in decision3.md.

## §4.3 Index memory (`scripts/index_memory.py`, `index_memory.json`)
The compact layout was built as numpy arrays, with every lookup checked against the Python index (0 mismatches):

| context | counted (ideal load factor) | measured arrays | Python index |
|---|---|---|---|
| 4K | 61.9 B/token | 82.0 B/token | 520 B/token |
| 16K | 64.4 B/token | 82.0 B/token | 545 B/token |
| 32K | 68.7 B/token | 90.0 B/token | 639 B/token |

- Every row also carries a fixed 250 KiB order-1 table.
- Prediction "compact 60–80, Python ≥ 500": the measured arrays are slightly over 80, but under the 100 B
  "quote the measured value" limit. **Quote 65–90 B/token.**
- Per sequence: 3.8 vs 10.8 MiB at 4K (2.8×) and 6.3 vs 66.8 MiB at 32K (11×). The slides use these.

## §4.4 No-fit conflict probe on fresh windows (`conflict_probe/zprobe_fresh.py`, windows 100–299)
Positions where the lookup's top guess s\* is wrong and the answer is among the VR candidates: 96,148 positions in
200 windows, none used before. Rank-1 rate of the right candidate:

| score | s\* wrong, y in set | all ambiguous, y in set |
|---|---|---|
| G_VR's pointer | **0.406** (pred 0.36–0.45 ✓) | 0.868 |
| backbone log-prob ℓ alone | **0.645** (pred 0.58–0.70 ✓) | 0.821 |
| cosine of current vs source state (final layer) | 0.455 | 0.845 |
| random 16-d int8 key | 0.411 | 0.771 |
| uniform random | 0.140 | 0.155 |

- **The gap ℓ − pointer is 0.239, over the pre-registered 0.15, so it can be cited.**
- Where the lookup's best guess is wrong, the backbone ranks the right occurrence first 65% of the time, while the
  verifier gets 41%. On ordinary ambiguous positions the verifier is better (0.87 vs 0.82).
- So a context-dependent mix of the two should beat both. This is the question for V2's backbone-rank feature q
  and its softplus γ (Kaggle run 2).
- The 40-window numbers from the first probe (0.401 / 0.645) replicate.

## §4.5 Stale-version edit test (`scripts/stale_edit.py`, `stale_edit.json`, 100 held-out code files)
- **Setup:** v1 is the file with 10% of its distinct identifiers renamed (a prefix such as `old_`) and 10% of its
  numbers changed (first digit); v2 is the original. The model reads v1 + v2; v2 is scored.

| v2 tokens | B loss / top-1 | G_VR loss / top-1 |
|---|---|---|
| unchanged (85,258) | 0.894 / 0.800 | **0.391 / 0.921** |
| changed (3,851) | 2.520 / 0.571 | 2.716 / 0.486 |
| first token of a changed span | 3.785 / 0.382 | 4.245 / 0.216 |
| predicts exactly the stale v1 token there | 0.133 | **0.649** |
| control: v2 alone, unchanged / changed | 1.658 / 2.639 | 1.629 / 2.587 |

- **Pre-registered: unchanged G_VR < B by ≥ 0.10.** Result −0.503 ✓.
- **Pre-registered: changed ≤ B + 0.20.** Result +0.196 ✓, just under the limit.
- The secondary first-token metrics are not pre-registered. There, ICE copies the stale name 65% of the time vs 13%
  for B. That is the conflict weakness on a realistic code workload, and the slide states it.
