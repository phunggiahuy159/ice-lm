# Round 3 brief (27 Oct 2026)

Read with: `results_t200/README.md`, `research/round2/decision2.md`, `research/round2/checks2.md` (F1),
`research/round2/impl_v2_notes.md`, `slides/talk_notes.md`.

## Where the project stands
**Method.** ICE is an exact in-context n-gram lookup on token ids. It runs on the CPU as a hash table, about
65 B/token.
- **ICE-V:** the lookup proposes candidate next tokens. A small verifier mixes them with the backbone's distribution
  in log space (pointer-sentinel). The backbone never sees the lookup.
- **ICE-VR** has 34 slots: the last and second occurrences, a deep suffix chain, and identifier anchors.
- **ICE-V2** (implemented, not run) adds 0-byte relative slot features: margin Δ, agreement n, recency r and
  backbone-rank q, plus a listwise loss.

**Main result.** 140M models, 200M tokens, 2 seeds, held-out code + FineWeb-Edu.

| | cache / token | code loss | prose loss | code top-1 | top-1 on copies > 512 back |
|---|---|---|---|---|---|
| G: 16 GDN, no attention | 0 | 1.899 | 3.444 | 62.4% | 64.6% |
| B: 7:1 GDN/attention | 2,048 B | 1.730 | 3.384 | 66.1% | 76.2% |
| G_VR: G + ICE-VR | 65 B | **1.690** | **3.324** | 66.5% | 80.8% |

- Seed 2 gives the same picture: G_VR − B = −0.040 / −0.045, with seed spread ≈ 0.004.
- Memory: 5.5 vs 66.8 MiB per 32K sequence. Decode is 9% slower, because the index is Python.

**Known weaknesses**
1. **Conflicts.** Stale values, "which object", other wordings: every ICE arm is ≤ B on CodeRecall. The right
   candidate is often in the slot set, but the verifier scores slots independently.
   - F1 (rules on real code): no fixed rule beats 0.07 where the lookup's top guess s* is wrong; the oracle is
     0.34.
   - Δ/n/r carry signal on all ambiguous positions, but almost none where s* is wrong.
2. **Wrong lookups.** Top-1 is 34.6% vs B's 37.7%, and G_VR follows a wrong lookup 34% of the time vs 24% for
   B. It hedges rather than refuses.
3. **Reliance.** With ICE off, the model is 0.14 worse than G. Arguably specialisation.
4. **Aggregate top-1 is only +0.4 pts over B,** although loss is −0.04. Most of the gain is calibration on copyable
   tokens.
5. **Missing equal-memory attention baseline.** G_MQA (one MQA layer, 256 B/token, 4× ICE's bytes) is implemented
   but not trained.
6. **One scale (140M), one tokenizer, 200M tokens.** No real pretrained model yet; Mamba2-370M vs Pythia-410M is
   planned.
7. **Synthetic recall was unfair to B.** With random full-vocab tokens, B scores 0.04 on repeat and ~0 on MQAR.
   A new CPU check on the real checkpoints, n = 10 per cell (`round3/recall_check_cpu_n10.json`), shows:

   | task | B | G_VR |
   |---|---|---|
   | repeat, random frequent tokens | 0.44–0.62 | 0.98–1.00 |
   | repeat, real text | 0.58–0.62 | 0.94–0.97 |
   | MQAR, frequent tokens | 0.09–0.19 | 0.61–0.73 |
   | MQAR, full vocabulary | ~0.01 | 0.84–0.92 |

   So the "0.00" was mostly a rare-token artefact. The real gap is about 60% vs 97% on verbatim repeat.
   - G_VR is *worse* on frequent-token MQAR than on full-vocab MQAR. Frequent tokens recur, so the lookup proposes
     spurious successors.

## Compute available now
- **Modal** (H100) is blocked: the workspace hit its spend limit, which resets on 1 November or when the user
  raises it.
- **Kaggle 2×T4 (free, about 30 GPU-h/week).** The user runs a notebook we write:
  `ice_lm/notebooks/ice_eval_kaggle.ipynb`.
  - It re-evaluates B/G/G_VR, the fitter gate, the recall check, greedy line-completion exact match, and the
    ICE-V2 fit ladder (decision2 runs 2–5).
  - All fp32: the T4 has no bf16. Roughly 6–10 h.
  - Training a new 140M model from scratch on a T4 is too slow: G_MQA's 200M tokens is about a day.
- **Mac CPU** (Apple Silicon): CPU-only checks and small evaluations on downloaded checkpoints (B and G_VR are
  local). A 148M forward on 4K tokens takes a few seconds.

## Rules
- Pre-register predictions.
- Keep old arms bit-identical.
- No claims under 0.02 loss from one seed.
- Never touch other projects' Modal resources.
- No credentials in files.
