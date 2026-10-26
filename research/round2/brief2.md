# Brief 2: ICE after the 200M-token runs (26 Oct 2026, night)

Read first: /Users/phunghuy/code/gstar/proposal_ice/results_t200/README.md (final tables, CodeRecall CIs,
pre-registered outcomes), /Users/phunghuy/code/gstar/proposal_ice/research/decision.md (the method ICE-V/VR),
/Users/phunghuy/code/gstar/proposal_ice/research/impl_notes.md (implementation), and diagnostics.md in this folder
(why the model behaves as it does: read it carefully, it is the newest evidence).

## Where we are
* 140M models, 200M tokens, one seed (seed 2 of B and G_VR is running).
* G = 16 GDN layers, no attention.
* Code loss: G 1.899, G_C (old ICE) 1.762, G_V (ICE-V) 1.703, G_VR (ICE-VR) 1.690, B (7:1 hybrid) 1.730.
  Prose: G_VR 3.324 vs B 3.384.
* Measured cache at 32K tokens per sequence: B 66.8 MiB, G_VR 5.5 MiB (ICE table 64.7 B/token).
  Max 32K sequences in 80 GB: 1,222 vs 14,907.
* Decode ~79 tok/s at batch 1; ICE CPU 0.35 ms/step, Python index.
* At 7:1 (400M tokens, old runs), ICE's gain over B vanished: 0.27 → 0.01 nats.

## Weaknesses to fix (the user asked to improve ALL of them)
1. **Over-copying:** top-1 = wrong lookup token 34% for G_V/G_VR vs 24% for B, although the loss on those tokens
   is fine.
2. **Laziness:** with ICE off, G_V is 0.14 worse than G on code. The backbone leaves copying to the lookup.
3. **Structured conflicts unsolved:** CodeRecall stale value (override_diff), "which object" (conflict_g2/g5) and
   other wording (copy_diff) are ≤ B for every ICE arm, although VR's candidates contain the answer (coverage
   100% / 87% / 91%).
4. **Exact copy weaker than the old ICE:** G_C conflict_g1 +0.90 / copy_same +0.34 vs G_V/VR +0.16–0.23. G_C has an
   input-side injection; G_V does not.
5. **7:1 story:** ICE adds ~nothing to a hybrid whose attention learned to copy.
6. **Evidence:** one seed; no A (3:1) at 200M; no "same-memory" attention baseline (e.g. G + one sliding-window
   attention layer with a cache of ~2 MB at 32K); no test on a real pretrained model.

## Options already on the table (evaluate, do not just adopt)
* **Real model:** Qwen/Qwen3.5-0.8B-Base, a real 3:1 GDN/attention hybrid (18 GDN + 6 attention layers; Apache-2.0;
  vocab 248K; needs a recent transformers; vision encoder to skip). Frozen; train only the ICE-V scorer (~2K
  params). Alternatives: Mamba-370M vs Pythia-410M (both on the Pile, with code). See real_models.md here.
* **Bolt-on on our own frozen G:** does a scorer trained on frozen G match the jointly trained G_V?
* **Baselines:** G + 1 sliding-window attention layer at equal cache memory; A at 200M; seeds.

## Budget and constraints
* Modal H100 ≈ $3.95/h.
  * One 140M × 200M-token arm ≈ 0.7 h ≈ $2.8.
  * Evals ≈ $0.2 per model.
  * A bolt-on on a frozen 0.8B model ≈ $1–3.
  * The workspace's spend limit is unknown: aim for ≤ ~$30 in total for the next round, and say which runs come
    first.
* ~1 day of implementation. Tests run on Modal CPU (`tests_cpu`). Keep the old arms bit-identical.
* Must keep the memory story (no per-token hidden vectors unless they are tiny and justified).
