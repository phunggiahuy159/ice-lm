# Technical critique, round 2 (checked against model.py, ice_features.py, inference.py, data.py, modal_app.py)

## Cross-cutting facts from the code
- **Causality.** `_candidates` returns src = p+1 (occurrence slots) or q+i (anchors), with `assert src <= t`, and empty anchor slots when q+i > t. So C's key index i_k−1 ≤ t−1 is causal. It is always in the same document: for occurrence slots i_k−1 = p, and for anchors it is ≥ q. A's Δ/n/ν use only slot tokens, lengths and positions, so neither proposal peeks at the target. The listwise CEs condition on y only to mask the loss, which is legitimate.
- **Decode.** `ICEIndex._candidates` already holds every p/q but returns only tok/ln/dist. Adding src is about 10 lines. For C the order matters: step(tok) builds the features, the forward computes h_t, and only then append_key(k_t). Bit-exact int8 keys between the chunked fla training kernel and recurrent decode are unlikely: values on rounding boundaries flip. The cache-equals-forward test needs a tolerance, or keys computed from the decode path in both.
- **Hidden gradient paths.** `loss_sum` leaves `ly` (p_bb(y) inside logaddexp) attached, and s0 = w·RMSNorm(h_last) is **not** detached. "Detach ⇒ ICE-off = G" needs **both** stopped, plus aux = 1. A writes sg(h_t), so A is correct. B's and C's "cannot become lazy" / "keep=0 bitwise" claims are about the forward pass only.
- **Length cap.** MAX_MATCH = 32. conflict_g5 lengths are 31.6 vs 28.0, so the correct slot is nearly always at the cap. Any longer shared prefix gives ties (Δ = 0), and the 0.97 "longest wins" rule partly reflects the template length.
- **Distance clamp.** V_DIST_MAX = 12 is a module constant. B's "Qwen at 16K needs no clamp" is false: the code clamps there. Making it a config value is also needed to keep the old arms bit-identical.
- **Tuning to the test.** ν and Δ were designed from the 2,070 evaluation items. Any claim needs fresh CodeRecall seeds plus perturbed templates (longer prefixes, ≥3 writes, swapped order), as the §4 increment controls did.
- **Noise.** Paired comparisons on one frozen backbone (bolt-on vs off, scorer A vs B) have tiny variance: only scorer-fit noise and eval sampling. Comparisons across runs carry ~0.01–0.02 of seed noise. CodeRecall uses 300 items per category, so SE ≈ 0.03 per arm and a paired difference needs about ≥ 0.10.

## Proposal A (ICE-V2, frozen G + scorer): feasibility 8, value 7
- **Correct:** with sg(p_bb), sg(h) and CE at weight 1, the backbone's training is literally G's, so fitting on frozen G is not a proxy but the method itself. ICE-off = G holds trivially.
- **The cost is the headline.** Diagnostics §3: G_VR's backbone is 0.054 better than G on lk_wrong (35%) and 0.047 better on no-match (17%), worth ≈ 0.027 nats.
  - On no-match the mixture passes almost all of that through (≈ 0.008).
  - On lk_wrong the mixture keeps only 0.006 of it (3.230 vs 3.236). Frozen G with the same mixer overhead (+0.05) lands near 3.28, which is ≈ 0.02 worse than G_VR overall.
  - Lookup-right tokens recover part of this, because G's backbone is better there (0.687 vs 1.049).
  - My estimate is 1.705–1.72. That puts it at B parity, and "beats B" probably dies. A's P6 band is honest, but the gap to G_VR (0.015–0.03) is 1–2σ of seed noise at one seed.
- **Novelty also shrinks** to "kNN-LM/Bakhtin-style post-hoc mixer" (real_models.md).
- **Gaps.** P9 (V2 on the G_VR backbone) is the right control. The "V2-joint" fallback should be a planned $2.8 run, not a contingency. T1 (+4 hand edit) is fine as a bound.
- **Cut:** P3 (conceded anyway). Do not claim seeds are "free": a G seed still costs a 140M run.

## Proposal B (evidence + real models): feasibility 4, value 7 (if trimmed)
- **G_SWA arithmetic is right.** 15 × 0.2 MiB + W × 1 KiB = 5.47 MiB gives W ≈ 2,530. C's "W ≈ 1,024 ≈ 2.7 MB" contradicts this. Use B's number.
- **But memory is matched only at 32K.** At 4K, where training and most evals happen, the window covers 62% of the context and holds 2.5 MiB against ICE's ~0.25 MiB. Most copy distances are < 2K, so this is effectively a 15:1 full-attention hybrid.
  - A G_SWA win at 4K is therefore not decisive.
  - The decisive evaluation is at copy distances > 2,560 at 8–32K, and it also depends on G/GDN length extrapolation.
  - Report a bytes-vs-context curve. Add W ≈ 700 (matched at 4K) only if cheap. int8 KV would double W, so state bf16.
- **No window code exists.** model.py has no band mask or ring cache, and the 3 h estimate is optimistic: a SWA layer plus a cache plus a new split comes to about 5 h with tests.
- **Bolt-on on G.** This is a test of a *different* method (no co-adaptation). It bounds joint ICE-V; it does not validate it. Label it.
- **Qwen3.5: cut this round.**
  - It needs a separate image: the current image pins torch 2.6 and fla 0.4.2 with `|| true`, so it falls back silently. It also needs transformers 5.x, fla GDN kernels and causal-conv1d.
  - The VL checkpoint has to be stripped.
  - The 248K vocab makes [T,V] fp32 4 GB per 4K window, so every eval path needs chunking (`diagnose_pointer` and `evaluate` materialize the full mixture).
  - The distance clamp needs fixing too.
  - Realistic cost is 1–1.5 days on its own, for a predicted 0.00–0.02 gain.
- **Mamba2/Pythia** is cheaper (plain transformers). But NeoX merges digits, so CodeRecall needs a multiple-choice redesign (another half day), and the slow Mamba2 path is slow at 20M tokens. Make it a stretch goal.
- **B uses K = 12 (V), not VR**, so it tests last round's scorer.

## Proposal C (ICE-K, 16-byte keys): feasibility 6, value 6
- **Mechanism.** Causal and decode-computable, and 16 B/token fits the memory story. C's 68.7 B/token conflicts with the measured 64.7.
- **Main risk: keys may add nothing beyond ℓ.** h_{i−1} is trained to predict x[i] = c_k, so q·k largely re-measures "does my state predict c_k", which is what ℓ does. Real value needs the GDN to keep line context (the server name) for a few tokens.
- **The conflict gains may be token-only.** P0's longest-match rule already gets 0.97–0.98 at 0 bytes. The keys' unique case is copy_diff and override_diff, and C itself rates override_diff low-confidence.
- **The frozen-G_VR probe (P1) is a fair asymmetric gate.** Positive is strong evidence. Negative is weak evidence, because joint training can shape h. Still, the kill rule (< 0.25 on all four) is reasonable.
- **Leaves laziness untouched.** Weakness 2 is ignored, and q·k adds another gradient path into h.
- **The code-loss prediction (−0.01 to −0.025, one run) cannot be resolved from noise.** Only the CodeRecall deltas (≥ +0.30) are testable.
- **Cut:** the full G_VRK run unless P1 passes *and* beats A's token-only scorer by ≥ 0.15 on copy_diff or override_diff.

## What I would run (≈ $27, in this order)
| # | Run | $ |
|---|---|---|
| 1 | CPU analyses, run once: A T0 + C P0, on fresh CodeRecall seeds and perturbed templates | 0.3 |
| 2 | Frozen fits: the A ladder on G; V2 on the G_VR backbone (P9); V2 on B | 5 |
| 3 | C P1 key probe | 1 |
| 4 | G_SWA (W = 2,560) and A @ 200M, plus long-context evals at 8–32K | 7.5 |
| 5 | V2-joint 140M run; G_VRK only if gate 3 passes | 2.8 + 2.8 |
| 6 | Seed 3 of B and G_VR (the G_VR-vs-B gap is 0.040) | 6 |
| 7 | Stretch: Mamba2 vs Pythia | 2.5 |

Cut: Qwen3.5, the W = 700 arm, and any loss-level claim < 0.03 without 3 seeds.
