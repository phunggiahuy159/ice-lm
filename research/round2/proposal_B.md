# Proposal B: evidence that convinces faculty, plus a real model (≤ $30)

Each run answers one objection a reviewer would raise. The frozen-backbone bolt-ons give the most evidence per dollar: each costs $0.3–4, and they cannot become lazy because the backbone never trains. New 140M arms at ~$3 each come second.

## 1. Baselines

**G_SWA: 15 GDN layers + 1 sliding-window attention layer**

Layout:
- It sits where F places its attention layer (layer 15).
- It uses B's KV layout: 4 KV heads × d64 × K,V × bf16 = 1,024 B/token.

Bytes at 32K:
- Each GDN layer holds 0.2 MiB of state. This is B's measured (66.8 − 64.0)/14, and matches 12 × 64² × fp32.
- G_VR = 16 × 0.2 + 64.7 B × 32,768 + 256 KB (order-1 table) ≈ 5.47 MiB, which matches the measured 5.5.
- G_SWA = 15 × 0.2 + W × 1 KiB. Setting this equal to 5.47 MiB gives **W = 2,530, rounded to 2,560**.

Caveats:
- The memory is equal only at 32K. At 4K, the window holds 2.5 MiB against ICE's ~0.5 MiB, and covers 62% of the context. This is a generous baseline, and it also stands in for the missing 15:1 arm.
- A window with equal memory at 4K would be W ≈ 700. It is optional.

Code: a band mask in SDPA, a ring-buffer cache, and a new split for copy distance > 2,560. About 3 h.

**A (3:1) at 200M** is the upper reference at 4,096 B/token. It is essential because Qwen3.5 is also 3:1.

**Seeds**

Add a seed only where the difference is < 3σ, with σ taken from the seed-2 pair that is already running.

| Comparison | Gap | Seeds |
|---|---|---|
| G vs G_VR | 0.21 | none |
| G_VR vs B | 0.040 | 3 each |
| A or G_SWA vs G_VR | — | a 2nd seed only if \|Δ\| < 0.03 |
| G_VR vs G_V | 0.013 | never claimed |

**Bolt-on on frozen G** (decision.md D4)

- Freeze G and train only the ICEPointer (1,976 params) on 20M tokens, with loss −log p_mix and no aux term.
- Repeat on frozen B, and later on A.
- This calibrates how to read the real-model results.
- It is also a $0.3 test bed for the scorer fixes from the diagnostics: relative match length, latest-write flag, γ ≥ 0, listwise CE.

## 2. Real-model track

**Primary pair: mamba2-370m-hf vs pythia-410m**

Why this pair:
- Both are trained on the Pile (~300B tokens; unverified for Mamba2), use the NeoX tokenizer and are of near-equal size.
- Both run in plain transformers, with no fla.
- Mamba2's torch fallback is a chunked SSD; Mamba-1's is a sequential loop.

Claim: "closes X% of the recurrent → Transformer gap at ~70 B vs 96 KB per token."

Control: ICE on Pythia should gain little.

**Second model: Qwen3.5-0.8B-Base**

Claim: "safe on a strong hybrid, helps at long context." It is not the headline, because its 6 attention layers already copy.

**Recipe (`scripts/bolt_on.py`)**

| Step | Detail |
|---|---|
| Data | Rerun `prepare_data` with each model's tokenizer: the same streams, 2M val tokens per split, 20M train. uint32 is chosen automatically; ICE hashing is vocab-agnostic; the alpha mask strips Ġ. |
| Backbone | Run under `inference_mode` in bf16. h = `model.model(ids).last_hidden_state` (after the final norm); z = h·W_lmᵀ per 1,024-position chunk. Test: equal to `model(ids).logits`. |
| Scorer | `ICEPointer(d=hidden, K=12)` in fp32, using `loss_sum` per chunk with aux = 0. One pass of 20M tokens at lr 1e-2: 5–10 min. |
| Length | Pythia/Mamba2 train at 2K. Qwen trains the scorer on 16K windows, so distance buckets ≤ 14 are seen and no clamp is needed. |
| Eval | 500 windows per split; all existing splits plus ICE-off. Qwen also runs 8–32K documents; Mamba2 ± ICE runs at 8K. |
| CodeRecall | NeoX merges digits, so score multiple choice over the item's values (summed log-probs). Rejection-sample values so their first tokens differ. Gaps 256/1K/1.5K for NeoX, 1K–24K for Qwen. Drop `increment` for NeoX. |

**Risks**

| Risk | Mitigation |
|---|---|
| Qwen3.5 needs transformers 5.x | Build a separate image `ice-real` (exact pins, fla-core, causal-conv1d). The main image stays untouched, so the old arms stay bit-identical. |
| The GDN fast path falls back to torch silently | Assert the fla kernels are used, and compare fast vs torch logits on 2K tokens. A $1 smoke covers this. |
| 248K vocab: [T, V] fp32 is 4 GB at 4K and 32 GB at 32K | Compute NLL with the gather-only `loss_sum`. Run `_mix` (top-1, follow-wrong) per chunk. Recount bytes: the order-1 table is 2 MB, tokens are 4 B. |
| Vision encoder | Load only `Qwen3_5ForCausalLM`/text_config, or delete `.visual`. Check that no LM keys are missing and that the text logits are unchanged. |
| Contamination | Treat CodeRecall (random values) and the copy splits as primary. |

## 3. Ordered run list

| # | Run | $ | Objection closed |
|---|---|---|---|
| 1 | Bolt-on on frozen G and B | 0.8 | "Joint training, laziness" |
| 2 | Re-tokenize the data for 2 tokenizers (CPU) | 0.4 | — |
| 3 | Mamba2 ± ICE, Pythia ± ICE, CodeRecall multiple choice | 2.5 | "Toy models" |
| 4 | **A** @ 200M | 3.0 | "Real hybrid?" |
| 5 | **G_SWA** (W = 2,560) | 3.0 | "Just use a window" |
| 6 | Long-context eval at 4–32K for G, G_VR, B, A, G_SWA | 1.5 | "Within 4K only" |
| 7 | Qwen3.5 bolt-on: smoke, 16K training, 4K/32K eval | 4.0 | "Not modern" |
| 8 | Bolt-on on frozen A | 0.4 | Cross-check of run 7 |
| 9 | Seed 3 of B and G_VR (conditional) | 6.0 | "One seed" |
| 10 | Seed 2 of A or G_SWA (conditional) | 3.0 | — |
| | Reserve | 5.0 | |
| | **Total** | **≤ 29.6** | |

## 4. Pre-registered predictions and kill criteria

1. **Bolt-on on G**
   - Prediction:
     - code ≤ 1.78 (≥ 60% of the G → G_V gain);
     - copyable gain ≥ 0.25;
     - lk_wrong ≤ G + 0.03;
     - ICE-off identical to G.
   - On B, the gain is ≤ 1/3 of the gain on G.
   - **Kill the real-model track** if the copyable gain is < 0.10.
2. **Mamba2 vs Pythia**
   - Premise: the gap on copy > 512 is ≥ 0.10. If it is < 0.05, report "no recall deficit" and stop.
   - Prediction:
     - ICE closes ≥ 50% of that gap;
     - code improves by ≥ 0.03;
     - prose changes by ≤ +0.005;
     - CodeRecall multiple choice on copy_same is ≥ Pythia − 0.05.
   - ICE gains ≤ 1/3 as much on Pythia.
3. **A**
   - Prediction: A ≈ 1.705, and G_VR is within 0.02 of A at 1/60 of the cache.
   - If A < G_VR − 0.03, the claim becomes "approaches B, not A".
4. **G_SWA**
   - Prediction within 4K:
     - G_VR beats G_SWA by ≥ 0.02 on code and by ≥ 0.3 on copy > 2,560;
     - G_SWA ≥ G_VR on the structured conflicts at gaps ≤ 1K.
   - Prediction at 8–32K: G_SWA reverts toward G.
   - **Kill the memory headline** if G_SWA ≤ G_VR on code and on 32K copyable. In that case, test G_SWA + ICE-V next.
5. **Qwen3.5**
   - Prediction: a gain of 0.00–0.02 within 4K, and a gain at 16–32K at least as large.
   - If the harm on prose or lk_wrong is > 0.02, narrow the claim to recurrent backbones.
6. **Seeds**
   - Claim "G_VR < B" only if all 3 seeds give Δ < 0 with a mean ≤ −0.025.
   - Otherwise claim "matches B".

**Figure for reviewers:** one frontier plot of loss vs measured bytes/token at 32K. Points: G, G_VR, G_SWA, B, A, Mamba2 ± ICE, Pythia, Qwen ± ICE.
