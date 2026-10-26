# ICE-V / ICE-VR: implementation notes

Code: /Users/phunghuy/code/gstar/proposal_ice/ice_lm (pre-change copy: scratchpad/ice_lm_frozen).

## What changed
- **ice_features.py**
  - `compute_ice_features(..., cand="" | "v" | "vr")` adds `ice_v_tok`, `ice_v_len` and `ice_v_dist`, each [K,T].
    - V (K=12): order j × {last, second}. Each slot has the successor, its exact match length (0..32) and bucket(t−p).
    - VR (K=34): V plus 6 chain slots (occurrences 3–8 of the longest-matching order, each with its own match length).
    - VR anchors (16 slots): 2 anchors × {order-1, order-2 key} × span x[q+1..q+4]. Anchors carry match length(a, q) and bucket(a−q). A slot is empty when q+i > t, and the builder asserts `src <= t`.
  - The alpha mask goes through `set_alpha_mask` / `get_alpha_mask`. The getter raises if the mask is unset.
  - `n_slots()` returns K.
  - One local variable was renamed (`cand` → `chk` in the match-length loop). Behaviour is unchanged.
- **model.py**
  - New `ICEPointer` (named `ice_ptr`) implements decision §2:
    - s_k = b[src] + L[src,len] + D[min(dist,12)] + γ[src]·clip(ℓ[c_k],−20,0).
    - s_0 = wᵀRMSNorm(h) + β·max ℓ + c0.
    - log π = log_softmax with −inf on empty slots and on keep=0.
  - `loss_sum` runs inside the existing 1024-chunk checkpointed `_loss_sum`. It computes −logaddexp(log π0+ls[y], logsumexp_{c_k=y} log π_k) − 0.5·ls[y] with gathers only, so it holds z and ls, as before.
  - `mix` is the single eval path, used by `_logits`. That covers forward, forward_cached, eval_loss_splits, score_answers/CodeRecall, follow-rate and benchmark.
    - It is in place under no_grad, so there is no extra [B,T,V] copy.
  - The backbone sees no ICE unless `ice_v_input` is set (G_V_in reuses ICEInput). ICEOutput is not built for V arms.
- **config.py**
  - New fields: `ice_v`, `ice_v_input`, `ice_v_verify`, `ice_v_aux=0.5`. Their defaults keep old configs loading unchanged.
  - New arms: G_V, G_VR, G_V_in, G_V_nover, C_V, C_VR, all with ICE-dropout 0. `ice_param_count` covers the pointer.
  - At 140m: G_V adds 1,976 parameters, G_VR adds 2,746.
- **data.py**
  - `make_example`, `TrainWindows` and `val_batches` take `cand`. `TrainWindows` carries the alpha mask explicitly to the DataLoader workers.
  - `model_ice_keys()` gives the keys a model needs. `split_batch` passes the V keys through.
  - `alpha_mask_for_data(data_dir)` builds the mask from the tokenizer: the piece with ▁/Ġ stripped must satisfy `.isalpha()`. A synthetic corpus counts every token except EOS.
- **inference.py**
  - `ICEIndex(..., cand)` returns the same slots incrementally and through `prefill`.
  - VR keeps per-order prev pointers: int32 per order per token, which `compact_bytes()["prev"]` counts.
  - `features_to_tensors` and `step_features_to_tensors` pass the V keys through.
- **evals.py, scripts/train.py, scripts/evaluate.py, scripts/benchmark.py**
  - `cand` and the alpha mask are plumbed through. In the benchmark, features are sliced by key.

## Bit-identity of existing arms
A separate throwaway Modal app (`ice-lm-bitcmp`, scratchpad/bitcmp/) ran the frozen code and the new code on the same inputs.

It covered all 13 old arms: A, B, C, D, E, F, G, G_C, C_in, C_out, C_nodrop, C_notrust and B_pm. It compared:
- state_dict keys and values;
- chunked training loss and every gradient;
- eval logits with reduction="none";
- keep=0;
- cached prefill and decode;
- batch features;
- ICEIndex prefill and step outputs, plus compact_bytes.

Result: **BIT-IDENTICAL** on every one. The checkpoint format of old arms is therefore unchanged. New runs' config.json has 4 extra model keys.

## Bytes/token and cost
Measured on 32K tokens of val_code, excluding the fixed 256 KB order-1 table:

| Set | Index B/token | ICEIndex.step | Batch features per 4096 window (DataLoader worker) |
|---|---|---|---|
| C / V | 44.7 (V adds 0) | 48 µs (C), 62 µs (V) | 9.7 ms (C), 16.7 ms (V) |
| VR | 68.7 (+24 prev pointers) | 82 µs | 26.8 ms |

Next-token coverage on one 4K code window:
- s\*: 60.0%
- any V slot: 65.0%
- any VR slot: 69.1%

This is all tokens, not the D3 gate categories.

## Tests (tests/test_ice_lm.py, Modal CPU): 125 passed
New or extended tests:
- incremental vs batch features for cand ∈ {"", v, vr}, both `extend` and `prefill`, also comparing tables and prev pointers;
- V/VR slots vs a brute-force reference;
- the alpha mask must be set (both the batch builder and ICEIndex raise);
- arm configs and parameter counts (G_V = G + pointer exactly, G_V − G_V_nover = K+1, VR − V = 22·35);
- no leakage: perturb x[200:] and check that log-probs at t ≤ 199 are bit-identical, for G_V and G_VR. Both runs use a small alphabet, so anchor spans would reach t+1 if they were not masked, and the test also asserts the pointer is active;
- keep=0 output == log_softmax(z) bitwise, and logsumexp(mixture) = 0 within 1e-5, for G_V, G_VR and G_V_in;
- the grouped duplicate handling equals the sequential slot-by-slot definition, with duplicates present;
- chunked training loss == −mean log p_mix(y) + 0.5·CE(z), in value and in every gradient;
- cached decoding == full forward for all 6 V arms, plus a batch-2 prefill test for G_VR;
- forward/backward for all arms, including the V arms.

## Deviations from decision.md (and why)
1. **Duplicates in `_mix`.**
   - Every slot of token c gets the logsumexp of all slots of c, then a single gather/scatter writes logaddexp(out[c], ·). All writers carry the same value.
   - This is mathematically the sequential definition (a test checks this). It avoids K out-of-place [B,T,V] copies, which would be 34 × 4 GB for an eval batch.
   - With autograd on, the gradient is kept only on the first duplicate. Otherwise scatter's backward double-counts it. That was found by the gradient test and only affects the grad-enabled full path, not training.
2. **Finite −1e4 instead of −inf** where gradients flow (the training `hit` term and the slot masses in `_mix`). This avoids NaN gradients from logsumexp over all −inf. log π itself still uses −inf, so keep=0 stays exact.
3. **Anchor features.**
   - The match length is measured at the anchor, (a, q), and the distance is bucket(a − q). The decision does not specify these.
   - Anchors are the 2 most recent alphabetic *positions*, not distinct token ids.
   - The Llama vocabulary is ~94% "alphabetic" by this rule (30,158 of 32,000). In practice the mask excludes punctuation, digits, whitespace and byte tokens.
4. **Initialisation (my choice).**
   - b = −3, so each filled slot starts at ~0.05·π0.
   - γ = β = c0 = w = L = D = 0. The verifier terms start silent, like the other ICE zero-inits.
5. **Feature dtype.** Features are int64 tensors in the batch, as for the existing keys. The values fit in int8; only the transfer size differs, which is ~3 MB per sequence for VR.
6. **Logged training loss.** For V arms it is NLL_mix + 0.5·CE(z). The eval splits report the mixture NLL only.

## GPU smoke (H100, 140m, 3M tokens, micro_batch 8, `modal_app.py::run --gpu`)
Both runs finished with no NaN, and the eval splits printed at steps 10 and 20. The train loss includes 0.5·aux.

| Arm | Train loss, step 1 → 10 → 20 | tok/s, steps 2–10 / 11–20 | code/all eval, step 10 → 20 | code/copyable eval, step 10 → 20 | code follow_wrong, step 20 |
|---|---|---|---|---|---|
| G_V | 13.92 → 9.86 → 9.43 | 81.2K / 72.9K | 4.311 → 4.252 | 2.090 → 2.149 | 0.852 |
| G_VR | 13.62 → 9.86 → 9.51 | 81.3K / 71.4K | 4.493 → 4.420 | 2.599 at step 20 | 0.598 |

- Throughput is in line with G_C's ~76K tok/s. The second window includes the eval at step 10.
- At 3M tokens these numbers are only a plumbing check, not a result.
- The "ICE=0.00M" in the train.py banner is 1,976 / 2,746 parameters rounded to 0.00M.
