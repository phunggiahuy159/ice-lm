# ICE-V2: implementation notes (decision2.md §2)

Code: /Users/phunghuy/code/gstar/proposal_ice/ice_lm. Pre-change copy: scratchpad/ice_lm_frozen_v2pre.

**Status.** Everything was implemented and tested locally on CPU. Modal was off limits: the workspace hit its spend limit, so neither the Modal CPU test run nor the GPU smoke ran. See "Still to run on Modal".

## What changed
- **ice_features.py**
  - `compute_ice_features(..., cand, v2=False)`. With `v2` it also returns:
    - `ice_v_src` [K,T]: the source position i_k (−1 = empty). This is not a model input.
    - `V2_KEYS = (ice_v_delta, ice_v_cnt, ice_v_rec)` [K,T], computed by `v2_features(tok, ln, src, n_occ)` with one K×K compare per position.
  - The features are encoded as follows:
    - **Δ:** bucket `8 + clip(len_k − max{len_j : j valid occurrence slot, c_j ≠ c_k}, −8, 8)` (0..16). Bucket 17 means "no competitor". Anchor slots keep their own anchor match length and never enter the max.
    - **n:** `clip(#same-token valid slots, 1, 6)`.
    - **r:** recency rank of m(c_k) = the max source index among distinct tokens, newest = 0, clipped to 3.
    - Empty slots get 0.
  - New helpers: `n_occ_slots`, `slot_groups`. Groups are last 0, second 1, chain 2, anchor 3; V has groups 0/1 only.
  - The value distribution on real val_code (one 4K window) is sane. All buckets are used, and 17% of filled slots have "no competitor".
- **inference.py**
  - `ICEIndex(..., cand, v2=False)` also returns `ice_v_src` and the V2 features from step, extend and prefill.
  - The features are an independent O(K) pure-Python implementation (per-token count / newest / best length, and the top-2 competitors), not a call to the batch code. They need no extra state: 0 bytes per token.
  - `features_to_tensors` and `step_features_to_tensors` pass `V2_KEYS` through.
- **model.py**
  - `ICEPointer(..., v2=True, groups, feats)` adds:
    - zero-init `nn.Embedding` tables R (18 buckets), A (7), N (4) and Q (3) per group, indexed as g·nb + feature;
    - `gamma_t` with softplus;
    - no β.
  - q_k comes from `ICEPointer.q_rank`: the number of *distinct other* slot tokens with a strictly larger detached ℓ, clipped to 2.
  - The old (non-V2) path is unchanged code. `log_pi`, `mix` and `loss_sum` take the V2 features as `*v2f`.
  - Listwise term: `loss_sum(..., listwise=λ)` adds λ·1[y ∈ slots, ≥ 2 distinct tokens]·(logsumexp_valid log π − logsumexp_{c_k=y} log π). The sentinel cancels. It runs inside the existing chunked, checkpointed loss and uses gathers and K×K masks only.
  - `_out_feats` passes the V2 keys when `ice_ptr.v2`.
- **config.py**
  - New fields: `ice_v2=False`, `ice_v2_feats="dnrq"` (which tables exist, for the F2 "keep R/A/N only if …" decision) and `ice_v_listwise=0.0`.
  - New arm `G_VR2`: G_VR + V2 + listwise 0.25, with G_VR's gradient paths.
  - `ice_param_count` covers V2. At 140m, G_VR2 has 2,873 ICE parameters against G_VR's 2,746: +128 for the tables, −1 for β.
- **data.py, evals.py, train.py, benchmark.py, diagnose_pointer.py**
  - `v2` is plumbed through `model_ice_keys`, `make_example`, `TrainWindows`, `val_batches`, `split_batch` and the `ICEIndex` constructor.
  - diagnose_pointer computes VR features with `v2=True`.
- **coderecall.py:** `build_items(..., make=None)` accepts another template maker. The default behaviour is unchanged.
- **New `ice_lm/syn_conflicts.py`**
  - **FIT families** (fit windows only):
    - `register_queue(topic=…, <1–4 shared kwargs>, max_inflight=v)` conflicts;
    - `limits["team"] = {…, "cpu_millis": v}` conflicts;
    - `opts.max_jobs = v1` followed by ≥ 2 later writes via `setattr(...)` / `opts.set(...)`, asked with the original syntax, so the answer is the newest write.
  - **HELD-OUT perturbed families** (eval only):
    - positional `add_route("/x", "GET", 30, True, v)`;
    - `Probe(kind=…, port=…, host=…, <shared>, budget=v)`, with shared fields on both sides of the identifier;
    - `self.k = v1` in a class, then 4–5 writes via `app.state.k` / `patch_state(k=)` / `runtime_set("k", …)`, asked as `self.k =`.
  - Every family has ≥ 3 records or writes and 4-digit values with distinct leading digits, and is scored on the deciding digit through `coderecall.build_items` and `score_answers`. None uses CodeRecall's names or syntax; a test checks this.
  - `item_pool` pre-tokenizes the items in the main process. `syn_window` packs items with real-token gaps (32–768) and a random lead-in. `SynMixWindows` replaces sample i with a synthetic window with probability `syn_frac`, fixed per index.
  - `run_syn_eval` / `summarize_syn` report `syn/<family>@<gap>/*`, plus `fit_mean` (fresh FIT items) and `heldout_mean`.
- **scripts/fit_scorer.py**
  - New flags: `--v2`, `--v2_feats`, `--listwise λ`, `--syn_frac`, `--syn_pool`.
  - `--init_from` with an old scorer copies b, L, D, w, norm and c0, and sets γ̃ = softplus⁻¹(max(γ, 1e-3)). β is dropped. The V2 tables start at 0. The script logs what was fresh or clamped.
  - The run tag is `fit_vr2` or `fit_vr2+syn`; the "+syn" marks a task-supervised fit.
- **scripts/evaluate.py:** `--n_syn N` (default 0, off) and `--syn_gaps` add the synthetic-template table.

## Deviations / choices
1. **Fresh γ̃ init.** γ̃ starts at softplus⁻¹(0.3), as in proposal A. γ = 0 is not reachable through softplus.
   - So "V2 starts as the current scorer" holds exactly only for the zero tables.
   - With `--init_from` it holds up to two things: γ clamped at ≥ 1e-3 (G_VR's o8_last γ is −0.17), and the dropped β·max ℓ (inert, std 0.16). A test checks that zero tables plus γ̃ = softplus⁻¹(γ) reproduce the old mixture to 1e-5.
2. **Ties and clipping in q.** q uses the raw detached ℓ and counts strictly larger values, so ties do not raise the rank. The γ term still uses ℓ clipped at −20.
3. **Δ of anchor slots** follows the spec literally: the anchor's own match length minus the best competing *occurrence* length. R_anchor is a separate table, so the scale mismatch is learnable.
4. **Listwise indicator.** "≥ 2 distinct tokens" counts valid slots. With keep = 0 the term is 0.
5. **V2 with cand "v"** is supported (2 groups). It is not in decision2.
6. **Synthetic windows** are trained with the full-window loss, like every fit window. The gap filler comes from train.bin; the evaluation filler comes from val_code, as in CodeRecall.
7. **Cost.** On one real 4K code window (laptop) the batch features take ~22–27 ms without V2 and ~43 ms with it; `v2_features` uses int16/int32 temporaries. `ICEIndex.step` goes from 39 to 64 µs. There are 0 cache bytes.
8. **macOS only.** DataLoader workers need the `fork` start method (scratchpad/lruns/forkrun.py). Linux and Modal fork by default, so there is no code change.

## Tests (local CPU, scratchpad/lvenv: torch 2.14.1, numpy 2.4.6, ICE_CPU_ONLY=1): **169 passed**
The frozen pre-change suite had 126 tests and also passes on the same venv. Nothing in the suite needs fla or CUDA, so nothing was skipped. New or extended tests:
- V2 Δ/n/r and `ice_v_src` vs a brute-force reference (`_brute_v2`, from decision2 §2), for V and VR. The V slots themselves are unchanged with v2.
- decode == batch for V2 (cand v/vr with v2): step, extend and prefill, including `ice_v_src`, tables and prev pointers.
- `q_rank` vs its definition, with ties.
- keep = 0 is bitwise log_softmax(z), and logsumexp = 0, for G_VR2, G_V+v2, G_VR+v2 (dq tables only) and C_VR+v2 without listwise.
- Chunked training loss == −mean log p_mix(y) + 0.5·CE + λ·listwise. The listwise reference is computed position by position from its definition. Values and every gradient match, and the term is active at > 10 positions.
- No leakage: perturbing x[200:] leaves log-probs at positions ≤ 199 bit-identical, with the pointer active, for all V2 variants.
- Cached decoding == full forward for G_VR2 and the V2 variants, plus batch-2 prefill for G_VR2.
- The eval mixture equals the sequential definition for V2.
- Parameter counts for V2, and forward/backward for G_VR2.
- V2 starts as the V scorer; state-dict keys of old and V2 pointers.
- Synthetic templates: no CodeRecall names or syntax, ≥ 3 records/writes, the answer appears once, the deciding digit is the span start. `SynMixWindows` mixes at about syn_frac and carries the V2 keys. These use an offline character tokenizer.

**Old-arm bit equality.** Script: scratchpad/bitcmp2 (`dump.py` / `cmp.py`), frozen vs new code, same torch. It covers **20 arms**: A, B, C, D, E, F, G, G_C, G_MQA, C_in, C_out, C_nodrop, C_notrust, B_pm, G_V, G_VR, G_V_in, G_V_nover, C_V and C_VR. It compares:
- state_dict keys and values;
- chunked training loss and every gradient;
- eval logits and per-token losses;
- keep = 0;
- cached prefill and 20 decode steps;
- `ice_param_count` and `ice_params`;
- feature dicts and ICEIndex outputs plus compact_bytes for cand "", v and vr.

Result: **BIT-IDENTICAL**. The only config difference is the 3 new default fields. A sanity check confirmed the comparison flags a 1e-7 relative change in a gradient.

**Local end-to-end (CPU, toy).** Plumbing check only; the numbers mean nothing.
- Data: val_prose as train.bin, tokenizer in scratchpad/tok.
- `train.py --preset smoke --arm G_VR` for 60K tokens.
- `fit_scorer.py --v2 --listwise 0.25 --syn_frac 0.2 --init_from <base>` ran: 1,593 trainable parameters, the R/A/N/Q tables fresh, 200 synthetic items, and the run saved.
- `evaluate.py --n_syn 4` printed the synthetic table.
- ICE-off code loss is identical for base and fit (9.1845), as it must be with a frozen backbone.

## Still to run on Modal (once the budget is back)
1. `ICE_CPU_ONLY=1 MODAL_PROFILE=run3 …/mvenv/bin/modal run modal_app.py::tests_cpu`: the full suite under the image's torch 2.6.
2. Optionally, rerun scratchpad/bitcmp2 under torch 2.6. The comparison is frozen vs new on the same torch, so it is version-independent in principle.
3. The GPU smoke of a V2 fit (not run). Suggested, with the syn path and held-out eval added: `modal run modal_app.py::run --gpu --cmd "python scripts/fit_scorer.py --base_run /vol/runs/t200_G_VR --cand vr --v2 --listwise 0.25 --syn_frac 0.05 --data_dir /vol/data --out_dir /vol/runs/smoke_v2 --tokens 2e6 && python scripts/evaluate.py --runs /vol/runs/smoke_v2 --data_dir /vol/data --val_batches 20 --n_task 5 --n_coderecall 10 --n_syn 10 --no_ice_off --out /vol/diag/smoke_v2.json"`.
   - Watch the fit's tok/s. V2 roughly doubles the batch feature cost (~43 ms per window), and with 6 workers the loader may become the bottleneck.
