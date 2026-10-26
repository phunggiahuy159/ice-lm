# Next research round: instructions for the session that resumes (written 26 Oct, 23:55 UTC+7)

Read first: `HANDOFF.md`, `results_t200/README.md`, and everything in `research/round2/`. In that folder:
* `brief2.md`, `diagnostics.md`, `real_models.md`;
* `proposal_A.md` (ICE-V2: relative slot features + listwise loss + detached backbone, all fitted on frozen
  models);
* `proposal_B.md` (evidence: bolt-on, Mamba2/Pythia, A, sliding-window baseline, Qwen3.5);
* `proposal_C.md` (ICE-K: 16-byte learned keys);
* `critique_tech2.md`, `critique_reviewer2.md`.

## State at hand-off
* 200M runs t200_{B,G,G_C,G_V,G_VR} are done and evaluated: `results_t200/`.
* Benchmark: `/vol/benchmark_t200.json`. B 66.8 MiB per 32K sequence vs G_VR 5.5 MiB; G_VR's table is
  64.7 B/token.
* Seed 2 of B and G_VR (`t200s2_*`) was finishing. `evaluate_when_done` is queued on Modal → `/vol/results_t200_seed2.json`.
  Seed 2 matched seed 1 at step 1000: G_VR − B = −0.127 (seed 1: −0.117).
* Diagnostics (round2/diagnostics.md), the key facts:
  * the correct VR candidate is ranked ~11–13 because slot features are absolute (no "longest wins" feature);
    the longest-match rule alone would get 97–98% on conflict_g2/g5;
  * the backbone log-prob is uninformative on conflicts;
  * laziness sits only on lookup-right tokens, through the non-detached backbone path;
  * G_C's increment +0.86 is a real "s*+1" heuristic.
* Critics agree on:
  * seeds first;
  * the ICE-V2 ladder on frozen models (cheap; "frozen G + fitted scorer" IS the detached method, expected
    ~0.02 worse than G_VR);
  * a sliding-window baseline (G_SWA) with an equal cache at 32K (window ≈ 2,560; generous at 4K, so say so);
  * Mamba2-370M vs Pythia-410M bolt-on (~$3);
  * ICE-K only if its $1 probe passes;
  * Qwen3.5-0.8B last or cut (new image, 248K vocab, 1–1.5 days; attention already copies, so likely null).

## Steps
1. **Judge.** A fresh judge agent reads round2/ and writes `research/round2/decision2.md`: the final ordered plan,
   predictions, kill criteria, a budget of ≤ $30.
2. **Collect seed 2:** `results_t200_seed2.json` from the volume (profile run3). Launch seed 3 of B and G_VR if the
   judge keeps it.
3. **Free checks first** (CPU):
   * does "longest match / latest write" hold on real code (not only on CodeRecall)?
   * the ICE-K token-only oracle.
4. **Implement ICE-V2** (proposal_A) as a scorer fit on frozen checkpoints (new script, e.g.
   `scripts/fit_scorer.py`). Run the ablation ladder on frozen G / G_VR / B. Evaluate with the existing evaluate path.
5. **Baselines:** G_SWA (needs a sliding-window attention layer option in `model.py`: keep old arms bit-identical,
   add tests); A at 200M if the budget allows.
6. **Real models:** Mamba2-370M and Pythia-410M with a bolt-on ICE-V2 scorer (data re-tokenized; CodeRecall rebuilt
   for that tokenizer, multiple-choice if digits merge).
7. **Every round:**
   * tests on Modal CPU (`ICE_CPU_ONLY=1 ... tests_cpu`) before GPU;
   * results into `results_*/README.md` + `HANDOFF.md`;
   * commit + push to `huypg-zenai/ice-lm`;
   * use fresh advisor/critic agents for any major decision.

## Rules
* Modal: CLI `/private/tmp/claude-501/-Users-phunghuy-code-gstar/a1fcb2a4-ae79-48f6-b6e8-350e5f8c37f0/scratchpad/mvenv/bin/modal`,
  profile `run3` (workspace tien-phat140205: never touch other projects' volumes/secrets).
  * H100 ≈ $3.95/h; stay within ~$30 for the round.
  * If "spend limit" errors appear: stop, save logs, report.
* Do not write credentials into files. Old arms must stay bit-identical. Pre-register predictions before runs.
