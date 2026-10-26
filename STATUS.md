# Status (26 Oct 2026, evening)

Running on Modal (workspace `tien-phat140205`, profile `run3`), all detached, no Mac needed:

* 5 training runs, 200M tokens each, H100: `t200_B`, `t200_G`, `t200_G_C`, `t200_G_V`, `t200_G_VR`
  (G = no attention; G_C = + ICE; G_V = + ICE-V verifier; G_VR = + ICE-VR candidates; B = 7:1 reference).
* `evaluate_when_done` waits for all 5, then writes `/vol/results_t200.json` and `/vol/figs_results_t200/`.

Next steps (from `proposal_ice/ice_lm`, `M=<modal binary>`, `export MODAL_PROFILE=run3`):

```bash
$M volume get ice-lm results_t200.json ../results_t200/ --force
$M volume get ice-lm figs_results_t200 ../results_t200/ --force
for r in t200_B t200_G t200_G_C t200_G_V t200_G_VR; do mkdir -p ../results_t200/runs/$r
  $M volume get ice-lm runs/$r/log.jsonl ../results_t200/runs/$r/ --force; done
# W&B upload of the 5 runs (they trained before the wandb secret existed)
$M run modal_app.py::run --cmd "python scripts/wandb_backfill.py /vol/runs/t200_B /vol/runs/t200_G /vol/runs/t200_G_C /vol/runs/t200_G_V /vol/runs/t200_G_VR"
```

If the evaluation failed, re-run it:
`$M run modal_app.py::evaluate --runs t200_B,t200_G,t200_G_C,t200_G_V,t200_G_VR --n-task 50 --val-batches 500 --out results_t200.json`

Background: `research/decision.md` (the plan), `research/checks_results.md` (coverage gate, long context),
`research/impl_notes.md` (ICE-V/VR implementation), `feedback_response.md` (results so far).
