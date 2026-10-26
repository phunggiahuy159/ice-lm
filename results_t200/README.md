# 200M-token runs: can an exact lookup stand in for attention?

140M preset, 200M training tokens per arm, one seed, same data order, Modal H100 (workspace tien-phat140205).
Each `runs/<run>/log.jsonl` holds the training loss and the held-out evaluation every 250 steps;
`logs/run3_train_<arm>.log` is the full console output. The final evaluation (500 val windows, CodeRecall, ICE-off
control, figures) will be added as `results_t200.json` and `figs_results_t200/`.

| Arm | Model | Cache per token |
|---|---|---|
| G | 16 GDN layers, no attention | 0 B |
| G_C | G + ICE (original: input injection + logit bonus + trust signals) | ~41 B |
| G_V | G + ICE-V (lookup-blind backbone verifies 12 candidates, log-space pointer-sentinel) | ~45 B |
| G_VR | G + ICE-VR (V + deep suffix chain + identifier-anchor spans, 34 candidates) | ~69 B |
| B | 7:1 GDN/attention hybrid, the reference | 2,048 B (KV) |

**Held-out loss at step 1500 (~197M tokens).** Training-time evaluation uses only 8 val windows of 4K tokens per
split, identical across arms, so paired but noisy at the ~0.01 level.

| Arm | Code, all | Code, copyable | Code, lookup wrong | Follow wrong lookup | Prose, all |
|---|---|---|---|---|---|
| G | 1.986 | 1.449 | 3.295 | 0.203 | 3.377 |
| G_C | 1.852 | 1.112 | 3.478 | 0.306 | 3.307 |
| G_V | 1.794 | 1.079 | 3.339 | 0.340 | 3.265 |
| G_VR | **1.777** | **1.068** | 3.288 | 0.340 | **3.251** |
| B | 1.816 | 1.138 | **3.245** | 0.234 | 3.316 |

* **No attention, with ICE-V/VR, is level with or better than the 7:1 hybrid** on code and prose loss, at ~2–3% of
  its cache memory.
* **The old ICE's harm on misleading lookups is gone with the verifier.** Where the lookup is wrong, G_C is 0.18
  worse than G; G_VR is level with G.
* Gains are ordered G < G_C < G_V < G_VR on all-token code loss.
* Open:
  * top-1 still lands on a wrong lookup ~34% of the time for V/VR (vs 23% for B), although the loss there is fine;
  * one seed;
  * the final 500-window evaluation is pending.
