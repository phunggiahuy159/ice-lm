# Reviewer 2: what to run before the final presentation

## Where the project stands for a reviewer
The headline, "G_VR beats B at 1/30 the cache", rests on one seed (a 0.040 gap), on quality measured at 4K while memory is measured at 32K, and on table bytes that are estimates (the benchmark has not been run on the G arms). Our feedback asked for three things: conflicting values, corrections, and cases where copying is wrong. On exactly those, every ICE arm is at or below B (override_diff, conflict_g2/g5, copy_diff), and the pre-registered VR test failed. At this point the honest story is "cheap copying with fewer side effects, but it cannot choose between occurrences". The next round has to change that sentence or own it.

## Essential vs nice-to-have
**Essential (without these I would score the evidence as weak):**
1. **Seeds.** At least 3 seeds each of B and G_VR. Without them, "beats B" turns into "matches B".
2. **A same-memory attention baseline.** G_SWA (W ≈ 2,560, Proposal B), plus an eval at 8–32K for G, G_VR, B and G_SWA. This is the single most dangerous objection, and so far nothing answers it.
3. **Measured memory and speed for the G arms.** Bytes/token, decode tok/s, and index overhead, with the Python index reported honestly. The feedback asked for this explicitly.
4. **A conflict fix with a natural-data check.** Proposal A's T0 / Proposal C's P0 must come first. If "longest match wins" and "newest write" don't hold on real code, then a Δ/ν feature only fits our own template.

**High value:** Proposal A's V2 ladder on frozen G / the G_VR backbone / B, and Mamba2-370M vs Pythia-410M.
**Nice-to-have:** A at 200M, ICE-K (only behind its $1 probe), and Qwen3.5.

## What I would attack in Q&A
**Proposal A (V2, detach = frozen G + scorer)**
- "Δ and ν are hand-made features for your benchmark: show that the rule holds on real code (P0) and that the gain transfers to a held-out template."
- "With detach, this is Grave's neural cache / pointer sentinel / Bakhtin 2018 on a frozen LM. What is new?" Defensible answer: exact-match candidates verified by the backbone's own ℓ, at ~65 B/token, on recurrent LMs. That novelty is modest, so say so.
- "You predict code 1.705–1.735, so you lose the 'beats B' headline." The student must choose in advance which claim survives.

**Proposal B (evidence)**
- "A 2,560-wide window covers 62% of a 4K eval window, so a loss there means nothing." The reverse is the risk: if G_SWA wins, the memory headline dies. That is exactly why it must run.
- "Multiple-choice CodeRecall on NeoX is a different metric from deciding-digit accuracy, so you can't put them on one plot."
- "Is a scorer on a frozen model just kNN-LM?" Mostly yes, and the real_models.md novelty wording should be used verbatim.

**Proposal C (ICE-K)**
- "This is tiny attention over 34 index-selected positions, i.e. Quest/NSA with the value path removed. Compare it to a sliding-window layer, and to top-k attention with 16-d keys, at the same bytes."
- "A 16 B key per token is a per-token hidden vector. Your 'no KV cache' line is now 'small KV cache'."
- "If the token-only scorer matches it, the keys are decoration." C's own kill criteria say this already, which is good.

**All proposals**
- "One seed."
- "increment": drop the G_C claim; it is a +1 template prior (diagnostics §4).
- "Your model follows wrong lookups more often than attention (34% vs 24%)." Present this as calibrated hedging, with the AUROC/ECE numbers, rather than hiding it.

## Real model vs method fixes
Method fixes come first, because they answer the actual faculty critique: conflicts and corrections. A real model does not touch that. Of the real models, **Mamba2-370M vs Pythia-410M** is worth the money: same data and same tokenizer, a known recall deficit, no fla dependency, about $3. It turns "toy 140M" into "closes X% of a real recurrent-to-Transformer gap".

**Qwen3.5-0.8B** is the weakest buy:
- It needs transformers 5.x, fla and a 248K vocab.
- It has 6 attention layers that already copy, and our own 7:1 result predicts a gain of about 0.
- A null result there only restates weakness 5, at the highest engineering risk.

Run it last, only from the reserve, and frame it as a "does no harm" check.

## Priority-ordered plan (≈ $30)
| # | Run | $ | Gate |
|---|---|---|---|
| 0 | CPU: T0/P0 rule check on natural code; memory/speed benchmark of G, G_VR, B | 0.5 | P0 < 0 → drop the conflict_g claim |
| 1 | Seed 2 (running) → seed 3 of B and G_VR | 6 | — |
| 2 | G_SWA W = 2,560 at 200M, plus the 8–32K long-context eval | 4.5 | If G_SWA ≤ G_VR fails, the headline becomes "copy module", not "memory" |
| 3 | V2 ladder (A's T2): frozen G, the G_VR backbone, B; seed-2 G for free | 5.5 | P1/P2 decide the conflict story |
| 4 | Mamba2-370M vs Pythia-410M bolt-on (loss splits + MC CodeRecall) | 3 | Premise: a copy > 512 gap ≥ 0.10 |
| 5 | ICE-K probe P1 on frozen G_VR ($1); full G_VRK run only on a "go" | 1 + 3 | Must beat the token-only scorer by ≥ 0.15 |
| 6 | A at 200M | 3 | — |
| 7 | Reserve (reruns, or a Qwen3.5 smoke if nothing has failed) | ~3.5 | — |

If one thing must be cut, cut item 6 before item 2. Freeze the pre-registrations for items 2–5 before launching anything.

## The claim I would accept if this succeeds
"On 140M models (3 seeds), a pure recurrent model with an exact in-context index and a ~2K-parameter verifier matches a 7:1 attention hybrid on code and prose, and beats one sliding-window attention layer at equal memory, using a measured ~65 B/token (5.5 vs 66.8 MiB at 32K). It also resolves the 'latest value' and 'which occurrence' conflicts that the lookup alone gets wrong. Bolted onto frozen Mamba2-370M, it closes X% of the long-copy gap to Pythia-410M. It adds nothing once attention is present, and it does not solve paraphrased requests (copy_diff)."

The last sentence is not optional: knowing where the method stops is what makes the rest of the claim believable.
