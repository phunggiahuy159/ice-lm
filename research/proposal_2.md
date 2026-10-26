# Proposal 2: ICE-Route (index-routed pointer read)

## Name
**ICE-Route**: the exact index stops being the answer and becomes a *router*. It proposes about 32 candidate positions. A small learned pointer-attention over 64-d per-token vectors picks one of them, or abstains.

## One-paragraph idea
C's failures come from *where the single candidate comes from*:
- **conflict_g5**: the right value follows an *older* occurrence of the same 8-gram.
- **override_diff / copy_diff**: the right value follows an earlier mention of the *key identifier*, while the suffix at t is just syntax (`] =`, `==`, `default=`).

ICE-Route keeps the CPU index but uses it to list candidates:
- **suffix chains**: the last 4 occurrences of the n-gram, not only the latest;
- **skip-gram anchors**: earlier occurrences of rare identifier tokens from the last 8 tokens, plus the 4-token span after each.

The model chooses among the candidates by content and copies through a pointer-sentinel that can always abstain. Exact matching decides *where to look*; learned matching decides *which match is right*.

## Exact mechanism
**CPU** (`ice_features.py`, about 80 lines; the order-n `prev` arrays already exist inside the loop and are thrown away today):
- Suffix block, 8 slots:
  - 4 slots: p+1 for the 4 most recent occurrences of the longest-matching n-gram, chaining p, prev[p], and so on. Slot 0 is today's s\*.
  - 4 slots: the same for the order-2 gram.
- Anchor block, 24 slots:
  - Anchors are the 3 most recent *content tokens* in x[t−7..t], meaning tokens outside the 512 most frequent. Ranks come from one `np.bincount` over the train memmap.
  - For each anchor: its 2 most recent earlier occurrences q, plus the span q+1..q+4.
- Output: `ice_cand` [T, 32] int32, with −1 for an invalid slot. Every slot satisfies c ≤ t and stays in the same segment. A slot index always means the same source, so no source id is stored.

**Model** (`model.py`, new `ICERoute`; h^(ℓ) is the residual after block ℓ):
- Stored per token: m_p = W_s RMSNorm(h_p^(2)) ∈ R^64.
- Slot key: κ_{t,k} = m_c + W_e E[x_c] + S_k + D[bucket(t−c)], with c = cand[t,k].
- Read after block 4: α = softmax over {valid k} ∪ {∅} of (W_q h_t^(4))·κ/8. Then h_t += W_o Σ α_k κ_k, with W_o zero-initialised.
- Pointer at the final layer: β is the same form with its own query, and the sentinel logit starts at +4. Then p(y) = β_∅ p_LM(y) + Σ_{k: x_{c_k}=y} β_k. This replaces C's additive bonus, so a wrong copy now costs probability mass directly (over-reliance).
- Loss: inside the existing chunked `_loss_sum`, NLL = −logaddexp(log β_∅ + logp_LM[y], log Σ_k β_k 1[x_{c_k}=y]). It only needs a [B,T,K] comparison with the labels.
- C's ICEInput stays. 15% of positions get all slots masked (candidate dropout).

**Causal and parallel**: c ≤ t, and m_c depends only on tokens ≤ c. Training is one `torch.gather` over h^(2) with a [B,T,32] index in the same forward pass, so there is no recurrence and the whole path is differentiable. The gathered tensor is about 67 MB per micro-batch.

**Files**: `ice_features.py`, `data.py` (`ice_cand` in ICE_KEYS, rank file), `model.py`, `config.py` (arms), `evals.py` (coverage). `inference.py` (prev pointers and a GPU [max_len, 64] store) is needed only for the decode benchmark.

## Why it handles cases exact ICE cannot
- **conflict_g2/g5**:
  - The candidates include every server's `port=` successor.
  - The query carries `name="staging"`, about 30 tokens back, which is within GDN reach.
  - Each m_c carries the name from its own line, so the choice becomes a content comparison.
- **override_diff**: the anchor `timeout` hits `config.update(timeout=60)`, and its span contains `6`. The newer value can be preferred through D and S_k.
- **copy_diff**: the anchor `db`/`port` hits the dict entry, and its span contains the digit.

**Day-0 gate (CPU only, free)**:
- Measure the fraction of CodeRecall deciding tokens that fall inside the 32 candidates, per category.
- Measure top-32 coverage on real code, against the 46.7% for s\*.
- If coverage is below 70% on any target category, fix the router before spending GPU time.

## Memory and speed
| Component | Bytes/token |
|---|---|
| ICE as now | ~41 |
| Prev pointers | 24 |
| m_t | 128 (fp16) or ~66 (int8, unverified) |
| **Total** | **~130–190** |

- That is about **10–16× below B's 2,048 B/token KV**. Moving from B to A costs +2,048 B/token.
- Decode reads 32×64 values per step, independent of T.
- Extra training FLOPs should be under 5%. I will measure tokens/s, not assume it.

## Experiment
Three arms, 140M, 400M tokens, same seed and data as B, C and C_notrust (A also needs one run if it does not exist yet):

| Arm | What it is | What it isolates |
|---|---|---|
| **R** | full method | – |
| **R_noanchor** | suffix chains only | contribution of the anchors |
| **R_tok** | no stored m_c; key = token + slot + distance; ~0 extra B/token | learned content selection |

**Metrics**:
- CodeRecall deciding-token accuracy per category, 300 items paired against C with bootstrap.
- Code loss: all, copyable, lookup-wrong.
- Follow-wrong rate.
- ICE-off loss vs B.
- Measured B/token and tokens/s.

**Pre-registered predictions**:
- R − C ≥ +0.15 on override_diff and on copy_diff.
- R − C ≥ +0.10 on conflict_g5.
- Code loss within +0.02 of C.
- Follow-wrong rate below 0.356.
- R_noanchor gets less than half of R's gain on copy_diff and override_diff.
- R_tok loses most of the conflict_g5 gain.

**Kill**: drop the "beyond exact" claim and report the negative result if any of these holds:
- R − C < 0.10 (95% CI excluding 0) on any of copy_diff, override_diff or conflict_g5;
- code loss is worse by more than 0.03;
- throughput falls by more than 25%.

The standard error is about 0.04 per difference. With one seed, the result is preliminary.

## Novelty vs closest papers
- **MATCH (2606.29844)**, the closest:
  - MATCH unmasks retrieved chunks inside SWA-transformer attention, using Sentence-BERT retrieval and recomputing QKV.
  - ICE-Route is token-level, uses an exact plus skip-gram CPU index, runs in a GDN hybrid, stores 64-d per token, and reads through a pointer-sentinel.
- **HOLA (2607.02303)**: full KV for tokens chosen by learned salience. ICE-Route selects by symbolic index and costs about 10× fewer bytes.
- **Akyürek (2401.12973)**: n-gram matching inside attention over all T, so O(T) KV. ICE-Route reads 32 indexed slots.
- **N-Grammer / Engram**: static and parametric. ICE-Route is in-context and non-parametric.
- **Pointer-sentinel / neural cache**: the same output form, but over a whole recent window. Index routing is the new part.

## Biggest risk
The router finds the candidates, but *selection* is not learned: patterns like "newest assignment across syntaxes" or "value of the named object" may be too rare in 400M tokens, giving R ≈ C. Coverage numbers together with R_tok tell a router failure apart from a selection failure, so a negative result is still informative. Smaller risks:
- The anchor rank threshold needs a check.
- The int8 storage of m is untested.
