# Novelty / prior-work critique

## 1. ID verification (all opened on arxiv.org/abs)
| id | Actual title | Matches claims? |
|---|---|---|
| 2607.02303 | HOLA: A Hippocampus for Linear Attention (Cui, Jul 2026) | Yes: state + bounded exact KV cache, 340M, needle recall to 32k |
| 2606.29844 | MATCH: Modulating Attention via In-Context Retrieval for Long-Context Transformers (Jun 2026) | Yes (abstract does not mention Sentence-BERT; that detail is from the lit report, unchecked) |
| 2609.35578 | FactorEngram: Factorized N-gram Memory with Basis-Level Gating (Sep 2026) | Yes |
| 2601.07372 | Engram: Conditional Memory via Scalable Lookup (v2 Jul 2026) | Yes |
| 2401.12973 | In-Context Language Learning: Architectures and Algorithms (Akyürek) | Yes: n-gram heads |
| 1609.07843 | Pointer Sentinel Mixture Models | Yes |
| 2411.04975 | SuffixDecoding | Yes |
| 2311.08252 | REST: Retrieval-Based Speculative Decoding | Yes |
| 2609.07681 | Recall Scaling Laws in Mamba via Hashing | Yes |
| 2609.15545 | The Token Before the Value Is the Key (hybrid induction circuits) | Yes |
| 2606.15378 | Rethinking the Role of Efficient Attention in Hybrid Architectures | Yes |

No bad ids. Also found by search: LLMA / Inference with Reference, 2304.04487 (not cited by any proposal), RetrievalAttention 2409.10516, ReplaySSM (Tri Dao blog / NVIDIA, 2026), RAD 2505.22135. Not re-opened: SAM-Decoding 2411.10666, NEST 2405.19325, kNN-LM ids (cited by P1, P3 from reports).

## 2. Proposal 1: ICE-V (verified candidate pointer)
Missed or underplayed prior work:
- Pointer-generator copy with learned gate (See et al. 1704.04368, cited) and the whole "copy-or-generate" family. The mixture form is textbook.
- kNN-LM / Efficient kNN-LM / Drozdov adaptive gating on LM entropy (cited). The "gate from the LM's own uncertainty plus the candidate's LM probability" is essentially adaptive kNN-LM with candidates as the datastore.
- Speculative-decoding acceptance (cited) is only an analogy, not a method.
- Unmentioned: contrastive/context-aware decoding style "does the LM already agree" scoring, and the 2020 implicit-cache-pointer LM (ar5iv 2009.13774), which adds history tokens to the output layer.
Novelty: **3/10**. Defensible claim: a pointer-sentinel mixture whose gate is fed by a detached lookup-free backbone, over an index-derived candidate set at ~41 B/token, so ICE-off equals the backbone by construction. Weakness: the 0.5 aux CE is the only thing that makes "C_off = B" true, and ablation V_nover is the entire scientific content.

## 3. Proposal 2: ICE-Route (index-routed pointer)
Missed or underplayed prior work:
- Retrieval-routed sparse attention: InfLLM, Landmark, RetrievalAttention (2409.10516), Unlimiformer, kNN-augmented transformers (Memorizing Transformers). All pick positions by a similarity index then attend. ICE-Route differs by using an exact symbolic index, but the "index proposes, learned attention selects" pattern is the shared idea. P2 compares only to MATCH, HOLA and Akyürek and skips these.
- Neural cache / pointer networks (cited as output form only).
- Storing a 64-d vector per token is a compressed KV cache; the "10-16x below KV" claim should be compared against KV compression and HOLA's bounded cache, not only against full KV.
Novelty: **5/10**, the highest of the three. Defensible claim: symbolic multi-occurrence and skip-gram anchors as the candidate generator for a pointer read in a recurrent hybrid, at ~10x less memory than KV. Risk: reviewers will call it "sparse retrieval attention with a hash index." R_tok is the key ablation for the claim.

## 4. Proposal 3: ICE-Draft
Missed or underplayed prior work:
- LLMA / Inference with Reference (2304.04487): copying spans from context and verifying in parallel; the direct ancestor of PLD. Not cited.
- Lookahead decoding, Medusa/EAGLE (mentioned, unverified), SAM-Decoding, NEST (cited).
- Speculative decoding on SSMs and hybrids is NOT "unverified existence": Mamba-in-the-Llama (hardware-aware spec decoding for Mamba), RAD (2505.22135), SGLang hybrid support, and ReplaySSM (Dao lab/NVIDIA, state rollback via input replay, ~1.9x) all exist. The proposal's "GDN rollback" claim is therefore not novel and ReplaySSM is a better rollback method than snapshot/restore.
- The run-length head is an acceptance/length predictor like SuffixDecoding's adaptive length and EAGLE-2's confidence trees; the free label is mild novelty.
Novelty: **3/10** for speed; **4/10** counting the dual-use run head as trust signal. Defensible claim: a drafter trained jointly with the lookup it drafts from, where one free LCP label supplies both draft length and a trust feature. Also the "lossless" claim is violated in training-influenced ways only if the model is changed; fine. Biggest conceptual problem: decode speed does not address any reviewer item (over-reliance, conflicts), and greedy verification will rubber-stamp copying.

## 5. Ranking and advice
Real novelty: P2 (5) > P1 (3) = P3 (3-4). All three are recombinations. The strongest honest story is a negative-result-friendly one: P1 as a cheap, decisive diagnostic (step-0 AUROC gate is excellent and nearly free), P2 for the beyond-exact claim. None should claim "first"; add citations to LLMA, ReplaySSM, RetrievalAttention, InfLLM, Memorizing Transformers, kNN-LM adaptive gating, pointer-generator, and compare memory against HOLA.
