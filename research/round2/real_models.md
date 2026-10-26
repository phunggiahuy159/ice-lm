# Real pretrained models for ICE-V bolt-on (verified 26 Oct 2026 by opening HF/arXiv pages)

"Verified" = read on the page. "Unverified" = from memory/papers, not on the page. No model was run; torch 2.6 compatibility is NOT confirmed by any page (fla needs a recent torch/triton; check `pip install flash-linear-attention` pin before the run).

## 1. Candidates

| Hub id | Size | Tokenizer | Train data / tokens | Code? | Licence | Load |
|---|---|---|---|---|---|---|
| fla-hub/delta_net-1.3B-100B | ~1.3B (config: 24 layers, d=2048, bf16) | vocab 32000 (Llama-2-class; config) | card empty; "100B" in name only | unverified (SlimPajama has some GitHub) | none stated | AutoModelForCausalLM, needs `import fla`; config saved with transformers 4.48.2 |
| fla-hub/gla-1.3B-100B | 1.3B | 32000 (unverified) | SlimPajama-627B tag; 100B per page | some (SlimPajama) | MIT | same, needs fla |
| fla-hub/transformer-1.3B-100B | 1.3B | same | SlimPajama-627B tag; 100B in name | some | MIT | plain transformers/fla. **Matched Transformer control: same data, tokens, tokenizer** |
| linear-moe-hub/Gated-Deltanet-340M | 0.4B | not stated | 15B tokens of SlimPajama | some | Apache-2.0 | model_type gated_deltanet; page gives no load code, needs fla. Third-party, not official |
| fla-hub/rwkv7-0.1B-g1a | 191M | RWKV World, 65,536 vocab | not stated ("more data" than prior) | not stated | Apache-2.0 | trust_remote_code=True, fla (git main), transformers>=4.48; must prepend token id 0 |
| RWKV/RWKV7-Goose-World3-1.5B-HF | 1.52B | RWKV World 65,536 | World v3, 3.119T tokens | not stated on page | Apache-2.0 | fla==0.3.0, transformers>=4.48, trust_remote_code |
| fla-hub/rwkv7-1.5B-g1a (also G1i/G1j 1.5B) | ~1.5B | RWKV World | not stated | not stated | not read | as above |
| AntonV/mamba2-370m-hf (also 130m, 780m, 1.3b, 2.7b) | 370M | page silent (GPT-NeoX in Mamba paper, unverified) | page silent; Pile ~300B per Mamba-2 paper (unverified) | Pile has GitHub | MIT | native transformers Mamba2; optional mamba-ssm/causal-conv1d, else slow path |
| state-spaces/mamba-370m-hf (also 2.8b-hf) | 370M | GPT-NeoX | page silent; Pile 300B (paper, unverified) | Pile GitHub | not stated | native transformers MambaForCausalLM. Original state-spaces/mamba2-* are NOT -hf format; the -hf Mamba2 ports are under AntonV |
| Zyphra/Zamba2-1.2B | ~1.2B hybrid Mamba2 + shared attention | not read | ~3T tokens text+code, +100B anneal | yes (stated) | Apache-2.0 | transformers (page says install from source) |
| tiiuae/Falcon-H1-0.5B-Base | 0.5B hybrid attn+Mamba2 | not read | not stated on page | has code evals | Falcon-LLM licence | latest transformers |
| EleutherAI/pythia-410m | 410M | GPT-NeoX | Pile, 299.9B tokens | Pile GitHub only | Apache-2.0 | standard transformers |
| HuggingFaceTB/SmolLM2-360M | 360M | not read | 4T tokens | The Stack among sources | Apache-2.0 | standard transformers |
| Qwen/Qwen2.5-0.5B | 0.49B | not read | not stated on page | code "improved" | Apache-2.0 | transformers>=4.37 |

Not found: official Gated DeltaNet weights (NVlabs/GatedDeltaNet says code only, per search). fla-hub's visible listing has only delta_net-1.3B/2.7B-100B plus RWKV7, but the org has 60 models (only 16 shown); gla and transformer 1.3B-100B exist (opened directly). I did not confirm a fla-hub gated_deltanet 1.3B.

## 2. Recommendation

1. **fla-hub/delta_net-1.3B-100B vs fla-hub/transformer-1.3B-100B (plus gla-1.3B-100B as a second recurrent).** Same SlimPajama data, same 100B budget, same size and tokenizer class, MIT. This is the only clean "recurrent vs Transformer of similar size" pair, so a gap narrowing is attributable to architecture, not data. DeltaNet is also the family closest to the project's GDN backbone. Weakness: little code in SlimPajama, so code-loss claims will be weak; report exact-recall and prose loss.
2. **Mamba-370m-hf or AntonV/mamba2-370m-hf vs Pythia-410m.** Both Pile ~300B, GPT-NeoX tokenizer, near-equal size, Pythia is the standard Transformer control; pure SSM with famously weak copying. Natively in transformers, so least dependency risk (no fla). Pile includes GitHub, so a code split is meaningful.
3. **RWKV7 (fla-hub/rwkv7-0.1B-g1a, or the 1.5B World3) as a stretch target,** with SmolLM2-360M / Qwen2.5-0.5B as references. These are heavily trained (trillions of tokens), code-capable, and test whether a strong backbone still benefits. Caveats: different tokenizer (ICE-V is token-level, so match on that tokenizer), no matched Transformer, fla pin and token-0 prefix quirk.

Skip for the main claim: Zamba2/Falcon-H1 (already hybrid with attention, so no "recurrent" story; Falcon licence).

## 3. Prior work and the correct novelty claim

Verified by opening the arXiv abstract:
- kNN-LM, 1911.00172 (Khandelwal et al., "Generalization through Memorization"): interpolates a pretrained LM with kNN over a datastore, "no additional training"; fixed lambda tuned on dev. Frozen LM plus non-learned mixer.
- Neural/continuous cache, 1612.04426 (Grave, Joulin, Usunier): cache of past hidden states, dot-product lookup, adds to existing LMs; Grave also did unbounded cache. Closest in spirit: in-context, no datastore. Keyed by hidden state, soft similarity, not exact n-gram match.
- Pointer sentinel, 1609.07843 (Merity et al.; not opened this session) and pointer-generator 1704.04368 (not opened): learned gate/sentinel; trained jointly with the LM. ICE-V reuses the sentinel mixture form.
- Adaptive Semiparametric LMs, 2102.02557 (Yogatama et al.): learned gate combining cache + kNN + LM; trained with the LM, not a bolt-on.
- Efficient kNN-LM, 2109.04212 (He et al.): efficiency only. Adaptive retrieval was reported to me from memory (Adaptive kNN-LM, with a learned lambda net, is a separate paper, unverified here).
- RetoMaton, 2201.12431: automaton over datastore pointers, built unsupervised on a pretrained LM (page did not state frozen explicitly, but the paper's setup is kNN-LM-style).
- Copy Is All You Need, 2307.06962 (Lan et al.): copies text segments from a collection via contextual embeddings; a new generator trained for it, not a bolt-on to arbitrary frozen LMs.
- Prompt Lookup Decoding (github.com/apoorvumang/prompt-lookup-decoding): exact n-gram match against the prompt (max n=3), proposing tokens for speculative decoding. Same lookup primitive as ICE, but for speed, output-preserving, no mixing.
- Bakhtin et al. 2018, 1804.07705: small gate mixing a pretrained neural LM with n-gram LMs, base models not retrained. Closest older precedent for "tiny trained mixer over frozen LM".
- Searches found no paper bolting an exact in-context copy/cache onto a frozen Mamba/RWKV/DeltaNet (negative search only, not proof).

**Safe novelty wording:** Frozen LM + learned or fixed mixer with a retrieval/cache distribution is NOT new (kNN-LM, Bakhtin 2018, neural cache). In-context exact n-gram lookup via pointer mixtures is not new either (pointer sentinel, prompt lookup). What ICE-V can claim: (a) a ~2K-parameter scorer that uses the frozen backbone's own log-probs of exact-match candidates as the key feature; (b) applied to pretrained recurrent/linear-attention LMs, targeting their known recall deficit, quantifying how much of the recurrent-vs-Transformer gap it closes at O(~50-70 B/token) memory; (c) the verifier result (harm when lookup is wrong). Phrase as "to our knowledge, not previously evaluated on frozen pretrained recurrent LMs", and avoid "first copy module on a frozen LM".
