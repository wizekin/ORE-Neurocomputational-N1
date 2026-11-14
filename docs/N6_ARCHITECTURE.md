# ORE N6 — 60-operation hierarchical orthographic sequence and lexical-access architecture

**Package:** ore_n6/  
**Input:** an N4 attention/precision-gated grapheme-patch sequence, an immutable N5 model snapshot, optional word-boundary evidence, and an optional externally supplied orthographic lexicon  
**Output:** context-conditioned grapheme posteriors, a best sequence path, ranked grapheme alternatives, estimated boundary affinities, a k-best word-segmentation lattice, and a 60-operation trace  
**Status:** software prototype; independent empirical, behavioral, and biological validation are pending

## Boundary and purpose

N6 is an additive stage above N4 and N5. It does not modify N1–N5, learn labels, or infer word meaning. N4 contributes its attended posterior beliefs, attention allocation, and sensory precision. N5 contributes its versioned grapheme vocabulary, class gains, exposure statistics, and fast/slow forward and reverse transition memories. N6 uses those signals to infer a more coherent grapheme sequence and bind adjacent graphemes into candidate orthographic word forms.

The lexicon is supplied by the caller. Each entry must include an entry ID, a grapheme tuple, a positive frequency weight, a verifier, and a source reference. The provenance fields make the data traceable; they do not independently prove the correctness of an entry. N6 does not silently create lexical entries or update N5. An optional n2_features matrix lets N6 compare same-space patch descriptors with N5's fast/slow prototype banks; it must have one N2 feature vector per N4 patch. When this matrix is absent, N6 uses N4's posterior plus N5's class gains and transition memories, and explicitly records that prototype evidence was not used.

N6 distinguishes three outputs:

1. **Grapheme inference:** probability distributions over known grapheme labels at each input patch.
2. **Orthographic segmentation:** candidate spans that partition the grapheme sequence into word-form hypotheses.
3. **Lexical access:** matching supplied lexicon entries, with unknown/unlexicalized fallback spans retained when no entry is supported.

The architecture is orthographic. It does not claim semantic comprehension, morphological understanding, general language understanding, or reproduction of a biological lexical circuit. Lexicon expansion and supervised model learning remain separate operations and are not performed inside N6 recognition.

## The 60 operations

### A. Input and upstream integrity — operations 1–8

1. Validate the N6 request object.
2. Validate the N4 result interface and required fields.
3. Validate the N5 model interface and required fields.
4. Validate the sequence identity.
5. Validate a unique, non-empty grapheme vocabulary; reserve <UNK> for N5 transition bookkeeping rather than a supervised visual class.
6. Validate the N4 posterior tensor as [patch, grapheme], within the configured maximum sequence length.
7. Normalize small row-sum drift and preserve a separate copy of the original N4 posterior.
8. Validate patch-aligned N4 attention and sensory precision; inspect upstream diagnostics and apply strict or warning behavior.

### B. N5 model alignment and integrity — operations 9–16

9. Validate the N5 schema version and positive model version.
10. Require exact N4/N5 grapheme-vocabulary equality and order.
11. Validate the N5 feature mean and positive scale vectors.
12. Validate fast/slow prototype and eligibility matrices against the N5 vocabulary and feature dimension.
13. Validate the fast/slow forward and reverse transition-count matrices, including the reserved unknown state.
14. Validate all consumed N5 arrays for finite values, valid shapes, non-negative counts, and positive gains.
15. Recompute the canonical N5 SHA-256 fingerprint and reject a mismatch.
16. Build the grapheme-to-class index used by subsequent operations.

### C. Experience-dependent transition integration — operations 17–25

17. Combine fast and slow forward transition counts over known grapheme classes.
18. Combine fast and slow reverse transition counts.
19. Add configurable pseudocount smoothing and row-normalize the forward transition table.
20. Add smoothing and column-normalize the reverse transition table.
21. Combine fast and slow unigram exposure counts.
22. Normalize the unigram counts into a population class prior.
23. Convert N5 homeostatic class gains into bounded modulation factors.
24. Apply class-gain calibration to the N4 patch posterior; when raw N2 patch descriptors are supplied, compare them with the N5 fast/slow prototype banks and blend the resulting population evidence.
25. Convert the calibrated posterior into log-likelihood values.

### D. Bidirectional context and recurrent decoding — operations 26–42

26. Estimate left-context support from the preceding patch and forward transitions; use the unigram prior at the left edge.
27. Estimate right-context support from the following patch and reverse transitions; use the unigram prior at the right edge.
28. Combine the two contextual channels while retaining their separate intermediate arrays.
29. Add weighted contextual support to the visual log evidence.
30. Normalize the context-conditioned posterior for each patch.
31. Construct a symmetric pairwise transition score from forward and reverse probabilities.
32. Initialize sequence-level Viterbi state from the first patch's log evidence.
33. Propagate recurrent sequence hypotheses with pairwise transition scores.
34. Backtrace the maximum-scoring grapheme path without deleting patch positions.
35. Compute forward log messages.
36. Compute backward log messages.
37. Estimate per-position grapheme marginals from the forward/backward messages.
38. Fuse the marginal estimate with the context-conditioned posterior while preserving the original N4 posterior separately.
39. Normalize the final grapheme posterior.
40. Measure normalized positionwise entropy as an uncertainty proxy.
41. Measure mean transition coherence along the best path.
42. Return the top configured grapheme alternatives at each position.

For transition matrices, N5's fast and slow memories are summed separately for each direction. With additive smoothing alpha, a forward transition row is

\[
T_f(i,j)=\frac{C_f(i,j)+\alpha}{\sum_k(C_f(i,k)+\alpha)}.
\]

The reverse table is column-normalized so it estimates the previous class given the next class:

\[
T_r(i,j)=\frac{C_r(i,j)+\alpha}{\sum_k(C_r(k,j)+\alpha)}.
\]

The pairwise score is a normalized geometric combination of the two directional estimates. It is a modeling choice for bidirectional sequence consistency, not a claim about the exact computation of biological neurons.

The forward/backward marginal approximation uses log-domain dynamic programming:

\[
\alpha_t(j)=\log e_t(j)+\operatorname{LSE}_i[\alpha_{t-1}(i)+\log T(i,j)],
\]

\[
\beta_t(i)=\operatorname{LSE}_j[\log T(i,j)+\log e_{t+1}(j)+\beta_{t+1}(j)],
\]

where e_t(j) is the calibrated visual likelihood and LSE is log-sum-exp. The final N6 posterior combines normalized alpha_t+beta_t marginals with the independent context-conditioned proposal.

### E. Lexicon and boundary preparation — operations 43–49

43. Validate optional boundary evidence as a finite [patch + 1] vector in [0,1]; force the sequence endpoints to be boundaries.
44. Validate lexicon entry IDs, grapheme sequences, positive frequency weights, verifier provenance, and source references.
45. Index lexicon entries by their first grapheme.
46. Match lexicon entries against contiguous spans of the posterior sequence.
47. Compute each lexical span's visual emission and N5 transition evidence.
48. Use caller-provided boundary evidence or derive a heuristic boundary affinity from lexicon-supported spans.
49. Add a one-grapheme unknown/unlexicalized fallback edge at every patch so the segmentation graph always has a complete path.

### F. Hierarchical segmentation and lexical access — operations 50–57

50. Score lexical and fallback edges using grapheme likelihood, transition evidence, bounded frequency contribution, and boundary logits.
51. Run a deterministic k-best dynamic program over the segmentation directed acyclic graph.
52. Normalize the retained alternative path scores with a temperature-controlled softmax.
53. Select the highest-scoring complete segmentation as the primary output.
54. Bind each input grapheme position to exactly one word-level or fallback unit.
55. Compute token confidence from span evidence, attention, and relative sensory precision.
56. Apply explicit abstention and unknown-label rules; low-confidence known-word matches remain marked as abstentions.
57. Return ranked alternative segmentations and token-level orthographic metadata.

For a lexicon entry spanning positions s...e-1, N6 scores its orthographic evidence using the sum of log posterior probabilities for its graphemes, a weighted sum of N5 pairwise transition log scores, a bounded log-frequency contribution, and boundary logits at internal span edges. Unknown single-grapheme edges receive a configurable penalty rather than being removed. A k-best dynamic program retains alternative segmentations instead of forcing a single irreversible boundary decision.

### G. Integrity and output — operations 58–60

58. Verify that every input patch is covered exactly once and that the N5 fingerprint is unchanged.
59. Build diagnostics, operation trace, provenance metadata, configuration, warnings, and scientific-scope declarations.
60. Return the structured N6 result with all intermediate probabilities, candidates, boundaries, tokens, and alternatives.

## Python API

~~~python
from ore_n6 import N6Config, N6Input, N6LexiconEntry, N6Processor

lexicon = [
    N6LexiconEntry(
        entry_id="word-001",
        graphemes=("a", "b"),
        frequency=12.0,
        verified_by="independent-reviewer",
        source_reference="dataset://release/word-001",
    ),
]

result = N6Processor(N6Config()).process(
    N6Input(
        n4_result=n4_result,
        n5_model=n5_model_state,
        lexicon_entries=lexicon,
        boundary_probabilities=None,  # optional caller-supplied [patch + 1] vector
        n2_features=optional_n2_patch_feature_matrix,  # optional [patch, N5 feature dimension]
    )
)

print(result.decoded_graphemes)
print(result.tokens)
print(result.segmentation_alternatives)
print(result.diagnostics["valid"])
~~~

The N5 model can be obtained from the N5 learning or snapshot stage. N6 requires the N4 and N5 grapheme vocabularies to match exactly. If no lexicon is supplied, N6 still returns a complete sequence by emitting explicit UNLEXICALIZED_GRAPHEME fallback tokens.

## Tests

~~~powershell
python -m pytest tests/test_n6.py -q
python -m pytest
~~~

The focused suite checks the exact 60-operation manifest and trace, N4/N5 contracts, sequence normalization, vocabulary alignment, N5 fingerprint and immutability, lexicon provenance, boundary evidence, unknown fallback, k-best segmentation normalization, output serialization, and configuration validation. These are software invariants; they do not demonstrate OCR accuracy or biological validity.

## Scientific limitations and validation requirements

- Transition inference, Viterbi decoding, forward/backward marginals, lexical span scoring, and k-best segmentation are explicit algorithmic approximations.
- N5 frequencies and transitions may encode dataset bias; an independent held-out evaluation is required.
- Boundary affinities inferred from lexicon span support are heuristic and are not calibrated probabilities unless calibrated against boundary-annotated data.
- Lexicon provenance fields provide traceability, not proof of lexical truth.
- The optional lexicon is orthographic only. N6 does not infer meanings, syntax, or concepts.
- N6 does not simulate individual cortical neurons, synapses, or a complete biological word-recognition network.
- Tests establish software behavior only. Evaluate on writer-disjoint and font-disjoint samples, scripts with different orthographic properties, OOV words, ambiguous segmentation cases, and independently annotated boundary/lexicon corpora. Compare against held-out baselines and report calibration, CER/WER, segmentation F1, and confidence/abstention curves before making performance or biological claims.
