# ORE N8 — 72-operation lexical-semantic access and composition

**Package:** `ore_n8/`  
**Parent evidence:** N6 hierarchical orthographic access + N7 predictive/contextual processing  
**Output:** context-weighted concept/sense candidates, preserved alternatives, explicit semantic graph, compositional evidence frames, diagnostics and a 72-operation trace  
**Status:** software prototype; independent linguistic, empirical and biological validation are pending

## Purpose and boundary

N8 adds a lexical-semantic layer after N6 has produced an orthographic token sequence and N7 has produced intent/domain context posteriors. It maps exact token spellings to externally supplied semantic entries, scores alternative senses using N7 context and explicit caller cues, and composes token concepts only through relation evidence explicitly supplied to N8.

N8 requires both the `N6Result` and the matching `N7Result`. N7 preserves contextual beliefs but does not carry the full lexical token list in its public result, so N6 is supplied alongside N7 to retain token identity, boundaries and orthographic provenance.

### Evidence discipline

- Semantic entries are supplied by the caller and require a verifier and source reference by default.
- Matching uses exact Unicode-normalized orthographic forms. Morphology, transliteration, spelling correction and subword composition are not silently introduced.
- N7 intent/domain probabilities act as contextual compatibility signals; they are not treated as semantic truth.
- `semantic_context` may contain `global_sense_weights`, `token_sense_weights`, `feature_hints` and `token_feature_hints`. These are explicit caller observations, not values inferred by N8.
- Relation edges are built only from `N8RelationEvidence`. N8 does not infer subject/object, dependency, temporal or causal relations from token order.
- Unknown, ambiguous and low-confidence tokens retain their alternatives and are not forced into a known concept.
- N6 and N7 inputs are read-only. N8 does not train its lexicon or mutate persistent context memory.

## The 72 operations

### A. Request, sequence and evidence integrity — operations 1–9

1. Validate the `N8Input` request type.
2. Validate the N7 result interface and schema.
3. Validate the N6 result interface and schema.
4. Require matching non-empty N6/N7 sequence identities.
5. Snapshot the upstream fields consumed by N8.
6. Validate the finite, non-negative, normalized N7 joint posterior and its ranked hypotheses.
7. Preserve N7 visual, orthographic, N6-contextual, prediction and error channels separately.
8. Validate non-empty N6 token records and required identifiers, spellings, spans and confidence.
9. Fingerprint consumed N6/N7 evidence so post-inference immutability can be checked.

### B. Token reconstruction and reliability — operations 10–18

10. Extract ordered token identifiers.
11. Normalize token spellings with Unicode NFC and optional case folding.
12. Validate positive, ordered, non-overlapping token spans.
13. Extract token confidence from N6.
14. Extract start/end boundary confidence.
15. Extract segmentation confidence and upstream lexical status.
16. Compute a bounded token reliability weight from N6 confidence, boundary confidence and segmentation confidence.
17. Mark upstream unknown/unlexicalized tokens without discarding them.
18. Construct aligned token feature records preserving orthography, span, language and reliability.

### C. Semantic resource validation — operations 19–27

19. Load the caller-supplied semantic lexicon snapshot.
20. Validate independent verifier and source-reference provenance.
21. Normalize semantic-entry orthographic forms with the same Unicode policy.
22. Validate unique entry IDs and non-empty concept/sense IDs.
23. Validate positive finite lexical frequency weights.
24. Validate bounded domain and intent compatibility maps.
25. Validate sparse semantic feature weights in `[0, 1]`.
26. Build a deterministic exact-form index.
27. Fingerprint the immutable lexicon snapshot.

### D. Context and candidate generation — operations 28–36

28. Read the explicit semantic context payload.
29. Validate global sense weights.
30. Validate token-specific sense weights and feature hints.
31. Marginalize the N7 joint posterior to an intent distribution.
32. Marginalize the N7 joint posterior to a domain distribution.
33. Generate exact-form, language-compatible sense candidates for each token.
34. Add an explicit unknown/unmapped fallback candidate to each token.
35. Apply a deterministic candidate limit while reserving room for the unknown fallback.
36. Measure exact-form lexical coverage.

### E. Evidence-weighted sense scoring — operations 37–45

37. Compute a frequency-derived candidate prior.
38. Compute candidate/domain compatibility.
39. Compute candidate/intent compatibility.
40. Apply explicit global or token-specific sense evidence.
41. Compare candidate association tags with N7's non-mutating association ranking.
42. Weight top-down evidence by the N6 token reliability estimate.
43. Combine the evidence terms into candidate log scores.
44. Normalize each token's sense candidates with a stable softmax.
45. Retain the complete ranked alternatives, including the unknown candidate.

### F. Uncertainty, ambiguity and abstention — operations 46–54

46. Compute categorical posterior entropy for each token.
47. Compute the top-two posterior margin.
48. Estimate known-concept posterior mass.
49. Measure conflict between N7 context and candidate compatibility.
50. Pull unreliable token posteriors toward the frequency/unknown prior.
51. Evaluate confidence and ambiguity criteria.
52. Apply the unknown and abstention gates.
53. Build resolved, ambiguous, low-confidence or unknown token state records.
54. Preserve unselected candidate senses and their provenance.

### G. Explicit semantic graph and composition — operations 55–63

55. Validate the caller-supplied relation evidence records.
56. Validate that relation endpoints reference existing N6 token indices.
57. Validate relation labels, confidence and relation provenance.
58. Bind relation endpoints to N8 token semantic states.
59. Weight relation confidence by both endpoint sense confidences.
60. Construct one graph node per token while preserving unresolved nodes.
61. Construct only the explicitly supported graph edges; infer no new edges.
62. Group connected supported evidence into compositional frames.
63. Measure frame completeness and confidence.

### H. Concept aggregation, diagnostics and return — operations 64–72

64. Aggregate concept activation across token candidate distributions.
65. Normalize and deterministically rank concept hypotheses.
66. Keep N6 visual/orthographic, N7 contextual and N8 semantic channels separate.
67. Compute sequence-level semantic confidence and mean normalized uncertainty.
68. Rank concepts and compositional frames deterministically.
69. Build the token-level semantic report with alternatives.
70. Build diagnostics, configuration and resource provenance.
71. Verify upstream immutability, output integrity and trace order.
72. Return the structured `N8Result`.

## Mathematical approximation

For token `i` and verified sense candidate `s`, define a log score:

\[
L_{i,s}=w_f\log F_{i,s}+r_i\left[
 w_d\log D_{i,s}+w_t\log T_{i,s}+w_h\log H_{i,s}
 +w_x\log(X_{i,s})+w_a\log(A_{i,s})\right]
\]

where:

- `F` is the normalized frequency prior;
- `D` and `T` are compatibility with the N7 domain and intent marginals;
- `H` is explicit caller-provided sense evidence;
- `X` is sparse semantic-feature similarity;
- `A` is an association score from explicit tags and N7's association ranking;
- `r_i` is N6-derived token reliability; and
- the `w` terms are configurable weights.

The posterior is:

\[
q_i(s)=\operatorname{softmax}(L_{i,s}/\tau)
\]

For unreliable N6 tokens, N8 mixes the posterior with a broad lexical/unknown prior `p_i`:

\[
\tilde q_i(s)=r_iq_i(s)+(1-r_i)p_i(s)
\]

An explicitly supplied relation edge `(i, j, R)` is retained with confidence:

\[
c_{ij}=c_R\,\tilde q_i(s_i^*)\,\tilde q_j(s_j^*)
\]

It is considered supported only when both endpoints have resolved concepts and the confidence exceeds the configured threshold. This is a transparent computational policy, not a claim that the brain uses the same equations.

## Python API

```python
from ore_n8 import N8Config, N8Input, N8Processor, N8SemanticEntry, N8RelationEvidence

semantic_lexicon = [
    N8SemanticEntry(
        entry_id="bank-finance",
        orthography="bank",
        concept_id="FINANCIAL_INSTITUTION",
        sense_id="bank.n.01",
        gloss="institution that provides financial services",
        verified_by="independent-reviewer",
        source_reference="lexicon://en/bank/finance",
        domain_weights={"language": 1.0},
    )
]

result = N8Processor(N8Config()).process(N8Input(
    n6_result=n6_result,
    n7_result=n7_result,
    semantic_lexicon=semantic_lexicon,
    relation_evidence=[N8RelationEvidence(
        source_token_index=0,
        target_token_index=1,
        relation="ASSOCIATED_WITH",
        confidence=0.9,
        verified_by="independent-parser-reviewer",
        source_reference="relations://release/record-001",
    )],
    semantic_context={"global_sense_weights": {"bank-finance": 0.9}},
))

assert result.diagnostics["operation_trace_complete"]
assert len(result.operation_trace) == 72
```

Install with `python -m pip install -e ".[dev]"` and run `python -m pytest tests/test_n8.py -q` followed by `python -m pytest`.

## Scientific and engineering limitations

- N8 is an algorithmic lexical-semantic research prototype, not a cellular, synaptic or cortical-circuit simulation.
- Exact-form mappings do not cover morphology, spelling correction, transliteration or productive word formation.
- Semantic correctness depends on the external lexicon and independent evaluation; provenance fields provide traceability, not proof.
- Intent/domain compatibility weights and semantic feature vectors are explicit modeling inputs and require held-out calibration.
- No dependency parse, proposition, causal relation or discourse relation is invented. Compositional frames contain only supplied relation evidence.
- Synthetic software tests verify contracts and numerical behavior, not human-like semantic understanding or biological correspondence.
- Evaluate with sense-annotated, writer/script-disjoint and domain-held-out corpora; report sense accuracy, coverage, calibration, abstention risk and graph-relation precision/recall.
