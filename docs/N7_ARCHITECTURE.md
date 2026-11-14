# ORE N7 — Predictive and Contextual Processing

**Branch:** `n7-complete-72-operations`  
**Status:** software implementation added; independent scientific validation pending.  
**Parent:** N6 hierarchical orthographic access (60 operations).

N7 adds a contextual predictive-inference layer above N6. It does not rewrite N6's observations or claim that contextual expectations are visual evidence. The output keeps visual, orthographic, and N6-contextual evidence separate from N7's contextual posterior.

## Purpose and boundary

N7 models a proposed computational motif: contextual latent causes generate predictions about lower-level cues; observed cues are compared with those predictions; precision weights the mismatch; and beliefs are revised to reduce expected prediction error plus a prior-divergence term.

Inputs:
- `n6_result`: an N6Result containing visual, contextual, and decoded posterior matrices, boundary probabilities, tokens, and sequence identity.
- `context_state`: optional intent/domain/style priors and weak contextual memory.
- `script_profile`: script identifier (metadata only; not a learned script model).
- `lexical_context`, `syntax_context`, `semantic_context`, `discourse_context`, `task_context`: optional upstream provider payloads.
- `provider_evidence`: explicit cue observations keyed by the configured cue vocabulary, each in [0,1].
- `processing_mode`: caller label recorded in output.

Output:
- preserved mean visual, orthographic, and N6 contextual evidence;
- top-down predictions, inverse-variance precision, and cue residuals;
- layer-specific and joint-hypothesis prediction errors;
- variational posterior over joint intent/domain hypotheses and style factors;
- ranked hypotheses, contextual association ranking, convergence, full 72-operation trace, diagnostics, and provenance.

## The 72 operations

Each operation is executed and receives one trace record. The operation trace is the authoritative record of what the implementation actually ran.

### A. Input and evidence separation (1–9)
1. Validate the N7 request and N6 result interface.
2. Preserve original N6 evidence as an immutable source.
3. Extract visual grapheme posterior evidence.
4. Extract decoded orthographic sequence posterior.
5. Extract boundary and token evidence.
6. Validate dimensions, probability bounds, normalization and finiteness.
7. Separate visual, orthographic and N6 contextual channels.
8. Attach N6 sequence/schema provenance.
9. Build an evidence-quality summary.

### B. Contextual state (10–18)
10. Load caller-supplied contextual state.
11. Validate intent prior.
12. Normalize intent prior.
13. Normalize domain prior.
14. Record script profile.
15. Encode task-goal context.
16. Encode discourse context.
17. Encode lexical/syntactic/semantic context availability.
18. Construct a normalized context-state vector.

### C. Hierarchical generative model (19–27)
19. Enumerate joint intent × domain hypotheses.
20. Construct separate style-factor hypotheses.
21. Combine contextual priors with a bounded smoothing prior.
22. Initialize latent belief state.
23. Map cue generators to joint hypotheses.
24. Construct the generative prediction matrix.
25. Assign cue channels to computational layers.
26. Compute the prior predictive distribution.
27. Validate generator dimensions and probability bounds.

### D. Top-down prediction (28–36)
28. Read explicit provider cue observations.
29. Derive visual evidence summary.
30. Derive orthographic evidence summary.
31. Derive lexical-context availability signal.
32. Derive discourse-context availability signal.
33. Derive task-context availability signal.
34. Propagate hierarchical prior predictions.
35. Blend prior predictions with observed cue baseline.
36. Bound and record final predicted cue means.

### E. Uncertainty and precision (37–45)
37. Load fixed cue observation variances.
38. Compute inverse-variance precision.
39. Clip precision to declared stability bounds.
40. Estimate categorical posterior entropy.
41. Estimate predictive residual variance.
42. Estimate prior confidence from normalized entropy.
43. Normalize channel precision for comparable weighting.
44. Construct precision-weighted residual evidence.
45. Bound residual magnitude for numerical stability.

### F. Prediction errors (46–54)
46. Compute visual-layer prediction error.
47. Compute orthographic-layer prediction error.
48. Compute lexical-context prediction error.
49. Compute discourse-context prediction error.
50. Compute task-context prediction error.
51. Aggregate precision-weighted layer errors.
52. Compute energy for each joint latent hypothesis.
53. Compute expected prediction-error energy.
54. Compute variational free energy including prior divergence.

### G. Belief revision (55–63)
55. Score joint hypotheses using prior and prediction energy.
56. Apply iterative exponentiated-gradient belief updates.
57. Reconcile direct posterior and iterative posterior estimates.
58. Update separate style-factor beliefs.
59. Measure convergence and preserve ambiguity when margins are small.
60. Create a contextual memory trace without mutating caller memory.
61. Compute decayed recent-context activation.
62. Score associations against stored context values.
63. Rank contextual associations deterministically.

### H. Contextual memory and output (64–72)
64. Keep visual, orthographic, N6 contextual and N7 contextual beliefs separate.
65. Construct a ranked intent/domain hypothesis report.
66. Compute final confidence and posterior uncertainty.
67. Validate diagnostics and probability state.
68. Validate and label the N7 result schema.
69. Verify that original N6 evidence remains represented in output.
70. Verify ranked-hypothesis cardinality and ordering.
71. Verify operation trace cardinality and ordering.
72. Return the structured N7 result.

## Mathematical approximation

For latent hypothesis (h=(i,d)), the generative model predicts cue (j):

[
\mu_j(h)=g_j(i,d)
]

The observed cue residual is:

[
\epsilon_j=y_j-\mu_j(h)
]

With a fixed diagonal observation-noise assumption:

[
\Pi_j=\frac{c}{\sigma_j^2}
]

The hypothesis energy and approximate variational objective are:

[
E(h)=\frac12\sum_j\Pi_j\epsilon_j^2
]

[
F(q)=\sum_hq(h)E(h)+\beta\,KL(q\|p)
]

An exponentiated-gradient approximation updates categorical beliefs:

[
q_{t+1}=softmax\left(\log q_t-\eta\nabla_qF\right)
]

These equations describe the implemented computational abstraction, not literal neuron equations. The current implementation uses diagonal cue precision, fixed hand-specified cue prototypes/variances, and a mean-field style-factor approximation.

## Python use

```python
from ore_n6 import N6Processor, N6Input
from ore_n7 import N7Config, N7Input, N7Processor

n6 = N6Processor().process(N6Input(
    n4_result=n4_result,
    n5_model=n5_model,
    lexicon_entries=verified_lexicon,
))
n7 = N7Processor(N7Config()).process(N7Input(
    n6_result=n6,
    context_state={
        "intent_prior": {"recognize": 0.8, "compare": 0.2},
        "domain_prior": {"orthography": 0.9, "language": 0.1},
        "memory": {"activation": 0.2},
    },
    script_profile="latin_basic",
    lexical_context={"provider": "local_lexicon"},
    syntax_context={},
    semantic_context={},
    discourse_context={"scope": "single_sequence"},
    task_context={"goal": "recognize"},
    provider_evidence={
        "visual_strength": 0.8,
        "orthographic_strength": 0.7,
        "context_strength": 0.4,
        "task_strength": 0.9,
    },
))
assert n7.diagnostics["operation_trace_complete"]
assert len(n7.operation_trace) == 72
```

N7 consumes a completed N6 result; the short example's `n4_result`, `n5_model`, and `verified_lexicon` are supplied by the caller.

## Scientific and engineering limitations

- N7 is an algorithmic-level predictive-processing approximation, not a neuron-, synapse-, or circuit-level simulation.
- Cue prototypes and observation variances are explicit engineering assumptions requiring held-out calibration.
- Cue covariance is diagonal; correlations are not modeled.
- Contextual memory is read as a weak prior and reported as a trace; N7 does not mutate a persistent memory store.
- No semantic truth validation, free-form language understanding, active inference, or autonomous synaptic plasticity is claimed.
- The tests establish software contracts and deterministic numerical behavior only. They do not demonstrate that N7 reproduces brain mechanisms or improves real-world recognition.

## Verification

Run:
```bash
python -m pytest tests/test_n7.py -q
python -m pytest
```
