# ORE N3 — Recurrent Orthographic Prediction and Error

**Implementation:** ore_n3/
**Algorithm:** 29 explicit operations
**Input boundary:** one N2 visual-structure result per already-segmented patch
**Scientific status:** software prototype / algorithmic approximation; independent validation pending

## 1. Purpose and boundaries

N3 lets candidate grapheme identities exchange information across positions in an
ordered visual sequence. It learns smoothed unigram and directed bigram statistics,
forms left-to-right and right-to-left contextual messages, combines those messages
with bottom-up visual evidence, and retains uncertainty and prediction-error traces.

The current repository N2 phase produces visual-structure maps: contours, junctions,
figure-ground evidence, shape fragments, and ventral/dorsal summaries. It does not
emit grapheme probability distributions. N3 therefore adds a separate, explicitly
supervised visual-population readout: it learns class prototypes from labelled N2
patch results and converts similarity scores into candidate distributions. This
readout is part of the N3 bridge, not a claim that N2 already performs grapheme
classification.

N3 accepts already-segmented patches in logical reading order. It does not
perform full-word segmentation, full-page OCR, lexical retrieval, semantic
correction, or sentence-context reasoning.

## 2. The 29 algorithmic operations

### Learning path — operations 1–15

1. **Validate labelled references.** Require non-empty grapheme labels and N2 patch
   outputs or explicit finite feature vectors. Reserve UNK for uncertainty.
2. **Extract N2 descriptors.** Convert each N2 result into a fixed descriptor
   containing ventral-map channel statistics, contour/boundary/junction/curvature/
   crowding/figure-ground/invariance/dorsal statistics, fragment summaries, and
   shape-graph topology.
3. **Validate descriptor dimensions.** Require every reference row to have the
   same finite feature dimension.
4. **Establish the grapheme vocabulary.** Build a stable sorted class set from
   explicit labels; retain UNK as a separate inference outcome.
5. **Estimate feature standardization.** Compute per-feature mean and standard
   deviation from the reference set; replace near-zero scales with one.
6. **Normalize reference descriptors.** Standardize each vector and L2-normalize it.
7. **Estimate grapheme prototypes.** Average normalized vectors within each
   grapheme class.
8. **Normalize population prototypes.** L2-normalize each class prototype.
9. **Count unigram exposures.** Count labels in reference patches and supplied
   labelled sequences.
10. **Count forward bigrams.** Count each observed ordered pair (c_t,c_(t+1)).
11. **Count reverse bigrams.** Preserve the same ordered counts for reverse
    conditional inference; do not reverse the grapheme order in the data.
12. **Smooth the unigram prior.** Apply additive Dirichlet/Laplace smoothing:
    P(c)=(N(c)+alpha)/(N+alpha|C|).
13. **Normalize the forward matrix.** Compute P(next | current) row-wise.
14. **Normalize the reverse matrix.** Compute P(previous | next) column-wise
    from the ordered bigram counts.
15. **Freeze and fingerprint the model.** Retain the class vocabulary,
    standardization statistics, prototypes, transition matrices, and SHA-256
    identifiers. Recognition does not mutate these learned values.

### Recognition and recurrent inference — operations 16–29

16. **Validate the sequence.** Require a non-empty ordered list of unique patch IDs
    and finite reliability values in [0,1].
17. **Extract query descriptors.** Produce one N2 descriptor per supplied patch.
18. **Validate query dimensions.** Reject missing channels, non-finite values, or
    dimensions incompatible with the trained model.
19. **Standardize query descriptors.** Apply the frozen training mean/scale and
    L2-normalize.
20. **Compute population activations.** Compute cosine similarity between each
    query descriptor and every grapheme prototype.
21. **Construct visual beliefs.** Apply a temperature-controlled softmax over
    known graphemes, then reserve residual probability for UNK using a
    configurable similarity-confidence function.
22. **Measure visual uncertainty.** Compute categorical entropy for each position.
23. **Compute sensory precision.** Use segmentation confidence and attention:
    Pi_v = b_v (0.5 + 0.5 s)(1 + g_a a).
24. **Compute prior precision.** Increase contextual influence with visual entropy
    and segmentation uncertainty:
    Pi_p = g_p (0.5 + H/log|C|)(2-s).
25. **Initialize beliefs.** Start each position from its own bottom-up visual
    distribution; do not force a single candidate.
26. **Generate bidirectional predictions.** For a left neighbor belief b_(i-1),
    compute m_L=b_(i-1)P(next|current). For a right neighbor belief b_(i+1),
    compute m_R=P(previous|next)b_(i+1). Include the smoothed unigram prior
    and normalize the available messages.
27. **Combine visual evidence and contextual prior.** Form a precision-weighted
    log-space posterior proposal:
    q*=softmax(Pi_v log q_visual + Pi_p log q_context).
28. **Iterate damping, error, and residual signals.** Update
    b_new=(1-rho)b_old+rho q*; calculate classwise divergence
    e(c)=b(c)[log b(c)-log q_context(c)]; estimate the feature residual
    r=x-q_context(prototypes); record a diagnostic energy proxy and mean
    belief change. Stop when the configured tolerance is met or the iteration
    limit is reached.
29. **Return structured hypotheses and trace.** Preserve initial visual
    distributions, final contextual priors, posterior beliefs, per-class error
    terms, feature residual vectors, ranked candidates, convergence diagnostics,
    model fingerprints, and explicit scope metadata.

## 3. Mathematical summary

Let C be the known grapheme classes and UNK the uncertainty class. For patch i,
N2 supplies descriptor x_i. N3 standardizes and normalizes the descriptor and
compares it with class prototypes w_c:

    z_i = normalize((x_i - mu) / sigma)
    a_i(c) = z_i dot w_c
    q_i(c) = softmax(a_i(c) / T)

A configurable confidence function reserves residual probability for UNK;
therefore visual beliefs remain normalized without forcing every patch into a
known grapheme class.

For ordered sequence counts N(a,b) and smoothing alpha:

    P(b|a) = (N(a,b)+alpha) / sum_j (N(a,j)+alpha)
    P(a|b) = (N(a,b)+alpha) / sum_i (N(i,b)+alpha)

At position i, the contextual prior is a normalized weighted combination of
the unigram distribution, the left-neighbor message, and the right-neighbor
message. Missing boundary messages are omitted and remaining weights are
renormalized.

The sensory and prior precision approximations are:

    Pi_v,i = b_v (0.5 + 0.5 s_i)(1 + g_a a_i)
    Pi_p,i = g_p (0.5 + H(q_i)/log(|C|))(2-s_i)

The posterior proposal and damped state update are:

    q*_i = softmax(Pi_v,i log q_i + Pi_p,i log prior_i)
    b_i^(t+1) = (1-rho)b_i^t + rho q*_i

Classwise divergence terms and prototype-feature residuals are retained as
diagnostics. The scalar energy is explicitly named an energy proxy, not a
measured biological free energy.

## 4. API

    from ore_n3 import N3Config, N3Patch, N3Sequence, LabeledN2Patch, N3Processor

    engine = N3Processor(N3Config())
    engine.fit(
        [
            LabeledN2Patch("a", n2_result=n2_a_1),
            LabeledN2Patch("a", n2_result=n2_a_2),
            LabeledN2Patch("b", n2_result=n2_b_1),
            LabeledN2Patch("b", n2_result=n2_b_2),
        ],
        grapheme_sequences=[("a", "b", "a"), ("b", "a", "b")],
    )
    result = engine.process(N3Sequence("sample-01", [
        N3Patch("p1", n2_result=query_n2_1, segmentation_confidence=0.9),
        N3Patch("p2", n2_result=query_n2_2, segmentation_confidence=0.8),
    ]))
    print(result.to_dict())

Each n2_result above represents one pre-segmented patch processed through N1
and N2. The caller or a separately validated segmentation component must provide
patch boundaries and logical reading order. For controlled tests or another
upstream model, features= may provide a finite one-dimensional vector directly;
all training and query vectors must share a dimension.

## 5. Interpretation and limitations

- Prototype similarity is a compact visual readout, not a measured population
  of biological neurons.
- Unigram/bigram transitions are simple statistical priors; they do not capture
  all orthographic dependencies or script-specific morphology.
- Bidirectional message passing is an algorithmic approximation.
- Precision formulas are configurable engineering assumptions, not measured
  neural gain laws.
- The feature residual is in engineered descriptor space, not pixel space.
- The energy proxy is a diagnostic, not a validated neural free-energy quantity.
- No semantic, lexical-dictionary, or sentence-context correction is performed.
- Synthetic tests establish code behavior, not human reading accuracy or neural
  validity.

N3 software tests must be followed by held-out writer/font/script evaluation,
prior-versus-visual conflict tests, ablations, calibration analysis, and
independent review before making biological claims.
