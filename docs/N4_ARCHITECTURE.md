# ORE N4 — 40-operation attention and precision-gating architecture

**Package:** ore_n4/  
**Input:** A validated N3 result with patch-level visual beliefs, contextual priors, posterior beliefs, classwise prediction-error terms and candidate metadata  
**Output:** An attention-gated posterior, attention allocations, class-selective gates, sensory/prior precision values, ranked alternatives and a 40-operation trace  
**Status:** Software prototype; independent empirical and biological validation are pending

## Boundary and purpose

N4 is an additive stage above N3. It does not replace or modify N1, N2 or N3. It treats N3's visual probability distribution and contextual prior as separate signals and applies explicit attention and precision controls to a new output state. The supplied N3 object and its arrays are read-only from N4's perspective.

The design separates four computational hypotheses:

1. Bottom-up salience: uncertainty, prediction error and disagreement between visual evidence and contextual expectation can influence which patch receives more processing.
2. Top-down relevance: caller-provided task relevance and class goals can bias the allocation toward relevant positions or candidate classes.
3. Competitive spatial allocation: attention is distributed across sequence positions, retains a configurable uniform floor and can be locally smoothed.
4. Precision gating: spatial attention modulates sensory precision, while uncertainty and task relevance modulate prior precision during a bounded belief update.

All task cues are optional. When absent, N4 derives salience from N3 uncertainty, prediction-error and visual/prior disagreement signals. It does not invent user goals or semantic context.

## The 40 operations

### A. Input and contract validation — operations 1–6

1. **Validate the N4 request.** Require an N4Input instance and a non-null upstream result.
2. **Validate the N3 interface.** Require sequence ID, class names, visual probabilities, contextual priors, posterior beliefs, classwise prediction errors and candidate metadata.
3. **Validate the class vocabulary.** Require at least one unique, non-empty class label.
4. **Validate tensors.** Require aligned finite matrices with shape [patch, class]; probabilities must be non-negative.
5. **Validate normalization.** Require each probability row to sum to one within tolerance. Correct only small numerical drift.
6. **Extract cues.** Read or validate segmentation confidence, seed attention, external salience and task relevance. Record which inputs were supplied and which defaults were used.

### B. Bottom-up salience — operations 7–15

7. **Measure visual entropy.** Compute H_i = -Σ_c q_i(c) log(q_i(c)+ε).
8. **Normalize uncertainty.** Divide entropy by log of the class count and bound it to [0,1].
9. **Measure classwise prediction error.** Read N3's per-class error decomposition.
10. **Reduce error to patch salience.** Average absolute classwise error and robustly scale the resulting patch vector.
11. **Measure visual/prior disagreement.** Compute symmetric KL divergence between visual probabilities and contextual priors.
12. **Normalize predictive surprise.** Robustly scale that disagreement to [0,1].
13. **Derive internal salience.** Combine normalized entropy, error and surprise with configured weights.
14. **Integrate external salience.** Optionally mix a caller-provided bottom-up cue with the internal estimate.
15. **Bound salience.** Clip and retain one salience value per patch.

### C. Top-down relevance — operations 16–19

16. **Extract task relevance.** Validate an optional per-patch relevance vector; otherwise use zero.
17. **Project class goals.** Project optional class-goal weights onto the current N3 posterior belief at each patch.
18. **Combine goal signals.** Blend positional task relevance and class-goal relevance.
19. **Integrate seed attention.** Add optional attention seeds from the caller or N3 patch metadata; top-down goal influence can be ablated.

### D. Competitive attention — operations 20–26

20. **Compute attention logits.** Combine bottom-up salience and top-down relevance.
21. **Allocate attention.** Apply a temperature-controlled softmax across sequence positions.
22. **Preserve a uniform floor.** Mix in a configurable uniform distribution so no position is eliminated by competition.
23. **Smooth spatial allocation.** Apply bounded immediate-neighbor smoothing without wrapping sequence edges.
24. **Renormalize attention.** Enforce non-negative allocation summing to one.
25. **Derive class attention.** Combine N3 class beliefs and normalized classwise error magnitude.
26. **Normalize class gates.** Preserve the class-attention distribution and create a bounded multiplicative gate centered approximately on one.

### E. Precision estimation — operations 27–32

27. **Estimate observation reliability.** Convert segmentation confidence into a bounded quality factor.
28. **Compute base sensory precision.** Scale observation quality by the configured sensory baseline.
29. **Apply spatial attention gain.** Modulate sensory precision by relative attention allocated to each patch.
30. **Compute prior precision.** Increase contextual precision with visual uncertainty and segmentation uncertainty.
31. **Apply top-down prior gain.** Modulate prior precision using task relevance.
32. **Bound precision.** Clip sensory and prior precision to configured positive ranges and verify finite values.

### F. Attended inference — operations 33–37

33. **Initialize attended beliefs.** Copy N3 posterior beliefs into a new N4 state.
34. **Construct a precision-weighted proposal.** Combine sensory log evidence and contextual log prior, with the class-selective gate applied to visual evidence.
35. **Normalize the proposal.** Apply stable softmax and check probability normalization.
36. **Iterate damped updates.** Update beliefs until mean absolute change reaches the configured tolerance or the iteration limit is reached.
37. **Compute attended prediction error.** Preserve classwise posterior-versus-prior divergence terms.

### G. Decision and output — operations 38–40

38. **Rank and abstain.** Return top-k alternatives; emit ABSTAIN below the confidence threshold and AMBIGUOUS when the top-two margin is too small.
39. **Build diagnostics and provenance.** Record normalization, finiteness, attention budget, precision, convergence, warnings, configuration and scientific limitations.
40. **Return the N4 result.** Package all retained channels, traces, hypotheses and metadata into N4Result.

## Core equations

Normalized uncertainty:

\[
U_i = \frac{-\sum_c q_i(c)\log(q_i(c)+\epsilon)}{\log(|C|)}
\]

Predictive surprise uses symmetric KL divergence:

\[
D_i = \frac12 KL(q_i\Vert p_i)+\frac12 KL(p_i\Vert q_i)
\]

Competitive position allocation:

\[
a_i = (1-\lambda_u)\operatorname{softmax}\left(\frac{s_i+t_i}{T_a}\right)
      + \lambda_u\frac{1}{N}
\]

Here s_i is bottom-up salience, t_i is top-down relevance, T_a is the attention temperature, and λ_u is the uniform-attention fraction. Optional neighbor smoothing is applied before final normalization.

Confidence-sensitive sensory precision:

\[
\pi_{v,i}=\operatorname{clip}
\left(\pi_{v,0}(0.5+0.5r_i)(1+g_a N a_i),
\pi_{v,\min},\pi_{v,\max}\right)
\]

Here r_i is segmentation confidence and N a_i converts the sequence allocation to a relative gain.

Uncertainty- and task-sensitive prior precision:

\[
\pi_{p,i}=\operatorname{clip}
\left(g_p(0.5+U_i)(2-r_i)(1+g_t t_i),
\pi_{p,\min},\pi_{p,\max}\right)
\]

Class-selective attended proposal and damped update:

\[
\hat b_i=\operatorname{softmax}
\left(\pi_{v,i}g_{i,c}\log q_i(c)+\pi_{p,i}\log p_i(c)\right)
\]

\[
b_i^{(k+1)}=(1-\rho)b_i^{(k)}+\rho\hat b_i
\]

The class gate g_{i,c} is derived from N3 class beliefs and prediction-error magnitude, then bounded. These formulas are explicit computational choices, not claims that neurons literally implement the same equations.

## Python API

N4 consumes an N3Result object produced by a separately trained/used N3Processor.

    from ore_n4 import N4Config, N4Input, N4Processor

    request = N4Input(
        n3_result,
        bottom_up_salience=[0.1, 0.8, 0.2],
        task_relevance=[0.0, 1.0, 0.0],
        goal_class_weights=[0.7, 0.2, 0.1],
    )
    result = N4Processor(N4Config()).process(request)
    print(result.candidates)
    print(result.diagnostics["valid"])

The optional input vectors are one value per patch, except goal_class_weights, which contains one non-negative value per N3 class. Optional seed_attention and segmentation_confidence vectors are also accepted. Every cue is checked for shape, finiteness and its allowed range.

## Testing

Run the focused suite with:

    python -m pytest tests/test_n4.py -q

Run the complete repository suite with:

    python -m pytest

The focused tests cover the exact 40-operation manifest, normalized outputs, finite precision, attention allocation, task-cue effects, spatial-attention ablation, invalid inputs, JSON serialization and immutability of the upstream N3 posterior.

## Scientific limitations

- Attention cues are algorithmic proxies, not measurements of neural firing, eye position or gaze.
- Precision values are configurable computational gains, not physiological constants.
- The implementation does not simulate cortical layers, neuromodulators, fixation, saccades or eye movements.
- N4 inherits N3's already-segmented patch-sequence interface; it does not perform character segmentation.
- The update acts on candidate-class beliefs; it is not a generative prediction of pixels or neural feature maps.
- Synthetic software tests establish code behavior only. Held-out writer/font/script evaluation and independent neural/behavioral evidence are still required before biological claims can be made.
