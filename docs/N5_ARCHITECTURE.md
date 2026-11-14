# ORE N5 — 52-operation experience-dependent plasticity architecture

**Package:** ore_n5/  
**Input:** A frozen N3-derived model state, independently verified labelled N2 feature vectors, explicit user authorization, and an optional valid N4 result  
**Output:** A separately fingerprinted child model with fast/slow prototype banks, eligibility traces, exposure counters, replay memory, updated transition statistics and a 52-operation trace  
**Status:** Software prototype; independent empirical and biological validation are pending

## Boundary and learning safeguards

N5 is an additive learning stage following N4. It does not replace or modify N1–N4. N5 never updates the parent N3-derived model: it clones the parent and returns a separately versioned child. Recognition remains read-only; learning is a distinct call.

Two hard gates are mandatory:

1. independently_verified=True — supplied labels must have been checked independently of the model's current prediction.
2. user_authorized=True — the caller must explicitly authorize the learning operation.

Every example also requires a reviewer, a source reference and a verification timestamp. An optional SHA-256 digest binds the example to its float32 feature vector. These are data-governance checks; they do not independently prove that an annotation is correct.

N5 learns from the same raw N2 descriptor space used by N3. When an N4 result is supplied, its attention allocation, sensory precision and class-selective gate modulate the learning rate. N4 remains read-only. Without N4, neutral gates are used.

## The 52 operations

### A. Authorization and parent integrity — operations 1–7

1. Validate the N5 learning-request type.
2. Require explicit user authorization.
3. Require independent label verification.
4. Validate the parent model schema and version.
5. Recompute and verify the parent fingerprint.
6. Check parent arrays, dimensions, finite values, positive feature scales and non-zero prototype rows.
7. Validate a unique grapheme vocabulary; reserve the unknown class for uncertainty rather than supervised target labels.

### B. Experience validation — operations 8–15

8. Require a non-empty batch within the configured maximum.
9. Check unique experience IDs and unique sequence/patch pairs.
10. Require per-example provenance: verifier, source reference and timestamp.
11. Validate finite feature vectors in the parent's feature dimension.
12. Verify optional feature digests.
13. Validate target labels and optional logical sequence positions; derive ordered sequences only when positions are supplied.
14. Snapshot the parent's fingerprint and numerical arrays.
15. Clone a detached child model and increment its version.

### C. N4-guided learning gates — operations 16–20

16. Validate the optional N4 result schema, vocabulary and diagnostics.
17. Match N4 candidate patch IDs to the verified experiences.
18. Extract sensory precision and compare it with the batch mean.
19. Extract the class-selective gate for each verified target class.
20. Combine annotation confidence, relative spatial attention, sensory precision and target-class gain into a bounded per-example learning gate. This influence can be ablated.

### D. Feature preparation and surprise — operations 21–27

21. Standardize each new feature vector using the frozen parent's mean and scale.
22. L2-normalize the standardized activity vector.
23. Measure similarity against the parent population prototypes.
24. Estimate novelty from the maximum prototype similarity.
25. Estimate target-class prediction error.
26. Combine reliability, attention, novelty and target mismatch into a plasticity weight.
27. Bound the weight and reject a batch with no effective learning signal.

### E. Replay and eligibility — operations 28–32

28. Append verified normalized examples to a separate replay buffer.
29. Enforce a fixed replay-memory capacity.
30. Select prior replay examples deterministically by importance.
31. Apply recency decay and importance weighting to replay examples.
32. Measure target-selective population activity before the update.

### F. Fast and slow prototype plasticity — operations 33–40

33. Compute fast Oja-style target-prototype updates.
34. Update fast eligibility traces with exponential decay.
35. Bound each fast update by a maximum vector norm.
36. Apply the update and L2-normalize fast prototypes.
37. Compute slow Oja-style updates with a lower learning rate.
38. Update slow eligibility traces with a longer decay constant.
39. Apply and normalize slow prototypes using a stricter step bound.
40. Construct an effective population from a weighted fast/slow combination.

For normalized activity x, prototype w, bounded activity y and per-example plasticity weight q:

    Δw = η q y (x - y w)
    E(t) = ρ E(t-1) + Δw
    w(t) = L2Norm(w(t-1) + clipNorm(E(t), δmax))

This is a computational learning rule, not a claim that biological synapses use this exact formula.

### G. Homeostasis and anti-forgetting — operations 41–43

41. Measure post-update target-class activity.
42. Adjust class gains toward a configured target activity, then bound the gains.
43. Clip fast and slow prototype drift relative to the frozen parent.

Homeostatic gain control uses:

    g(c) <- clip(g(c) exp(ηh (h* - mean_activity(c))), gmin, gmax)

The prototype drift limit is measured relative to the parent snapshot, not relative to the immediately preceding example.

### H. Orthographic exposure and transition learning — operations 44–49

44. Increment per-class verified exposure counters.
45. Update fast unigram exposure memory.
46. Update fast forward bigram counts from verified sequences.
47. Update fast reverse transition counts.
48. Add a slower contribution to consolidated unigram and transition memory.
49. Mark high-importance replay records as consolidated for inspection and reproducibility.

Forward and reverse transition tables remain separate. They are normalized only when exported back into a new N3 processor; these learned statistics are not conflated with visual evidence.

### I. Integrity and return — operations 50–52

50. Measure prototype drift, gain drift and memory growth.
51. Validate finite values, dimensions, normalized prototypes, child fingerprint and parent immutability.
52. Fingerprint and return the child model, metrics, provenance metadata and all 52 operation records.

## Core equations

Feature standardization and normalization:

    z = (x - μ) / σ
    x̂ = z / max(||z||₂, ε)

Novelty and target mismatch:

    ν(i) = clip(1 - max(c) x̂(i) · w(c), 0, 1)
    e(i) = clip((1 - x̂(i) · w(y(i))) / 2, 0, 1)

Plasticity weight:

    q(i) = reliability(i) × attention(i) × precision(i) × class_gate(i)
           × [novelty_floor + novelty_gain × ν(i)
              + (1 - novelty_gain) × (0.25 + 0.75 × e(i))]

The fast and slow weight banks have distinct learning rates and eligibility decays. Updates are clipped, prototypes are normalized, and parent-relative drift is bounded.

## Python API

Initialize an independent N5 snapshot from a fitted N3 processor:

    from ore_n5 import N5Processor, N5Experience, N5LearningRequest

    n5 = N5Processor()
    parent = n5.initialize(fitted_n3_processor)

    examples = [
        N5Experience(
            experience_id="verified-001",
            patch_id="patch-001",
            sequence_id="sequence-001",
            grapheme="a",
            features=raw_n2_descriptor,
            verified_by="independent-reviewer",
            source_reference="dataset://release/record-001",
            verified_at="2026-11-14T05:00:00Z",
            confidence=0.98,
        )
    ]
    result = n5.learn(N5LearningRequest(
        parent_model=parent,
        experiences=examples,
        independently_verified=True,
        user_authorized=True,
        n4_result=optional_n4_result,
        verified_sequences=[("a", "b", "a")],
    ))

    child_n3 = result.child_model.to_n3_processor()
    print(result.diagnostics["valid"])
    print(result.child_model.fingerprint)

The child can be exported into a new N3 processor. The original N3 processor and N5 parent snapshot remain unchanged.

## Tests

    python -m pytest tests/test_n5.py -q
    python -m pytest

Tests cover the exact 52-operation manifest, authorization gates, verification provenance, feature digests, N4 cue integration, bounded prototype updates, parent immutability, replay capacity, transition normalization, serialization and export back to N3.

## Scientific limitations

- Oja-style updates, eligibility traces, replay and homeostatic gain control are algorithmic approximations.
- N5 does not simulate individual neurons, dendrites, synapses, neuromodulators, sleep physiology or a complete cortical learning circuit.
- N4 attention and precision are proxies; they are not measured neural signals.
- Labels must be independently verified outside this algorithm. Provenance fields alone do not establish truth.
- Tests establish software invariants only. They do not establish recognition accuracy, writer/font generalization or biological validity.
