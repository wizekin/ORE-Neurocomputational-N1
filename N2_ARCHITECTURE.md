# ORE N2 — 17-operation biological visual-processing approximation

N2 consumes the spatial feature maps preserved by ORE N1, specifically
`N1Result.stage_outputs["normalized_energy"]` with layout
`[orientation, height, width]`. It never attempts to reconstruct spatial
information from N1's flattened feature vector. N1 remains unchanged.

## Ordered algorithm

1. **N2-01 Retinotopic coordinates:** retain pixel coordinates, radial distance,
   and eccentricity (u=k\log(1+e/e_0)).
2. **N2-02 Multiscale receptive fields:** Gaussian scale-space
   (R_s=G_s*F_1).
3. **N2-03 Contour integration:** connect nearby responses using proximity and
   orientation agreement; produce integrated contour strength and tangent.
4. **N2-04 Boundary cues:** combine contour evidence and spatial gradient cues.
5. **N2-05 Contour completion:** bounded iterative support fills weak contour gaps
   while preserving observed evidence.
6. **N2-06 Border ownership:** soft two-sided ownership probabilities using a
   temperature-controlled logistic assignment.
7. **N2-07 Figure-ground:** estimate figure and ground probabilities with local
   smoothing and boundary competition; this is an approximation to a pairwise
   energy model, not exact graph-cut inference.
8. **N2-08 Junctions and curvature:** orientation-population entropy estimates
   junction evidence; (κ=dθ/ds) is approximated by tangent gradients.
9. **N2-09 Shape fragments:** threshold completed contours and label connected
   fragments with area, centroid, bounding box, and junction statistics.
10. **N2-10 Structural shape graph:** connect spatially close fragments; report
    component and cycle counts (β_0,β_1) and
    (χ=β_0-β_1).
11. **N2-11 Crowding:** estimate eccentricity-dependent critical spacing
    (d^*=bE+w) and a local crowding field.
12. **N2-12 Position/scale tolerance:** soft log-sum-exp aggregation across
    receptive-field scales.
13. **N2-13 Ventral population:** preserve form/shape-oriented feature maps and
    a compact population summary; no grapheme labels are assigned.
14. **N2-14 Dorsal population:** expose spatial-energy features. Motion energy
    is explicitly unavailable and zero for single-frame input; temporal motion
    requires a future sequence interface.
15. **N2-15 Recurrent refinement:** bounded local recurrence
    (A^{t+1}=\phi(W_{ff}F+W_{lat}A^t+W_{fb}H^t-b)), with relative-change
    convergence diagnostic and iteration cap. No lexical or semantic priors.
16. **N2-16 Representation integration:** integrate channel summaries while
    preserving each original channel independently.
17. **N2-17 Structured visual output:** emit structured visual statistics,
    graph topology, diagnostics, and provenance without deciding grapheme
    identity or word meaning.

## Install and run

```powershell
python -m pip install -e ".[dev]"
python -m pytest
```

```python
from ore_n1 import N1Processor
from ore_n2 import N2Processor

n1 = N1Processor().process(gray_image)
n2 = N2Processor().process(n1)
print(n2.structured_output)
```

## Scientific scope

These stages are explicit, testable computational hypotheses inspired by
biological visual processing. They do not reproduce the complete human brain,
and default parameters are not fitted physiological constants. Synthetic
unit tests establish software behavior only, not biological validity.
