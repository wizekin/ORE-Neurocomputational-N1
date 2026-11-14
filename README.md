# ORE Neurocomputational N1

A modular Python implementation of the 12-operation N1 early-visual-processing pipeline for the Orthographic Representation Engine (ORE). It exposes intermediate maps, configurable biological approximations, optional temporal state, and per-stage numerical diagnostics.

> Scientific scope: this is a biologically motivated computational approximation. It is not a cellular simulation, a fitted model of a particular species, or evidence that ORE reproduces the complete human visual system. Default parameters are engineering starting points and require calibration against empirical data before physiological claims are made.

## The 12 operations

| # | Operation | Main computation | Main output |
|---:|---|---|---|
| 1 | Photoreceptor adaptation | Gaussian local-illumination estimate and logarithmic response compression | photoreceptor response, adaptation field |
| 2 | Retinal ON/OFF processing | Local reference subtraction and positive/negative rectification | retinal ON and OFF |
| 3 | Center-surround receptive fields | Center-minus-surround Difference of Gaussians | ganglion ON and OFF |
| 4 | LGN contrast gain control | Local RMS estimate and divisive gain | LGN ON/OFF and gain map |
| 5 | V1 orientation-selective populations | Even/odd Gabor filter bank over configured orientations | V1 linear responses |
| 6 | Simple-cell-like responses | Phase-specific ON/OFF half-wave rectification | simple-cell response bank |
| 7 | Complex-cell-like quadrature energy | Square-root sum of even/odd phase energies | complex-cell energy |
| 8 | Population divisive normalization | Local pooled activity across orientation channels | normalized energy |
| 9 | Spatial pooling | Adaptive non-overlapping mean pooling | pooled orientation maps |
| 10 | Feature-vector construction | Orientation-major flattening of pooled maps | feature vector |
| 11 | Optional temporal neural dynamics | Leaky state integration with an exponential time constant | temporal activity |
| 12 | Diagnostics and numerical validation | Shapes, ranges, moments, timings, and finite-value checks | diagnostics report |

Operation 11 is applied between the normalized V1 population output (operation 8) and spatial readout (operation 9). When disabled, it is an identity operation and processing is frame-independent.

## Installation

Python 3.10+ is required.

    python -m venv .venv
    python -m pip install -e ".[dev]"

Activate the environment first. On Windows PowerShell use .venv\Scripts\Activate.ps1; on macOS/Linux use source .venv/bin/activate.

## Quick start

    import numpy as np
    from ore_n1 import N1Config, N1Processor

    image = np.random.default_rng(7).random((128, 128), dtype=np.float32)
    result = N1Processor(N1Config()).process(image)

    print(result.feature_vector.shape)
    print(result.metadata)
    print(result.diagnostics["valid"])
    retinal_map = result.stage_outputs["retinal_on"]

Inputs are two-dimensional grayscale images. Floating inputs must be finite and in [0, 1]. Integer grayscale inputs are scaled by their dtype maximum (uint8 is divided by 255). RGB conversion is deliberately outside the first N1 input contract so color handling cannot silently alter the grayscale signal.

Run the included synthetic experiment:

    python -m ore_n1
    python -m ore_n1 --output n1_demo.npz
    python -m pytest

## Mathematical sketch

Let I(x,y) be the normalized input image and G_sigma * I denote Gaussian smoothing.

1. Photoreceptor approximation: A = G_sigma_a * I; P = log(1 + I / (A + epsilon_a)).
2. ON/OFF pathways: C = P - G_sigma_b * P; ON = max(C, 0); OFF = max(-C, 0).
3. Center-surround: for either pathway X, R = max(G_sigma_c * X - G_sigma_s * X, 0), with sigma_s greater than sigma_c.
4. LGN gain: E = ON² + OFF²; g = 1 / (sqrt(G_sigma_l * E + epsilon) + beta)^gamma; multiply each pathway by g.
5. Orientation population: convolve the combined afferent drive with even and odd Gabor kernels at each orientation theta.
6. Simple-cell approximation: preserve positive and negative half-wave responses for each orientation and phase.
7. Complex-cell approximation: Q_theta = sqrt(E_even,theta² + E_odd,theta² + epsilon). Quadrature energy reduces sensitivity to carrier phase.
8. Divisive normalization: N_theta = Q_theta / (epsilon_n^p + G_sigma_n * sum_phi(Q_phi^p))^(1/p).
9. Spatial pooling: average each normalized orientation map over each cell of a fixed grid-height by grid-width partition.
10. Feature vector: concatenate pooled values in orientation-major, row-major order.
11. Temporal dynamics: state <- state + (1 - exp(-dt/tau)) * (input - state).
12. Validation: ensure each stage is finite and non-empty; record shape, dtype, minimum, maximum, mean, standard deviation, and elapsed time.

The equations are explicit so each approximation can be replaced, measured, or tested independently. Gaussian boundary handling uses reflection. Gabor kernels are zero-mean and unit-L2 normalized.

## Output contract

N1Result contains:

- feature_vector: one-dimensional float32 vector with length orientation_count × grid_height × grid_width.
- pooled_features: array shaped [orientation, grid_y, grid_x].
- stage_outputs: dictionary retaining intermediate arrays for analysis and debugging.
- metadata: input shape, orientation angles, pooling grid, feature dimension, temporal setting, and model-scope declaration.
- diagnostics: JSON-serializable dictionary with validity, stage count, warnings, and numerical records.

The numerical stage functions are mostly pure functions in ore_n1/stages.py. N1Processor owns orchestration and optional temporal state. Call reset_temporal_state() between unrelated video sequences, or pass reset_temporal=True for the first frame of a sequence.

## Validation and limitations

The software tests check input contracts, constant-field behavior, feature dimensions, finite outputs, phase-tolerant quadrature energy at a population scale, temporal-state behavior, and configuration validation. Synthetic tests are software sanity checks, not physiological validation.

A research-grade validation phase should compare responses to controlled gratings, edges, contrast series, natural-image corpora, and published measurements. Parameter fitting should use training data, with final performance assessed on held-out data and independent replication.

## Repository layout

- ore_n1/config.py — validated configuration
- ore_n1/filters.py — even/odd Gabor filter bank
- ore_n1/stages.py — numerical functions for operations 1–10
- ore_n1/temporal.py — operation 11 state integrator
- ore_n1/diagnostics.py — operation 12 recording and validation
- ore_n1/pipeline.py — end-to-end orchestration and result contract
- ore_n1/synthetic.py — reproducible visual stimuli
- ore_n1/__main__.py — runnable demonstration
- tests/ — automated software tests
- examples/ — example integration script

## License

MIT. See LICENSE.


## N2 biological visual processing (17 operations)

The repository now includes an independent `ore_n2` package implementing the
17-step N2 visual-organization pipeline. N2 consumes the preserved
`normalized_energy` spatial maps from N1; it does not reconstruct maps from the
flattened N1 feature vector and does not modify N1. See
[`N2_ARCHITECTURE.md`](N2_ARCHITECTURE.md) for the ordered algorithm, equations,
interfaces, and scientific limitations.

Quick start:

```python
from ore_n1 import N1Processor
from ore_n2 import N2Processor

n1_result = N1Processor().process(grayscale_image)
n2_result = N2Processor().process(n1_result)
print(n2_result.structured_output)
print(n2_result.diagnostics["valid"])
```

Install with `python -m pip install -e ".[dev]"` and run the complete suite
with `python -m pytest`. N2 outputs remain visual-structural representations;
grapheme identity and lexical semantics are deliberately outside this phase.


## N3 recurrent orthographic prediction and error (29 operations)

The repository now includes an independent `ore_n3` package. It learns labelled
N2 patch descriptors and smoothed grapheme unigram/bigram transitions, then
combines visual candidate beliefs with left- and right-neighbor predictions
using precision-weighted, damped inference. It preserves alternatives,
classwise prediction-error terms, feature-space residuals, and convergence
diagnostics. N3 expects one already-segmented N2 result per patch; it does not
perform segmentation or semantic correction.

See [docs/N3_ARCHITECTURE.md](docs/N3_ARCHITECTURE.md) for the exact 29
operations, equations, contracts, API, and scientific limitations.

Quick start:

```python
from ore_n3 import N3Config, LabeledN2Patch, N3Patch, N3Sequence, N3Processor

n3 = N3Processor(N3Config()).fit(
    labelled_reference_patches,
    grapheme_sequences=[("a", "b", "a"), ("b", "a", "b")],
)
result = n3.process(N3Sequence("sample", [
    N3Patch("p1", n2_result=n2_patch_1, segmentation_confidence=0.9),
    N3Patch("p2", n2_result=n2_patch_2, segmentation_confidence=0.8),
]))
print(result.candidates)
```

Install and run all tests with `python -m pip install -e ".[dev]"` followed by
`python -m pytest`. N3 is an algorithmic research approximation, not a
validated neural circuit model.
