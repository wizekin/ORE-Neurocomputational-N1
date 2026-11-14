# ORE Neurocomputational — N1 through N8

A modular Python research implementation of ORE's early visual processing (N1), visual organization (N2), recurrent orthographic prediction (N3), attention and precision gating (N4), verified experience-dependent plasticity (N5), hierarchical orthographic/lexical access (N6), predictive and contextual processing (N7), and lexical-semantic access/composition (N8). Each phase is an additive package with explicit contracts, numerical operations, diagnostics, and tests.

> Scientific scope: ORE is a biologically motivated computational approximation. It is not a complete cellular or brain simulation, and software tests do not establish physiological correspondence, OCR accuracy, semantic correctness, or general intelligence. Parameters require empirical calibration and independent evaluation.

## Installation

Python 3.10+ required.

~~~powershell
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest
~~~

## Phase overview

| Phase | Operations | Primary function |
|---|---:|---|
| N1 | 12 | Early visual and orientation-selective processing |
| N2 | 17 | Multiscale visual organization, contours, figure-ground, and shape structure |
| N3 | 29 | Recurrent grapheme hypotheses and orthographic prediction error |
| N4 | 40 | Competitive attention and precision-gated inference |
| N5 | 52 | Verified experience-dependent prototype plasticity and transition learning |
| N6 | 60 | Bidirectional grapheme-sequence decoding, word-boundary estimation, and orthographic lexical access |
| N7 | 72 | Hierarchical generative predictions, precision-weighted errors, contextual inference and belief revision |
| N8 | 72 | Context-sensitive lexical-semantic access and explicit relation composition |

Detailed phase documents are `N2_ARCHITECTURE.md`, `docs/N3_ARCHITECTURE.md` through `docs/N8_ARCHITECTURE.md`.

## N1–N5 integration

N1 produces orientation-population features. N2 builds visual shape and structure from N1 spatial maps. N3 learns labelled N2 patch descriptors and grapheme transition statistics. N4 uses N3 visual beliefs, contextual priors, and prediction errors to allocate attention and precision. N5 is a distinct, explicitly authorized learning stage: it requires independently verified labels and provenance, preserves the parent model, and returns a detached child with fast/slow prototype banks, eligibility traces, replay, and transition memory.

See `docs/N5_ARCHITECTURE.md` for the complete 52-operation algorithm and safeguards.

## N6 hierarchical orthographic sequence and lexical access (60 operations)

The independent `ore_n6` package consumes an N4 result and an immutable N5 model snapshot. It calibrates N4 grapheme posteriors with N5 class gains, estimates left/right context from fast and slow transition memories, performs Viterbi and forward/backward sequence inference, preserves ranked grapheme alternatives, estimates word-boundary affinities, and searches an externally supplied provenance-checked orthographic lexicon. A k-best segmentation lattice preserves alternative word groupings. When no lexicon entry is available, explicit unlexicalized-grapheme fallback tokens preserve complete input coverage.

N6 does not modify N4 or N5, add lexical labels, or infer word meaning. Lexicon entries require a verifier and source reference; those fields support traceability but do not independently prove correctness.

## N7 predictive and contextual processing (72 operations)

`ore_n7` consumes a completed N6 result and explicit context-provider evidence. It preserves visual, orthographic and contextual channels, forms hierarchical cue predictions, computes precision-weighted prediction errors, and revises an intent/domain posterior using a bounded variational update. It keeps context memory non-mutating and preserves uncertainty and multiple hypotheses.

See `docs/N7_ARCHITECTURE.md` for the full N7 operation trace, equations, API and limitations.

## N8 lexical-semantic access and composition (72 operations)

`ore_n8` consumes the matching N6 token sequence together with its N7 contextual result. It performs exact-form lookup against a caller-supplied semantic lexicon, uses N7 intent/domain posteriors as contextual compatibility signals, preserves alternative senses and unknown tokens, and builds a compositional concept graph only from explicit relation evidence. Semantic entries and relation evidence require verifier/source provenance by default. N8 does not generate missing definitions or infer grammatical relations from token order.

Quick start:

~~~python
from ore_n8 import N8Config, N8Input, N8Processor

result = N8Processor(N8Config()).process(N8Input(
    n6_result=n6_result,
    n7_result=n7_result,
    semantic_lexicon=verified_semantic_lexicon,
    semantic_context={"global_sense_weights": caller_sense_weights},
))
print(result.token_semantics)
print(result.semantic_graph)
print(result.diagnostics["operation_trace_complete"])
~~~

Run focused tests with `python -m pytest tests/test_n8.py -q` and the full suite with `python -m pytest`. See `docs/N8_ARCHITECTURE.md` for all 72 operations, equations, contracts, API and scientific limitations.
