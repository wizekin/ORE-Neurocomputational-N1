# ORE Neurocomputational — N1 through N6

A modular Python research implementation of ORE's early visual processing (N1), visual organization (N2), recurrent orthographic prediction (N3), attention and precision gating (N4), verified experience-dependent plasticity (N5), and hierarchical orthographic sequence/lexical access (N6). Each phase is an additive package with explicit contracts, numerical operations, diagnostics, and tests.

> Scientific scope: ORE is a biologically motivated computational approximation. It is not a complete cellular or brain simulation, and the software tests do not establish physiological correspondence, OCR accuracy, or general intelligence. Parameters require empirical calibration and independent evaluation.

## Installation

Python 3.10+ is required.

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

The detailed phase documents are: N2_ARCHITECTURE.md, docs/N3_ARCHITECTURE.md, docs/N4_ARCHITECTURE.md, docs/N5_ARCHITECTURE.md, and docs/N6_ARCHITECTURE.md.

## N1–N5 integration

N1 produces orientation-population features. N2 builds visual shape and structure from N1 spatial maps. N3 learns labelled N2 patch descriptors and grapheme transition statistics. N4 uses N3 visual beliefs, contextual priors, and prediction errors to allocate attention and precision. N5 is a distinct, explicitly authorized learning stage: it requires independently verified labels and provenance, preserves the parent model, and returns a detached child with fast/slow prototype banks, eligibility traces, replay, and transition memory.

See docs/N5_ARCHITECTURE.md for the complete 52-operation algorithm and safeguards.

## N6 hierarchical orthographic sequence and lexical access (60 operations)

The independent ore_n6 package consumes an N4 result and an immutable N5 model snapshot. It calibrates N4 grapheme posteriors with N5 class gains, estimates left/right context from the fast and slow transition memories, performs Viterbi and forward/backward sequence inference, preserves ranked grapheme alternatives, estimates word-boundary affinities, and searches an externally supplied provenance-checked orthographic lexicon. A k-best segmentation lattice preserves alternative word groupings. When no lexicon entry is available, explicit unlexicalized-grapheme fallback tokens preserve complete input coverage.

N6 does not modify N4 or N5, add lexical labels, or infer word meaning. Lexicon entries require a verifier and source reference; those fields support traceability but do not independently prove correctness.

Quick start:

~~~python
from ore_n6 import N6Config, N6Input, N6LexiconEntry, N6Processor

lexicon = [
    N6LexiconEntry(
        entry_id="word-001",
        graphemes=("a", "b"),
        frequency=12.0,
        verified_by="independent-reviewer",
        source_reference="dataset://release/word-001",
    )
]
result = N6Processor(N6Config()).process(
    N6Input(
        n4_result=n4_result,
        n5_model=n5_model_state,
        lexicon_entries=lexicon,
        n2_features=optional_n2_patch_feature_matrix,  # optional same-space N2 descriptors
    )
)
print(result.decoded_graphemes)
print(result.tokens)
print(result.diagnostics["valid"])
~~~

Run the focused N6 tests with python -m pytest tests/test_n6.py -q and the full suite with python -m pytest. See docs/N6_ARCHITECTURE.md for all 60 operations, equations, contracts, examples, limitations, and validation requirements.
