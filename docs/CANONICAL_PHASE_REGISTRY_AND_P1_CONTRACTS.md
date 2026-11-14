# ORE canonical phase registry and P1 contracts

Specification ORE-ARCH-REGISTRY-001 v1.0.0. This additive change freezes the P0 registry and implements P1, contracts IF-00 through IF-04.

## Canonical phase registers

Functional: F-A through F-J. Biological: B-N1 through B-N9.

| ID | Canonical responsibility |
|---|---|
| B-N1 | Neural simulation foundations |
| B-N2 | Biological visual processing |
| B-N3 | Hierarchical visual architecture |
| B-N4 | Neural orthographic representations |
| B-N5 | Neural connectivity and recurrence |
| B-N6 | Biological learning and memory |
| B-N7 | Predictive and contextual processing |
| B-N8 | Brain-data validation |
| B-N9 | Integrated neurocomputational ORE |

Phase identifiers are stable. Operation counts are revision metadata, not identifiers. Status values are independent: SPECIFIED, IMPLEMENTED, SOFTWARE_TESTED, TASK_VALIDATED and BIOLOGICALLY_SUPPORTED must never be inferred from one another.

## Legacy compatibility

Existing ore_n1 through ore_n9 packages predate this canonical register and use overlapping N labels for different responsibilities. They are not renamed or rewritten by this change. They remain legacy implementation packages during migration and must not be cited as proof that canonical B-N1 through B-N9 are implemented. New code and reports use explicit F-* and B-N* namespaces. The old N9 lexical, syntax and semantic proposal is not reassigned to B-N8 or B-N9.

## P1 contracts

- IF-00 ContractEnvelope: schema/run/producer identity, SHA-256 configuration and data fingerprints, UTC timestamp, evidence kind and parent lineage.
- IF-01 ScriptProfile: script identity, grapheme segmentation and Unicode policies, LTR/RTL/TTB/BTT/MIXED direction, logical axis and geometry constraints. Physical image coordinates remain separate from logical reading order.
- IF-02 RunContext: operation mode, deterministic seed, compute budget, backend selection, timeout limits and immutable model references.
- IF-03 VisualObservation: immutable source reference/hash, image geometry, numeric range, pixel or patch payload reference, segmentation alternatives, masks and transformation provenance.
- IF-04 DatasetManifest: dataset/license/provenance, per-sample hashes, label provenance, writer/font/source/string-family groups, split assignment, seed and leakage checks. Preregistered test splits remain locked.

The ore_contracts package uses the Python standard library and does not depend on a neural simulator or OCR framework.

## Test

Run python -m pip install -e ".[dev]" followed by python -m pytest. The P1 test module covers registry uniqueness, IF-00 hashing and round-trip, UTC validation, RTL/LTR profiles, mixed-layout policy, run budgets, visual observation round-trip, dataset round-trip, group leakage and locked test splits.

## Scope boundary

This change is a registry and contract foundation, not a claim that the engines are integrated. P2 (IF-05 through IF-08), P3 (IF-18), remaining shared interfaces, adapters and the B-N9 controller remain separate backlog items. Passing software tests does not establish OCR task validity or biological correspondence.
