# ORE canonical shared interfaces: P2–P8 implementation

**Architecture:** ORE-ARCH-REGISTRY-001 v1.0.0  
**Depends on:** merged P0/P1 canonical registry and IF-00–IF-04  
**Package:** \`ore_contracts\`  
**Scope:** shared contracts, a reference B-N1 simulation substrate, guarded-learning and evaluation infrastructure, legacy adapters, and the B-N9 orchestration shell.

## Delivery matrix

| Priority | Delivered | Main code |
|---|---|---|
| P2 | IF-05 visual features, IF-06 grapheme evidence, IF-07 C/A/B/O position channels, IF-08 orthographic memory | \`advanced.py\` |
| P3 | IF-18 simulator-neutral neural snapshot/protocol and deterministic rate-based reference simulator | \`advanced.py\`, \`simulation.py\` |
| P4 | IF-09 learning request, IF-11 model reference (from P1), IF-15 learning result and detached-child transaction manager | \`advanced.py\`, \`governance.py\` |
| P5 | IF-10 prediction state, IF-12 hypotheses, IF-13 recognition result, IF-14 append-only trace event | \`advanced.py\` |
| P6 | IF-16 preregistered experiment specification, IF-17 separate evaluation report, fixed-seed replicates and independent-reference gates | \`advanced.py\` |
| P7 | Adapters for legacy N1/N2 visual outputs, N3 grapheme probabilities, N4 prediction/attention, N5 memory, N6 alternatives, N9 results, and neural state snapshots | \`adapters.py\` |
| P8 | Dependency-injected functional/biological/hybrid/validation controller, backend registry, explicit unavailable states, frozen fusion policy and legacy N9 functional bridge | \`integration.py\` |

## Evidence and ownership safeguards

- Every feature tensor declares its representation kind, layout, dtype and units.
- Raw visual evidence, predictions, posterior beliefs, precision and neural state remain distinct.
- The position bundle requires explicit C/A/B/O channels. The legacy N4 adapter marks these unavailable because N4 does not expose those codes; it does not relabel attention as position.
- The neural simulator records membrane potential in mV, firing rate in Hz, synaptic-current units, integration step, stable unit indices and a connectivity fingerprint. The included simulator is rate-based and does not generate spike events.
- Learning requires explicit authorization, a verified target/provenance policy or named unsupervised regime, a matching parent fingerprint, a detached proposal and at least one regression gate. Accepted children are returned separately; durable promotion is not performed.
- Task, robustness, cost, behavioral and neural evaluation statuses are independent. Held-out task/robustness/cost metrics must use the TEST split. Biological correspondence requires independent reference metrics.
- B-N8 is invoked only in offline VALIDATE mode. Inference modes run B-N1…B-N7; the B-N9 controller coordinates rather than becoming another recognition algorithm.
- Hybrid fusion is disabled unless a validation-fitted, frozen combiner and its validation-manifest reference are registered.

## Legacy compatibility policy

The existing \`ore_n1\`–\`ore_n9\` packages remain unchanged. The new adapters label them as \`legacy.ore_n*\`; they do not equate old N labels with canonical B-N phase IDs. In particular, the legacy N5 memory adapter uses a legacy functional-memory namespace, and the legacy N9 adapter explicitly reports that its output is not evidence of canonical B-N9 integration.

The \`LegacyN9FunctionalBackend\` is a compatibility bridge for existing \`ore_n9.N9Request\` callers. It is not declared to implement the canonical functional F-A…F-J register.

## Runtime use

A caller registers a functional engine exposing \`run(payload, context)\` and optional canonical biological phase backends exposing \`run(PhaseInvocation)\`. The controller:
- runs the functional branch in FUNCTIONAL/TRAIN/INFER modes;
- runs injected B-N1…B-N7 stages in BIOLOGICAL mode;
- runs both branches side-by-side in HYBRID mode;
- runs only an injected B-N8 validator in VALIDATE mode;
- reports missing implementations as \`CHANNEL_UNAVAILABLE\`, failures as \`FAILED\`, and blocked downstream phases explicitly.

The included \`RateBasedB1PhaseBackend\` is a runnable B-N1 reference substrate. Canonical B-N2…B-N8 biological phase algorithms still need their own scientifically specified implementations and independent evidence; they are not fabricated by the controller.

## Tests

Run:

\`\`\`bash
python -m pip install -e ".[dev]"
python -m pytest
\`\`\`

The added \`tests/test_advanced_contracts.py\` covers the new schemas, round-trip/validation behavior, deterministic neural stepping, learning immutability and rollback, split governance, adapter provenance, alternative preservation, backend routing, fusion prerequisites and B-N8 offline validation.

Passing software tests establish software-contract behavior only. They do not establish task accuracy, biological correspondence or reproduction of human neural mechanisms.
