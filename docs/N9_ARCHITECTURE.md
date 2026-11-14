# ORE N9 — Integrated Neurocomputational ORE

**Package:** \`ore_n9/\`  
**Algorithm:** 120 ordered operations in ten subsystems  
**Status:** Software integration prototype; real-corpus and biological validation remain pending  
**Compatibility:** Additive. Existing N1-N4 packages and outputs are not replaced or modified.

## 1. Repository boundary

This repository snapshot contains N1, N2, N3 and N4. N5-N8 are not present as first-party packages in this branch. N9 therefore:

- invokes N1 and N2 automatically for supplied, already-localized grayscale image patches;
- can invoke a separately fitted N3Processor and the N4 attention/precision processor;
- exposes explicit extension adapters for N5 learning, N6 lexical access, N7 context/discourse inference, N8 semantic composition, and syntactic parsing;
- records missing adapters as \`not_configured\` instead of claiming that the missing stage ran.

The archived roadmap and later implementation line use different N-stage registers. This package follows the newer N1-N8 implementation line.

## 2. Inputs

A request is an ordered list of localized patches. Each patch must provide exactly one of:

1. \`image\`: 2-D grayscale values, suitable for the existing N1 pipeline;
2. \`n2_result\`: a result from the current N2 package;
3. \`features\`: a finite one-dimensional feature vector;
4. \`grapheme_probabilities\`: a mapping from grapheme labels to non-negative weights.

N9 deliberately does not invent character segmentation for a full line image. A caller or future segmentation module must provide localized patches and their logical reading order. The optional \`word_spans\` field provides half-open patch ranges for lexical grouping.

Example:

\`\`\`python
from ore_n9 import N9Patch, N9Processor, N9Request

request = N9Request(
    sequence_id="demo",
    patches=[
        N9Patch("p0", grapheme_probabilities={"a": 0.9, "o": 0.1}),
        N9Patch("p1", grapheme_probabilities={"b": 0.8, "d": 0.2}),
    ],
    word_spans=[(0, 2)],
    script_profile={"script": "Latin", "reading_direction": "ltr"},
)
result = N9Processor().process(request)
assert len(result.operation_trace) == 120
\`\`\`

## 3. Existing processor integration

For image patches N9 calls \`N1Processor.process\` and \`N2Processor.process\`. When a fitted N3 processor is supplied, N9 builds an N3Sequence and uses its visual candidate probabilities, transition model and prediction-error outputs. N9 then passes that N3 result to N4 with the request's task relevance and segmentation reliability.

\`\`\`python
from ore_n9 import N9Adapters, N9Processor

adapters = N9Adapters(n3=fitted_n3_processor)
result = N9Processor(adapters=adapters).process(request)
\`\`\`

N3 training remains explicit: N9 does not create labels or train a grapheme model from an unlabelled recognition request.

## 4. Extension adapter contracts

Each extension receives a mapping payload and returns a mapping. An adapter may be a callable or an object exposing the named method:

| Slot | Expected method | Output keys consumed |
|---|---|---|
| \`n5\` | \`propose_update\` / \`train_child\` | \`child_model\`, optional \`metadata\` |
| \`n6\` | \`resolve\` / \`process\` / \`predict\` | \`lexical_hypotheses\`, \`morphology\`, \`phonological_forms\`, \`pos_features\`, optional \`syntax\` |
| \`syntax\` | \`parse\` / \`process\` / \`predict\` | \`parses\`, \`thematic_roles\`, \`scope\`, \`phrases\`, \`consistency\`, \`coreference\` |
| \`n7\` | \`predict\` / \`process\` / \`infer\` | \`intents\`, \`predictions\`, \`situation_model\`, \`coherence\`, \`uncertainty\`, \`temporal_links\`, \`causal_links\` |
| \`n8\` | \`compose\` / \`process\` / \`resolve\` | \`semantic_graph\`, \`propositions\`, \`completeness\`, \`ambiguity\`, optional feedback |

An N8 semantic graph has \`nodes\` with unique non-empty \`id\` and \`type\` fields, and \`edges\` with valid \`source\`, \`target\` node IDs and a non-empty \`relation\`. Dangling or untyped edges are discarded. The controller does not infer missing semantic relations.

## 5. The 120 operations

| Range | Subsystem | Main implementation |
|---:|---|---|
| 1–12 | Input orchestration and state initialization | Contract checks, source digest, masks, coordinates, reliability, context snapshot, model fingerprints and dependency graph |
| 13–24 | Visual processing and organization | N1/N2 intermediate maps and descriptors, segmentation hypotheses, grouping and alignment records |
| 25–36 | Position coding and orthographic sequence | Grapheme distributions, UNK handling, absolute/relative position codes, overlap matrix, transition memory and beam candidates |
| 37–48 | Attention and precision | Entropy, prediction error, task salience, competitive allocation, precision estimates and optional N4 gating |
| 49–60 | Sequence and lexical interfaces | Bidirectional transition cues, bounded evidence combination, N6 lexical/morphological/phonological adapter and lexical abstention |
| 61–72 | Syntax and semantic composition | Explicit grammatical features, parser adapter, thematic roles, scope, discourse links, N8 semantic graph and propositions |
| 73–84 | Context and situation model | Caller context, N7 predictions, coherence, temporal/causal links, conflicts and uncertainty |
| 85–96 | Hierarchical feedback | Top-down projections, precision-weighted errors, free-energy proxy, damped posterior revision and convergence |
| 97–108 | Global decision and output | Cross-level integration, confidence, abstention, explanations, rollback and integrity checks |
| 109–120 | Controlled learning and acceptance | Inference freeze, authorization, detached child proposal, transition statistics, provenance, ablations and separate acceptance states |

Every request returns a trace containing exactly 120 ordered operation records. The status field distinguishes executed, fallback, unavailable and not-applicable operations.

## 6. Mathematical core

For patch \(i\), let \(v_i(c)\) be preserved visual evidence, \(p_i(c)\) a contextual or transition prior, and \(b_i(c)\) the integrated posterior. The bounded proposal has the form:

\[
\hat b_i = \operatorname{softmax}\left(
\pi_{v,i}\log(v_i+\epsilon)
+\pi_{p,i}\log(p_i+\epsilon)
+\gamma_f\log(f_i+\epsilon)
\right)
\]

\[
b_i^{(t+1)}=(1-\rho)b_i^{(t)}+\rho\hat b_i
\]

The implementation retains \`visual_probabilities\` separately from \`posterior_beliefs\`. Context and lexical feedback can adjust the posterior, but cannot rewrite the preserved bottom-up distribution. This is a computational approximation; it is not asserted to be a literal neuronal equation.

A free-energy-like diagnostic combines visual cross-entropy and a divergence term against the transition prior. It is a comparison metric for software experiments, not a measured quantity in the human brain.

## 7. Learning safety

Inference does not update active models. Learning is skipped unless the request sets \`learning_authorized=True\`. When authorized, N9 deep-copies the N5 adapter and asks the detached copy to propose a child model. The original adapter fingerprint is compared before and after. Plain function adapters are not guaranteed to have isolated closure state, so a stateful N5 adapter should be an object whose state can be deep-copied. A changed parent fingerprint causes the proposal to be rejected and reported.

A child proposal is not automatically promoted. Versioning, provenance, held-out evaluation, anti-forgetting checks and independent review are still required.

## 8. Validation

Run:

\`\`\`powershell
python -m pip install -e ".[dev]"
python -m pytest
\`\`\`

The N9 tests cover the operation manifest, trace length/order, probability normalization, unknown fallback, N1/N2 image integration, semantic graph validation, adapter routing, opt-in learning and JSON serialization.

Acceptance categories remain separate:

- **Software contracts:** automated checks can establish these.
- **Task accuracy:** requires held-out, writer/font/script-disjoint real corpora.
- **Biological validity:** requires independent neurocognitive evidence and replication.

Passing Python tests does not establish the latter two.
