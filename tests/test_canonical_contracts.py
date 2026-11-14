from __future__ import annotations

import pytest

from ore_contracts import (
    ALL_PHASES, BIOLOGICAL_PHASES, FUNCTIONAL_PHASES, ArtifactStatus,
    ComputeBudget, ContractEnvelope, DatasetManifest, DatasetSample, ModelBundleRef,
    EvidenceKind, OperationMode, ReadingDirection, RunContext, ScriptProfile,
    SegmentationHypothesis, VisualObservation, sha256_json, validate_registry,
)

A = "a" * 64
B = "b" * 64


def make_sample(sample_id, split, writer, digest):
    return DatasetSample(sample_id, digest * 64, split, "x", "verified:review-1", writer_id=writer)


def test_canonical_registry_is_unique_and_separates_biological_n8_n9():
    validate_registry()
    assert len(FUNCTIONAL_PHASES) == 10
    assert len(BIOLOGICAL_PHASES) == 9
    assert len({p.phase_id for p in ALL_PHASES}) == 19
    assert BIOLOGICAL_PHASES[7].name == "Brain-data validation"
    assert BIOLOGICAL_PHASES[8].name == "Integrated neurocomputational ORE"
    assert all(p.status is ArtifactStatus.SPECIFIED for p in ALL_PHASES)


def test_envelope_round_trip_and_canonical_hash():
    item = ContractEnvelope("IF-00", "1.0.0", "run-1", "F-B@1.0", A,
                            "2026-11-14T10:00:00Z", EvidenceKind.OBSERVED, data_hash=B)
    restored = ContractEnvelope.from_dict(item.to_dict())
    assert restored == item
    assert restored.fingerprint() == item.fingerprint()
    assert sha256_json({"b": 2, "a": 1}) == sha256_json({"a": 1, "b": 2})


def test_envelope_rejects_invalid_hash_and_non_utc_timestamp():
    with pytest.raises(ValueError, match="SHA-256"):
        ContractEnvelope("IF-00", "1", "r", "p", "bad", "2026-11-14T10:00:00Z", EvidenceKind.OBSERVED)
    with pytest.raises(ValueError, match="UTC"):
        ContractEnvelope("IF-00", "1", "r", "p", A, "2026-11-14T10:00:00+02:00", EvidenceKind.OBSERVED)


@pytest.mark.parametrize("direction,axis", [
    (ReadingDirection.LTR, (1.0, 0.0)),
    (ReadingDirection.RTL, (-1.0, 0.0)),
])
def test_script_profile_round_trip_keeps_logical_direction(direction, axis):
    item = ScriptProfile("profile-1", "1", "test-script", direction,
                         "UNICODE_GRAPHEME_CLUSTER", "NFC", logical_reading_axis=axis)
    assert ScriptProfile.from_dict(item.to_dict()) == item


def test_mixed_direction_requires_policy():
    with pytest.raises(ValueError, match="mixed_layout_policy"):
        ScriptProfile("mixed", "1", "mixed", ReadingDirection.MIXED, "UAX29", "NFC")


def test_run_context_rejects_negative_seed_and_keeps_budget():
    item = RunContext("run-2", OperationMode.HYBRID, 17, ComputeBudget(5.0, 12, 256.0))
    assert item.to_dict()["mode"] == "HYBRID"
    assert RunContext.from_dict(item.to_dict()) == item
    model = ModelBundleRef("functional", "1.0", A, "1.0", parent_fingerprint=B)
    with_model = RunContext("run-4", OperationMode.FUNCTIONAL, 2, ComputeBudget(),
                            model_refs=(model,))
    assert RunContext.from_dict(with_model.to_dict()) == with_model
    with pytest.raises(ValueError, match="cannot be negative"):
        RunContext("run-3", OperationMode.INFER, -1, ComputeBudget())


def test_visual_observation_round_trip_preserves_segmentation_and_source_hash():
    patch = SegmentationHypothesis("p1", (2, 3, 12, 20), 0.8, 0)
    item = VisualObservation("obs-1", "dataset://image/1", A, "script-v1", 64, 32,
                             (0.0, 1.0), pixel_payload_ref="blob://immutable/1",
                             segmentation_hypotheses=(patch,))
    assert VisualObservation.from_dict(item.to_dict()) == item


def test_manifest_round_trip_and_writer_leakage_detection():
    item = DatasetManifest("ds", "1", "CC-BY-4.0", "curated",
        (make_sample("a", "TRAIN", "writer-a", "a"),
         make_sample("b", "VALIDATION", "writer-b", "b"),
         make_sample("c", "TEST", "writer-c", "c")),
        seed=11, preregistered=True)
    restored = DatasetManifest.from_dict(item.to_dict())
    assert restored == item
    assert restored.fingerprint() == item.fingerprint()
    with pytest.raises(ValueError, match="Leakage detected"):
        DatasetManifest("leak", "1", "license", "fixture",
            (make_sample("x", "TRAIN", "same", "c"),
             make_sample("y", "TEST", "same", "d")))


def test_preregistered_test_split_cannot_be_unlocked():
    with pytest.raises(ValueError, match="test split locked"):
        DatasetManifest("ds", "1", "license", "fixture",
            (make_sample("x", "TRAIN", "w1", "c"),
             make_sample("y", "TEST", "w2", "d")),
            preregistered=True, test_split_locked=False)
