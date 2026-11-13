import numpy as np
import pytest

from ore_n1 import N1Config, N1Processor, TemporalDynamics
from ore_n1.stages import as_grayscale_float, complex_cell_energy, simple_cell_responses
from ore_n1.synthetic import oriented_grating


def test_complete_pipeline_shapes_and_diagnostics():
    config = N1Config(pooling_grid=(3, 5))
    result = N1Processor(config).process(oriented_grating(shape=(64, 80)))
    assert result.pooled_features.shape == (config.orientation_count, 3, 5)
    assert result.feature_vector.shape == (config.orientation_count * 15,)
    assert result.metadata["feature_dimension"] == result.feature_vector.size
    assert result.diagnostics["valid"] is True
    assert result.diagnostics["stage_count"] >= 12
    assert all(record["finite"] for record in result.diagnostics["stages"])
    assert np.isfinite(result.feature_vector).all()


def test_constant_field_has_no_retinal_contrast_or_v1_energy():
    image = np.full((48, 48), 0.5, dtype=np.float32)
    result = N1Processor().process(image)
    assert np.max(result.stage_outputs["retinal_on"]) < 1e-6
    assert np.max(result.stage_outputs["retinal_off"]) < 1e-6
    assert np.max(result.stage_outputs["complex_cell_energy"]) < 1e-5


def test_simple_cell_quadrature_energy_is_nonnegative():
    linear = np.random.default_rng(4).normal(size=(4, 2, 16, 16)).astype(np.float32)
    simple = simple_cell_responses(linear)
    energy = complex_cell_energy(simple)
    assert simple.shape == (4, 2, 2, 16, 16)
    assert energy.shape == (4, 16, 16)
    assert np.all(energy >= 0)


def test_phase_shift_changes_simple_responses_but_preserves_complex_energy():
    cfg = N1Config(orientations=(0.0,), pooling_grid=(2, 2))
    a = N1Processor(cfg).process(
        oriented_grating(shape=(80, 80), theta=0, phase=0)
    ).stage_outputs["complex_cell_energy"]
    b = N1Processor(cfg).process(
        oriented_grating(shape=(80, 80), theta=0, phase=np.pi / 2)
    ).stage_outputs["complex_cell_energy"]
    ratio = float(np.mean(b) / max(np.mean(a), 1e-8))
    assert 0.65 < ratio < 1.5


def test_temporal_integrator_initializes_then_leaks_toward_input():
    temporal = TemporalDynamics(tau=0.2)
    first = temporal.update(np.zeros((2, 3), dtype=np.float32), dt=1 / 30)
    second = temporal.update(np.ones((2, 3), dtype=np.float32), dt=1 / 30)
    assert np.all(first == 0)
    assert np.all(second > 0)
    assert np.all(second < 1)
    temporal.reset()
    assert temporal.state is None


def test_temporal_pipeline_is_enabled_only_by_configuration():
    cfg = N1Config(temporal_enabled=True)
    processor = N1Processor(cfg)
    first = processor.process(oriented_grating(), reset_temporal=True)
    second = processor.process(np.zeros((96, 96), dtype=np.float32))
    assert first.metadata["temporal_enabled"] is True
    assert not np.array_equal(
        first.stage_outputs["temporal_activity"],
        second.stage_outputs["temporal_activity"],
    )


def test_invalid_input_is_rejected():
    with pytest.raises(ValueError):
        as_grayscale_float(np.zeros((8, 8, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        as_grayscale_float(np.full((8, 8), 2.0, dtype=np.float32))
    with pytest.raises(ValueError):
        N1Processor(N1Config(temporal_enabled=True)).process(
            np.ones((8, 8), dtype=np.float32), reset_temporal=True, dt=0
        )


def test_integer_grayscale_is_scaled():
    arr = np.full((8, 8), 128, dtype=np.uint8)
    converted = as_grayscale_float(arr)
    assert converted.dtype == np.float32
    assert np.allclose(converted, 128 / 255)
