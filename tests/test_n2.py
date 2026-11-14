import numpy as np
import pytest
from ore_n1 import N1Config, N1Processor
from ore_n2 import N2Config, N2Processor, adapt_n1_result

def stimulus(size=64):
    image=np.zeros((size,size),np.float32)
    image[size//4:3*size//4,size//3:2*size//3]=1.0
    return image

def test_n2_runs_all_17_steps_and_preserves_spatial_channels():
    n1=N1Processor(N1Config()).process(stimulus())
    result=N2Processor().process(n1)
    assert result.diagnostics["valid"]
    assert result.diagnostics["stage_count"]==17
    assert len(result.metadata["steps"])==17
    # Multiscale layout is [scale, orientation, height, width].
    # Assert spatial dimensions without discarding the scale/orientation axes.
    multiscale = result.representations["multiscale"]
    assert multiscale.ndim == 4
    assert multiscale.shape[0] == len(N2Config().scales)
    assert multiscale.shape[1] == n1.stage_outputs["normalized_energy"].shape[0]
    assert multiscale.shape[-2:] == (64, 64)
    assert result.representations["contour"].shape == (64, 64)
    assert result.structured_output["schema"]=="ore.n2.structured_visual_output.v1"
    assert "lexical semantics" in result.structured_output["interpretation_scope"]

def test_n2_rejects_flattened_vector_without_spatial_maps():
    with pytest.raises(ValueError,match="normalized_energy"):
        adapt_n1_result({"feature_vector":np.zeros(10,np.float32)})

def test_n2_rejects_nonfinite_spatial_maps():
    with pytest.raises(ValueError,match="non-finite"):
        adapt_n1_result({"normalized_energy":np.full((4,8,8),np.nan)})

def test_recurrent_stage_is_bounded():
    from ore_n2.stages import n2_15_recurrent_refinement
    cfg=N2Config(recurrent_iterations=3)
    x=np.ones((16,16),np.float32)
    result=n2_15_recurrent_refinement(x,x,cfg)
    assert result["iterations"]<=3
    assert np.isfinite(result["refined_activity"]).all()

def test_constant_input_produces_finite_n2_result():
    n1=N1Processor().process(np.ones((32,32),np.float32)*0.5)
    result=N2Processor().process(n1)
    assert all(np.isfinite(v).all() for v in result.representations.values())
