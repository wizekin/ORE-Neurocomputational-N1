import pytest

from ore_n1 import N1Config


def test_config_rejects_invalid_scales():
    with pytest.raises(ValueError):
        N1Config(center_sigma=3.0, surround_sigma=2.0)
    with pytest.raises(ValueError):
        N1Config(gabor_kernel_size=10)
    with pytest.raises(ValueError):
        N1Config(orientations=(0.0, 3.141592653589793))
