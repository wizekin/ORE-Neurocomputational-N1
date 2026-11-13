"""Pure numerical functions implementing N1 operations 1 through 10."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve, gaussian_filter

from .filters import build_gabor_bank


def as_grayscale_float(image: np.ndarray) -> np.ndarray:
    """Validate a 2-D grayscale image and convert it to float32 in [0, 1].

    Integer inputs are scaled by their dtype maximum (e.g. uint8 -> /255).
    Floating inputs must already be finite and in [0, 1].
    """
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale image, received shape {arr.shape}")
    if min(arr.shape) < 4:
        raise ValueError("image dimensions must both be at least 4 pixels")
    if np.issubdtype(arr.dtype, np.integer):
        if np.issubdtype(arr.dtype, np.signedinteger) and np.any(arr < 0):
            raise ValueError("signed integer images must not contain negative values")
        scale = float(np.iinfo(arr.dtype).max)
        arr = arr.astype(np.float32) / scale
    else:
        arr = arr.astype(np.float32, copy=False)
    if not np.isfinite(arr).all():
        raise ValueError("input image contains NaN or infinite values")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError("floating-point image values must lie in [0, 1]")
    return np.ascontiguousarray(arr, dtype=np.float32)


def photoreceptor_adaptation(image: np.ndarray, config) -> tuple[np.ndarray, np.ndarray]:
    """1. Local luminance adaptation plus a compressive logarithmic response.

    The adaptation field is a Gaussian estimate of local illumination. The
    response is log(1 + I / (A + floor)), an engineering approximation of
    photoreceptor gain adaptation and response compression.
    """
    adaptation = gaussian_filter(image, sigma=config.photoreceptor_sigma, mode="reflect")
    response = np.log1p(image / (adaptation + config.photoreceptor_floor))
    return response.astype(np.float32), adaptation.astype(np.float32)


def retinal_on_off(response: np.ndarray, config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2. Produce rectified ON and OFF channels from local contrast polarity."""
    local_reference = gaussian_filter(response, sigma=config.bipolar_sigma, mode="reflect")
    signed_contrast = response - local_reference
    on = np.maximum(signed_contrast, 0.0)
    off = np.maximum(-signed_contrast, 0.0)
    return on.astype(np.float32), off.astype(np.float32), signed_contrast.astype(np.float32)


def center_surround(on: np.ndarray, off: np.ndarray, config) -> tuple[np.ndarray, np.ndarray]:
    """3. Apply a difference-of-Gaussians center-surround to ON and OFF paths."""
    def dog(channel: np.ndarray) -> np.ndarray:
        center = gaussian_filter(channel, sigma=config.center_sigma, mode="reflect")
        surround = gaussian_filter(channel, sigma=config.surround_sigma, mode="reflect")
        return np.maximum(center - surround, 0.0).astype(np.float32)
    return dog(on), dog(off)


def lgn_gain_control(on: np.ndarray, off: np.ndarray, config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """4. Local contrast-gain control using a smoothed ON/OFF energy estimate."""
    energy = on * on + off * off
    local_rms = np.sqrt(gaussian_filter(energy, sigma=config.lgn_sigma, mode="reflect") + 1e-12)
    gain = 1.0 / np.power(local_rms + config.lgn_gain_floor, config.lgn_gain_exponent)
    return (on * gain).astype(np.float32), (off * gain).astype(np.float32), gain.astype(np.float32)


def v1_orientation_population(on: np.ndarray, off: np.ndarray, config) -> np.ndarray:
    """5. Compute even/odd oriented linear responses for each V1 orientation.

    Output layout is [orientation, quadrature_phase, y, x]. ON and OFF LGN
    channels are combined as nonnegative afferent drive before oriented filtering.
    """
    afferent_drive = on + off
    bank = build_gabor_bank(config)
    responses = []
    for even_kernel, odd_kernel in bank:
        even = convolve(afferent_drive, even_kernel, mode="reflect")
        odd = convolve(afferent_drive, odd_kernel, mode="reflect")
        responses.append(np.stack((even, odd), axis=0))
    return np.stack(responses, axis=0).astype(np.float32)


def simple_cell_responses(linear_responses: np.ndarray) -> np.ndarray:
    """6. Retain positive and negative polarity of each phase-specific response.

    Output layout is [orientation, phase, polarity(ON/OFF), y, x].
    """
    positive = np.maximum(linear_responses, 0.0)
    negative = np.maximum(-linear_responses, 0.0)
    return np.stack((positive, negative), axis=2).astype(np.float32)


def complex_cell_energy(simple_responses: np.ndarray) -> np.ndarray:
    """7. Combine quadrature simple-cell pairs into phase-tolerant energy.

    Squared positive and negative polarity responses reconstruct each signed
    phase's energy; the even/odd pair is then combined by a square root.
    Output layout is [orientation, y, x].
    """
    phase_energy = np.sum(simple_responses * simple_responses, axis=2)
    energy = np.sqrt(np.sum(phase_energy, axis=1) + 1e-12)
    return energy.astype(np.float32)


def divisive_normalization(energy: np.ndarray, config) -> tuple[np.ndarray, np.ndarray]:
    """8. Normalize each orientation by a local pooled population energy.

    denominator = (epsilon**p + GaussianPool(sum_orientation(E**p)))**(1/p)
    """
    p = config.normalization_exponent
    population = np.sum(np.power(np.maximum(energy, 0.0), p), axis=0)
    pooled_population = gaussian_filter(population, sigma=config.normalization_sigma, mode="reflect")
    denominator = np.power(config.normalization_epsilon ** p + pooled_population, 1.0 / p)
    normalized = energy / denominator[None, :, :]
    return normalized.astype(np.float32), denominator.astype(np.float32)


def spatial_pooling(activity: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """9. Adaptive non-overlapping mean pooling to a fixed spatial grid."""
    grid_h, grid_w = grid
    height, width = activity.shape[-2:]
    if grid_h > height or grid_w > width:
        raise ValueError(f"pooling grid {grid} cannot exceed image shape {(height, width)}")
    y_bins = np.array_split(np.arange(height), grid_h)
    x_bins = np.array_split(np.arange(width), grid_w)
    pooled = np.empty((activity.shape[0], grid_h, grid_w), dtype=np.float32)
    for gy, ys in enumerate(y_bins):
        for gx, xs in enumerate(x_bins):
            patch = activity[:, ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1]
            pooled[:, gy, gx] = patch.mean(axis=(-2, -1))
    return pooled


def construct_feature_vector(pooled: np.ndarray) -> np.ndarray:
    """10. Flatten orientation-major pooled maps into a stable float32 vector."""
    return np.ascontiguousarray(pooled.reshape(-1), dtype=np.float32)
