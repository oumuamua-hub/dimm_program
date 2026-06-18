import numpy as np

from dimm_analyzer.config import AnalysisConfig
from dimm_analyzer.gaussian_fit import fit_gaussian_2d
from dimm_analyzer.models import FitResult


def synthetic_gaussian(size=31, x0=15.4, y0=14.7, amplitude=2500.0, background=120.0):
    yy, xx = np.indices((size, size), dtype=float)
    data = background + amplitude * np.exp(
        -(((xx - x0) ** 2) / (2 * 1.8**2) + ((yy - y0) ** 2) / (2 * 2.1**2))
    )
    rng = np.random.default_rng(42)
    return data + rng.normal(0, 4.0, data.shape)


def test_fit_recovers_center():
    config = AnalysisConfig()
    roi = synthetic_gaussian()
    result = fit_gaussian_2d(
        roi,
        origin_x=100,
        origin_y=200,
        guess_x=115.2,
        guess_y=214.9,
        fit_config=config.gaussian_fit,
        quality_config=config.quality,
        saturation_level=65535,
    )
    assert result.success
    assert result.x0_global == pytest_approx(115.4, abs=0.25)
    assert result.y0_global == pytest_approx(214.7, abs=0.25)


def test_saturated_roi_is_rejected():
    config = AnalysisConfig()
    roi = synthetic_gaussian()
    roi[15, 15] = 65535
    result = fit_gaussian_2d(
        roi,
        origin_x=0,
        origin_y=0,
        guess_x=15,
        guess_y=15,
        fit_config=config.gaussian_fit,
        quality_config=config.quality,
        saturation_level=65535,
    )
    assert not result.success
    assert result.failure_reason == "saturated"
    assert result.peak == 65535
    assert result.peak_raw_max == 65535
    assert result.peak_core_max == 65535
    assert result.saturated_core_pixel_count == 1
    assert result.saturated_roi_pixel_count == 1
    assert result.hot_pixel_or_roi_outlier is False


def test_saturated_roi_edge_pixel_is_hot_pixel_outlier():
    config = AnalysisConfig()
    roi = synthetic_gaussian()
    roi[0, 0] = 65535
    result = fit_gaussian_2d(
        roi,
        origin_x=0,
        origin_y=0,
        guess_x=15,
        guess_y=15,
        fit_config=config.gaussian_fit,
        quality_config=config.quality,
        saturation_level=65535,
    )
    assert not result.success
    assert result.failure_reason == "hot_pixel_or_roi_outlier"
    assert result.peak_raw_max == 65535
    assert result.peak_core_max < 65535
    assert result.saturated_core_pixel_count == 0
    assert result.saturated_roi_pixel_count == 1
    assert result.hot_pixel_or_roi_outlier is True


def test_saturation_core_radius_and_min_pixels_are_configurable():
    config = AnalysisConfig()
    config.quality.saturation_core_radius_px = 0.5
    config.quality.saturation_core_min_pixels = 2
    roi = synthetic_gaussian()
    roi[15, 15] = 65535
    result = fit_gaussian_2d(
        roi,
        origin_x=0,
        origin_y=0,
        guess_x=15,
        guess_y=15,
        fit_config=config.gaussian_fit,
        quality_config=config.quality,
        saturation_level=65535,
    )

    assert result.success
    assert result.saturated_core_pixel_count == 1


def test_failed_fit_result_can_carry_saturation_diagnostics():
    result = FitResult.failed(
        "hot_pixel_or_roi_outlier",
        peak_raw_max=65535,
        peak_core_max=2400,
        saturated_core_pixel_count=0,
        saturated_roi_pixel_count=1,
        hot_pixel_or_roi_outlier=True,
    )

    assert not result.success
    assert result.failure_reason == "hot_pixel_or_roi_outlier"
    assert result.peak_raw_max == 65535
    assert result.peak_core_max == 2400
    assert result.saturated_core_pixel_count == 0
    assert result.saturated_roi_pixel_count == 1
    assert result.hot_pixel_or_roi_outlier is True


def pytest_approx(*args, **kwargs):
    import pytest

    return pytest.approx(*args, **kwargs)
