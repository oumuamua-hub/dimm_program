from pathlib import Path

import pytest

from dimm_analyzer.config import (
    AnalysisConfig,
    effective_pixel_scale_arcsec_per_px,
    load_config,
    validate_config,
)
from dimm_analyzer.exceptions import ConfigError


def test_effective_pixel_scale_base_times_binning():
    config = AnalysisConfig()
    config.camera.binning = 2
    assert effective_pixel_scale_arcsec_per_px(config) == pytest.approx(1.286)


def test_zenith_deg_required():
    config = AnalysisConfig()
    with pytest.raises(ConfigError):
        validate_config(config)


def test_binning_two_allowed_when_zenith_provided():
    config = AnalysisConfig()
    config.camera.binning = 2
    config.instrument.zenith_deg = 12.0
    validate_config(config)
    assert effective_pixel_scale_arcsec_per_px(config) == pytest.approx(1.286)


def test_default_roi_sizes():
    config = AnalysisConfig()
    assert config.roi.size_px == 25
    assert config.roi.fallback_size_px == 31
    assert config.roi.safety_margin_px == 5.0
    assert config.roi.auto_safety_check is True
    assert config.roi.auto_shrink_if_unsafe is False
    assert config.roi.min_allowed_size_px == 21
    assert config.roi.max_allowed_size_px == 41
    assert config.roi.safety_reference_percentile == 5.0
    assert config.roi.unsafe_fraction_threshold == 0.01
    assert config.roi.warning_fraction_threshold == 0.05


def test_default_quality_saturation_and_tracking_fields():
    config = AnalysisConfig()
    assert config.quality.saturation_level is None
    assert config.quality.reject_spot_tracking is True
    assert config.quality.max_spot_tracking_distance_px == 15.0
    assert config.quality.reject_fwhm_outlier is True
    assert config.quality.max_fwhm_relative_deviation == 0.5
    assert config.quality.fwhm_reference == "running_median"
    assert config.quality.fwhm_window_frames == 101
    assert config.quality.reject_bad_relative_motion is True
    assert config.quality.max_relative_motion_deviation_px == 10.0
    assert config.quality.relative_motion_reference == "running_median"
    assert config.quality.relative_motion_window_frames == 101
    assert config.quality.saturation_core_radius_px == 2.0
    assert config.quality.saturation_core_min_pixels == 1
    assert config.quality.reject_saturated_core is True
    assert config.quality.classify_roi_saturated_pixel_as_hot_pixel is True


def test_vmc260l_yaml_loads_all_roi_and_quality_fields():
    config_path = Path(__file__).parents[1] / "configs" / "vmc260l_imx432.yaml"
    config = load_config(config_path)

    assert config.roi.safety_margin_px == 5
    assert config.roi.auto_safety_check is True
    assert config.roi.auto_shrink_if_unsafe is False
    assert config.roi.min_allowed_size_px == 21
    assert config.roi.max_allowed_size_px == 41
    assert config.roi.safety_reference_percentile == 5
    assert config.roi.unsafe_fraction_threshold == 0.01
    assert config.roi.warning_fraction_threshold == 0.05
    assert config.quality.saturation_level is None
    assert config.quality.reject_spot_tracking is True
    assert config.quality.max_spot_tracking_distance_px == 15.0
    assert config.quality.reject_fwhm_outlier is True
    assert config.quality.max_fwhm_relative_deviation == 0.5
    assert config.quality.fwhm_reference == "running_median"
    assert config.quality.fwhm_window_frames == 101
    assert config.quality.reject_bad_relative_motion is True
    assert config.quality.max_relative_motion_deviation_px == 10.0
    assert config.quality.relative_motion_reference == "running_median"
    assert config.quality.relative_motion_window_frames == 101
    assert config.quality.saturation_core_radius_px == 2.0
    assert config.quality.saturation_core_min_pixels == 1
    assert config.quality.reject_saturated_core is True
    assert config.quality.classify_roi_saturated_pixel_as_hot_pixel is True


def test_odd_roi_size_is_accepted():
    config = AnalysisConfig()
    config.instrument.zenith_deg = 12.0
    config.roi.size_px = 21
    config.roi.fallback_size_px = 31
    validate_config(config)


def test_even_roi_size_is_rejected():
    config = AnalysisConfig()
    config.instrument.zenith_deg = 12.0
    config.roi.size_px = 24
    with pytest.raises(ConfigError, match="ROI size must be an odd integer"):
        validate_config(config)


def test_invalid_roi_safety_margin_is_rejected():
    config = AnalysisConfig()
    config.instrument.zenith_deg = 12.0
    config.roi.safety_margin_px = -1
    with pytest.raises(ConfigError, match="safety_margin_px"):
        validate_config(config)


def test_invalid_roi_allowed_sizes_are_rejected():
    config = AnalysisConfig()
    config.instrument.zenith_deg = 12.0
    config.roi.min_allowed_size_px = 25
    config.roi.max_allowed_size_px = 21
    with pytest.raises(ConfigError, match="min_allowed_size_px"):
        validate_config(config)


def test_even_allowed_roi_size_is_rejected():
    config = AnalysisConfig()
    config.instrument.zenith_deg = 12.0
    config.roi.min_allowed_size_px = 20
    with pytest.raises(ConfigError, match="ROI size must be an odd integer"):
        validate_config(config)


def test_invalid_roi_safety_fraction_threshold_is_rejected():
    config = AnalysisConfig()
    config.instrument.zenith_deg = 12.0
    config.roi.unsafe_fraction_threshold = 0.10
    config.roi.warning_fraction_threshold = 0.05
    with pytest.raises(ConfigError, match="unsafe_fraction_threshold"):
        validate_config(config)
