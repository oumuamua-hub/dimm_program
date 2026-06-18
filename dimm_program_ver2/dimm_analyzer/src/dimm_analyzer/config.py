"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from .exceptions import ConfigError


class InputConfig(BaseModel):
    format: str = "ser"
    expected_source: str = "SharpCap"
    preferred_color_mode: str = "mono16"
    reject_color_ser: bool = True
    allow_camera_roi_recording: bool = True


class CameraConfig(BaseModel):
    binning: int = 1
    base_pixel_scale_arcsec_per_px: float = 0.643
    pixel_scale_arcsec_per_px: Optional[float] = None
    pixel_scale_mode: str = "base_times_binning"
    warn_if_binning_greater_than_1: bool = True


class TimingConfig(BaseModel):
    mode: str = "auto"
    fps: Optional[float] = None
    allow_variable_fps: bool = True
    use_ser_timestamps_if_available: bool = True
    fallback_time_mode: str = "frame_index"


class InstrumentConfig(BaseModel):
    aperture_diameter_m: float = 0.045
    baseline_m: float = 0.150
    wavelength_m: float = 5.0e-7
    zenith_deg: Optional[float] = None
    require_zenith_deg: bool = True
    zenith_correction: bool = True


class SpotsConfig(BaseModel):
    expected_separation_px: float = 50
    separation_tolerance_px: float = 25


class OrientationConfig(BaseModel):
    mode: str = "auto_consistency"
    mask_angle_deg: Optional[float] = None
    angle_grid_step_deg: float = 1.0
    allow_pair_angle_as_fallback: bool = True


class DetectionConfig(BaseModel):
    method: str = "DAOStarFinder"
    threshold_sigma: float = 5.0
    fwhm_guess_px: float = 5.0
    max_sources: int = 10


class ROIConfig(BaseModel):
    size_px: int = 25
    fallback_size_px: int = 31
    safety_margin_px: float = 5.0
    auto_safety_check: bool = True
    auto_shrink_if_unsafe: bool = False
    min_allowed_size_px: int = 21
    max_allowed_size_px: int = 41
    safety_reference_percentile: float = 5.0
    unsafe_fraction_threshold: float = 0.01
    warning_fraction_threshold: float = 0.05


class GaussianFitConfig(BaseModel):
    model: str = "elliptical_no_rotation"
    fit_sigma_min_px: float = 0.6
    fit_sigma_max_px: float = 8.0
    fit_fwhm_min_px: float = 1.2
    fit_fwhm_max_px: float = 18.0
    max_iterations: int = 200
    min_amplitude: float = 50
    robust_loss: str = "linear"


class QualityConfig(BaseModel):
    reject_saturated: bool = True
    saturation_level: Optional[float] = None
    saturation_margin: float = 100
    min_flux: float = 1000
    reject_spot_tracking: bool = True
    max_spot_tracking_distance_px: float = 15.0
    reject_large_jump: bool = True
    max_center_jump_px: float = 10
    reject_bad_fwhm: bool = True
    reject_fwhm_outlier: bool = True
    max_fwhm_relative_deviation: float = 0.5
    fwhm_reference: str = "running_median"
    fwhm_window_frames: int = 101
    reject_bad_separation: bool = True
    reject_bad_relative_motion: bool = True
    max_relative_motion_deviation_px: float = 10.0
    relative_motion_reference: str = "running_median"
    relative_motion_window_frames: int = 101
    saturation_core_radius_px: float = 2.0
    saturation_core_min_pixels: int = 1
    reject_saturated_core: bool = True
    classify_roi_saturated_pixel_as_hot_pixel: bool = True


class CalibrationConfig(BaseModel):
    subtract_dark: bool = False
    dark_path: Optional[Path] = None
    dark_method: str = "median"
    max_dark_frames: int = 1000


class StatisticsConfig(BaseModel):
    block_mode: str = "time"
    block_duration_sec: float = 6.0
    block_size_frames: int = 3000
    min_valid_frames_per_block: int = 500
    min_valid_fraction_per_block: float = 0.6
    representative_statistic: str = "median"


class OutputConfig(BaseModel):
    save_csv: bool = True
    save_json: bool = True
    save_plots: bool = True
    save_example_fit_images: bool = True


class AnalysisConfig(BaseModel):
    input: InputConfig = InputConfig()
    camera: CameraConfig = CameraConfig()
    timing: TimingConfig = TimingConfig()
    instrument: InstrumentConfig = InstrumentConfig()
    spots: SpotsConfig = SpotsConfig()
    orientation: OrientationConfig = OrientationConfig()
    detection: DetectionConfig = DetectionConfig()
    roi: ROIConfig = ROIConfig()
    gaussian_fit: GaussianFitConfig = GaussianFitConfig()
    quality: QualityConfig = QualityConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    statistics: StatisticsConfig = StatisticsConfig()
    output: OutputConfig = OutputConfig()


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")  # type: ignore[attr-defined]
    return model.dict()


def load_config(path: Path) -> AnalysisConfig:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    try:
        return AnalysisConfig(**data)
    except Exception as exc:  # pydantic v1/v2 expose different exception classes.
        raise ConfigError(f"{path} の設定が不正です: {exc}") from exc


def apply_cli_overrides(
    config: AnalysisConfig,
    *,
    fps: Optional[float] = None,
    dark_path: Optional[Path] = None,
    zenith_deg: Optional[float] = None,
    roi_size_px: Optional[int] = None,
    roi_safety_margin_px: Optional[float] = None,
    disable_roi_safety_check: bool = False,
    auto_shrink_roi_if_unsafe: bool = False,
    disable_large_jump_rejection: bool = False,
    disable_saturation_rejection: bool = False,
    disable_relative_motion_rejection: bool = False,
    max_relative_motion_deviation_px: Optional[float] = None,
    saturation_level: Optional[float] = None,
    saturation_margin: Optional[float] = None,
    orientation_mode: Optional[str] = None,
    orientation_angle_deg: Optional[float] = None,
) -> AnalysisConfig:
    if fps is not None:
        config.timing.fps = fps
    if dark_path is not None:
        config.calibration.dark_path = dark_path
        config.calibration.subtract_dark = True
    if zenith_deg is not None:
        config.instrument.zenith_deg = zenith_deg
    if roi_size_px is not None:
        config.roi.size_px = roi_size_px
        if config.roi.fallback_size_px < roi_size_px:
            config.roi.fallback_size_px = roi_size_px
    if roi_safety_margin_px is not None:
        config.roi.safety_margin_px = roi_safety_margin_px
    if disable_roi_safety_check and auto_shrink_roi_if_unsafe:
        raise ConfigError(
            "--disable-roi-safety-check と --auto-shrink-roi-if-unsafe は同時指定できません。"
        )
    if disable_roi_safety_check:
        config.roi.auto_safety_check = False
        config.roi.auto_shrink_if_unsafe = False
    if auto_shrink_roi_if_unsafe:
        config.roi.auto_safety_check = True
        config.roi.auto_shrink_if_unsafe = True
    if disable_large_jump_rejection:
        config.quality.reject_large_jump = False
    if disable_saturation_rejection:
        config.quality.reject_saturated = False
    if disable_relative_motion_rejection:
        config.quality.reject_bad_relative_motion = False
    if max_relative_motion_deviation_px is not None:
        config.quality.max_relative_motion_deviation_px = max_relative_motion_deviation_px
    if saturation_level is not None:
        config.quality.saturation_level = saturation_level
    if saturation_margin is not None:
        config.quality.saturation_margin = saturation_margin
    if orientation_mode is not None:
        config.orientation.mode = orientation_mode
    if orientation_angle_deg is not None:
        config.orientation.mask_angle_deg = orientation_angle_deg
        if orientation_mode is None:
            config.orientation.mode = "manual"
    return config


def effective_pixel_scale_arcsec_per_px(config: AnalysisConfig) -> float:
    camera = config.camera
    if camera.pixel_scale_arcsec_per_px is not None:
        if camera.pixel_scale_arcsec_per_px <= 0:
            raise ConfigError("camera.pixel_scale_arcsec_per_px は正の値にしてください。")
        return float(camera.pixel_scale_arcsec_per_px)
    if camera.binning <= 0:
        raise ConfigError("camera.binning は正の整数にしてください。")
    if camera.base_pixel_scale_arcsec_per_px <= 0:
        raise ConfigError("camera.base_pixel_scale_arcsec_per_px は正の値にしてください。")
    if camera.pixel_scale_mode != "base_times_binning":
        raise ConfigError(
            "pixel_scale_arcsec_per_px が null の場合、"
            "camera.pixel_scale_mode='base_times_binning' のみ対応しています。"
        )
    return float(camera.base_pixel_scale_arcsec_per_px * camera.binning)


def validate_config(config: AnalysisConfig) -> None:
    if config.input.format.lower() != "ser":
        raise ConfigError("MVP では input.format='ser' のみ対応しています。")
    if config.instrument.require_zenith_deg and config.instrument.zenith_deg is None:
        raise ConfigError(
            "zenith_deg は config または CLI で明示してください。暗黙には仮定しません。"
        )
    if config.instrument.zenith_deg is not None and not (0 <= config.instrument.zenith_deg < 90):
        raise ConfigError("instrument.zenith_deg は [0, 90) の範囲にしてください。")
    if config.instrument.aperture_diameter_m <= 0:
        raise ConfigError("instrument.aperture_diameter_m は正の値にしてください。")
    if config.instrument.baseline_m <= 0:
        raise ConfigError("instrument.baseline_m は正の値にしてください。")
    if config.instrument.wavelength_m <= 0:
        raise ConfigError("instrument.wavelength_m は正の値にしてください。")
    roi_sizes = [
        config.roi.size_px,
        config.roi.fallback_size_px,
        config.roi.min_allowed_size_px,
        config.roi.max_allowed_size_px,
    ]
    if any(size <= 0 for size in roi_sizes):
        raise ConfigError("ROI size must be a positive integer.")
    if any(size % 2 == 0 for size in roi_sizes):
        raise ConfigError("ROI size must be an odd integer.")
    if config.roi.min_allowed_size_px > config.roi.max_allowed_size_px:
        raise ConfigError("roi.min_allowed_size_px は roi.max_allowed_size_px 以下にしてください。")
    if not (
        config.roi.min_allowed_size_px
        <= config.roi.size_px
        <= config.roi.max_allowed_size_px
    ):
        raise ConfigError(
            "roi.size_px は min_allowed_size_px と max_allowed_size_px の範囲に"
            "してください。"
        )
    if not (
        config.roi.min_allowed_size_px
        <= config.roi.fallback_size_px
        <= config.roi.max_allowed_size_px
    ):
        raise ConfigError(
            "roi.fallback_size_px は min_allowed_size_px と max_allowed_size_px の範囲に"
            "してください。"
        )
    if config.roi.safety_margin_px < 0:
        raise ConfigError("roi.safety_margin_px は 0 以上にしてください。")
    if not (0 < config.roi.safety_reference_percentile <= 50):
        raise ConfigError("roi.safety_reference_percentile は (0, 50] の範囲にしてください。")
    if not (0 <= config.roi.unsafe_fraction_threshold <= 1):
        raise ConfigError("roi.unsafe_fraction_threshold は [0, 1] の範囲にしてください。")
    if not (0 <= config.roi.warning_fraction_threshold <= 1):
        raise ConfigError("roi.warning_fraction_threshold は [0, 1] の範囲にしてください。")
    if config.roi.unsafe_fraction_threshold > config.roi.warning_fraction_threshold:
        raise ConfigError(
            "roi.unsafe_fraction_threshold は roi.warning_fraction_threshold 以下にしてください。"
        )
    if config.roi.auto_shrink_if_unsafe and not config.roi.auto_safety_check:
        raise ConfigError(
            "roi.auto_shrink_if_unsafe を使う場合は roi.auto_safety_check を true にしてください。"
        )
    if config.statistics.block_duration_sec <= 0:
        raise ConfigError("statistics.block_duration_sec は正の値にしてください。")
    if config.statistics.block_size_frames <= 1:
        raise ConfigError("statistics.block_size_frames は 1 より大きくしてください。")
    if config.quality.saturation_level is not None and config.quality.saturation_level <= 0:
        raise ConfigError("quality.saturation_level は正の値にしてください。")
    if config.quality.saturation_margin < 0:
        raise ConfigError("quality.saturation_margin は 0 以上にしてください。")
    if config.quality.max_spot_tracking_distance_px <= 0:
        raise ConfigError("quality.max_spot_tracking_distance_px は正の値にしてください。")
    if config.quality.max_center_jump_px <= 0:
        raise ConfigError("quality.max_center_jump_px は正の値にしてください。")
    if config.quality.max_relative_motion_deviation_px <= 0:
        raise ConfigError("quality.max_relative_motion_deviation_px は正の値にしてください。")
    if config.quality.relative_motion_window_frames <= 0:
        raise ConfigError("quality.relative_motion_window_frames は正の整数にしてください。")
    if config.quality.fwhm_window_frames <= 0:
        raise ConfigError("quality.fwhm_window_frames は正の整数にしてください。")
    if not (0 < config.quality.max_fwhm_relative_deviation < 10):
        raise ConfigError("quality.max_fwhm_relative_deviation は 0 より大きい値にしてください。")
    if config.quality.saturation_core_radius_px <= 0:
        raise ConfigError("quality.saturation_core_radius_px は正の値にしてください。")
    if config.quality.saturation_core_min_pixels <= 0:
        raise ConfigError("quality.saturation_core_min_pixels は正の整数にしてください。")
    effective_pixel_scale_arcsec_per_px(config)


def config_warnings(config: AnalysisConfig) -> List[str]:
    warnings: List[str] = []
    if config.camera.warn_if_binning_greater_than_1 and config.camera.binning > 1:
        warnings.append(
            "camera.binning > 1: 重心精度が低下する可能性があります。"
            "有効ピクセルスケールを確認してください。"
        )
    if config.instrument.baseline_m <= 2 * config.instrument.aperture_diameter_m:
        warnings.append("DIMM 計算では通常 instrument.baseline_m は 2D より大きい必要があります。")
    if config.orientation.mode == "auto_pair":
        warnings.append(
            "orientation.auto_pair は暫定推定です。星像ペア角が物理的な開口ベースライン角と"
            "一致するとは限りません。"
        )
    return warnings
