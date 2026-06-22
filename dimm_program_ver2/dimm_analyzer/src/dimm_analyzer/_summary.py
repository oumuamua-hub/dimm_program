"""Aggregate analysis results and validate the public summary contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ._diagnostics import _finite_or_nan, _frame_metric_value
from .config import AnalysisConfig, model_to_dict
from .dimm_math import seeing_from_r0_arcsec, zenith_correction_factor
from .exceptions import PipelineError
from .models import AnalysisResult, BlockResult, FrameResult, OrientationResult, SERMetadata
from .quality import ratio


def validate_result_consistency(result: AnalysisResult) -> None:
    """Validate summary values against the tabular data sources before writing files."""

    # Frame and block tables are the source of truth for duplicated summary values.
    summary = result.summary
    total_frames = len(result.frame_results)
    valid_frames = sum(1 for frame in result.frame_results if frame.frame_valid)
    valid_blocks = _valid_blocks_with_finite_seeing(result.block_results)
    if summary.get("total_frames") != total_frames:
        raise PipelineError("summary.total_frames が per_frame_fits.csv 相当の行数と一致しません。")
    if summary.get("valid_frames") != valid_frames:
        raise PipelineError("summary.valid_frames が frame_valid=True の行数と一致しません。")
    if summary.get("number_of_valid_blocks") != len(valid_blocks):
        raise PipelineError(
            "summary.number_of_valid_blocks が finite seeing block 数と一致しません。"
        )
    if not valid_blocks:
        for key in (
            "median_seeing_zenith_corrected_arcsec",
            "seeing_zenith_arcsec",
            "mean_seeing_zenith_corrected_arcsec",
            "std_seeing_zenith_corrected_arcsec",
            "r0_zenith_m",
            "median_r0_zenith_m",
        ):
            if summary.get(key) is not None:
                raise PipelineError(f"valid block が0なのに summary.{key} が null ではありません。")
        return

    expected = {
        "median_seeing_zenith_corrected_arcsec": _nanmedian(
            [block.seeing_mean_zenith_arcsec for block in valid_blocks]
        ),
        "seeing_zenith_arcsec": _nanmedian(
            [block.seeing_mean_zenith_arcsec for block in valid_blocks]
        ),
        "median_r0_zenith_m": _nanmedian([block.r0_mean_zenith_m for block in valid_blocks]),
        "r0_zenith_m": _nanmedian([block.r0_mean_zenith_m for block in valid_blocks]),
    }
    for key, value in expected.items():
        if not _optional_close(summary.get(key), value):
            raise PipelineError(f"summary.{key} が valid block から再計算した値と一致しません。")


def _saturation_summary(
    *,
    frame_results: Sequence[FrameResult],
    saturation_level: Optional[float],
    saturation_margin: float,
    saturation_core_min_pixels: int,
) -> Dict[str, Any]:
    threshold = saturation_level - saturation_margin if saturation_level is not None else None
    peaks1 = np.asarray(
        [_finite_or_nan(frame.fit1.peak) for frame in frame_results],
        dtype=float,
    )
    peaks2 = np.asarray(
        [_finite_or_nan(frame.fit2.peak) for frame in frame_results],
        dtype=float,
    )
    peak_raw_max1 = np.asarray(
        [_finite_or_nan(frame.fit1.peak_raw_max) for frame in frame_results],
        dtype=float,
    )
    peak_raw_max2 = np.asarray(
        [_finite_or_nan(frame.fit2.peak_raw_max) for frame in frame_results],
        dtype=float,
    )
    peak_core_max1 = np.asarray(
        [_finite_or_nan(frame.fit1.peak_core_max) for frame in frame_results],
        dtype=float,
    )
    peak_core_max2 = np.asarray(
        [_finite_or_nan(frame.fit2.peak_core_max) for frame in frame_results],
        dtype=float,
    )
    saturated_core_count = 0
    saturated_roi_pixel_outlier_count = 0
    # Core saturation takes precedence; edge-only pixels remain a separate diagnostic class.
    for frame in frame_results:
        core1 = frame.fit1.saturated_core_pixel_count or 0
        core2 = frame.fit2.saturated_core_pixel_count or 0
        if core1 >= saturation_core_min_pixels or core2 >= saturation_core_min_pixels:
            saturated_core_count += 1
        elif frame.fit1.hot_pixel_or_roi_outlier or frame.fit2.hot_pixel_or_roi_outlier:
            saturated_roi_pixel_outlier_count += 1
    return {
        "saturation_level": saturation_level,
        "saturation_margin": saturation_margin,
        "saturation_threshold": threshold,
        "saturated_frame_count": saturated_core_count,
        "saturated_frame_fraction": ratio(saturated_core_count, len(frame_results)),
        "saturated_core_frame_count": saturated_core_count,
        "saturated_core_frame_fraction": ratio(saturated_core_count, len(frame_results)),
        "saturated_roi_pixel_outlier_count": saturated_roi_pixel_outlier_count,
        "peak_raw_max1": _array_stat(peak_raw_max1, "max"),
        "peak_raw_max2": _array_stat(peak_raw_max2, "max"),
        "peak_core_max1": _array_stat(peak_core_max1, "max"),
        "peak_core_max2": _array_stat(peak_core_max2, "max"),
        "peak1_median": _array_stat(peaks1, "median"),
        "peak2_median": _array_stat(peaks2, "median"),
        "peak1_p95": _array_stat(peaks1, "p95"),
        "peak2_p95": _array_stat(peaks2, "p95"),
        "peak1_max": _array_stat(peaks1, "max"),
        "peak2_max": _array_stat(peaks2, "max"),
    }


def _valid_rejected_medians(frame_results: Sequence[FrameResult]) -> Dict[str, Any]:
    groups = {
        "valid": [frame for frame in frame_results if frame.frame_valid],
        "rejected": [frame for frame in frame_results if not frame.frame_valid],
    }
    output: Dict[str, Any] = {}
    metrics = {
        "peak1": "peak1",
        "peak2": "peak2",
        "fwhm1": "fwhm_mean1",
        "fwhm2": "fwhm_mean2",
        "separation": "separation_px",
    }
    for group_name, frames in groups.items():
        for output_name, metric in metrics.items():
            values = np.asarray(
                [
                    value
                    for value in (_frame_metric_value(frame, metric) for frame in frames)
                    if value is not None and np.isfinite(value)
                ],
                dtype=float,
            )
            output[f"{group_name}_{output_name}_median"] = (
                float(np.median(values)) if values.size else None
            )
    return output


def _orientation_mismatch_score(rows: Sequence[Dict[str, Any]]) -> Optional[float]:
    values = [
        row.get("mismatch_abs_log_ratio")
        for row in rows
        if row.get("mismatch_abs_log_ratio") is not None
    ]
    if not values:
        return None
    return float(min(values))


def _result_reliability(
    *,
    valid_block_count: int,
    frame_fit_success_rate: float,
    saturated_frame_fraction: float,
    orientation_reliable: bool,
    relative_motion_outlier_count: int,
    roi_safety_status: str,
    total_frames: int,
) -> str:
    relative_bad = relative_motion_outlier_count > max(5, int(0.01 * total_frames))
    if roi_safety_status == "unsafe":
        return "bad"
    if (
        valid_block_count < 3
        or frame_fit_success_rate < 0.7
        or saturated_frame_fraction > 0.05
        or relative_bad
    ):
        return "bad"
    if (
        valid_block_count >= 5
        and frame_fit_success_rate >= 0.8
        and saturated_frame_fraction < 0.01
        and orientation_reliable
        and roi_safety_status != "warning"
    ):
        return "good"
    return "caution"


def _array_stat(values: np.ndarray, stat: str) -> Optional[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    if stat == "median":
        return float(np.median(finite))
    if stat == "p95":
        return float(np.percentile(finite, 95))
    if stat == "max":
        return float(np.max(finite))
    raise ValueError(f"unknown stat: {stat}")


def _valid_blocks_with_finite_seeing(blocks: Sequence[BlockResult]) -> List[BlockResult]:
    return [
        block
        for block in blocks
        if block.valid and _is_finite_number(block.seeing_mean_zenith_arcsec)
    ]


def _is_finite_number(value: Optional[float]) -> bool:
    return value is not None and bool(np.isfinite(value))


def _optional_close(left: Optional[float], right: Optional[float], *, atol: float = 1e-12) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return bool(np.isfinite(left) and np.isfinite(right) and np.isclose(left, right, atol=atol))


def _representative_block_values(valid_blocks: Sequence[BlockResult]) -> Dict[str, Optional[float]]:
    return {
        "seeing_observed_arcsec": _nanmedian(
            [block.seeing_mean_observed_arcsec for block in valid_blocks]
        ),
        "seeing_zenith_arcsec": _nanmedian(
            [block.seeing_mean_zenith_arcsec for block in valid_blocks]
        ),
        "r0_observed_m": _nanmedian([block.r0_mean_observed_m for block in valid_blocks]),
        "r0_zenith_m": _nanmedian([block.r0_mean_zenith_m for block in valid_blocks]),
        "var_L_px2": _nanmedian([block.var_L_px2 for block in valid_blocks]),
        "var_T_px2": _nanmedian([block.var_T_px2 for block in valid_blocks]),
        "var_L_rad2": _nanmedian([block.var_L_rad2 for block in valid_blocks]),
        "var_T_rad2": _nanmedian([block.var_T_rad2 for block in valid_blocks]),
        "seeing_L_observed_arcsec": _nanmedian(
            [block.seeing_L_observed_arcsec for block in valid_blocks]
        ),
        "seeing_T_observed_arcsec": _nanmedian(
            [block.seeing_T_observed_arcsec for block in valid_blocks]
        ),
        "seeing_L_zenith_arcsec": _nanmedian(
            [block.seeing_L_zenith_arcsec for block in valid_blocks]
        ),
        "seeing_T_zenith_arcsec": _nanmedian(
            [block.seeing_T_zenith_arcsec for block in valid_blocks]
        ),
    }


def _sanity_check_from_r0(
    *,
    r0_zenith_m: Optional[float],
    seeing_zenith_arcsec: Optional[float],
    wavelength_m: float,
) -> Dict[str, Any]:
    if r0_zenith_m is None:
        return {
            "formula": "seeing_arcsec = 0.98 * wavelength_m / r0_m * 206265",
            "r0_zenith_m": None,
            "wavelength_m": wavelength_m,
            "seeing_from_r0_arcsec": None,
            "reported_seeing_zenith_arcsec": seeing_zenith_arcsec,
            "absolute_difference_arcsec": None,
        }
    seeing_from_r0 = seeing_from_r0_arcsec(r0_zenith_m, wavelength_m)
    return {
        "formula": "seeing_arcsec = 0.98 * wavelength_m / r0_m * 206265",
        "r0_zenith_m": r0_zenith_m,
        "wavelength_m": wavelength_m,
        "seeing_from_r0_arcsec": seeing_from_r0,
        "reported_seeing_zenith_arcsec": seeing_zenith_arcsec,
        "absolute_difference_arcsec": abs(seeing_from_r0 - seeing_zenith_arcsec)
        if seeing_zenith_arcsec is not None
        else None,
    }


def build_summary(
    *,
    frame_results: Sequence[FrameResult],
    block_results: Sequence[BlockResult],
    config: AnalysisConfig,
    pixel_scale_arcsec_per_px: float,
    orientation: OrientationResult,
    warnings_out: Sequence[str],
    input_path: Optional[Path],
    ser_metadata: Optional[SERMetadata],
    time_source: str,
    fps: Optional[float],
    saturation_level: Optional[float],
    reject_reason_counts: Dict[str, int],
    frame_distribution_rows: Sequence[Dict[str, Any]],
    orientation_scan_rows: Sequence[Dict[str, Any]],
    roi_safety_summary: Dict[str, Any],
    input_path_original: Optional[Path] = None,
    input_path_resolved_ser: Optional[Path] = None,
    input_was_directory: bool = False,
    ser_selection_mode: Optional[str] = None,
    companion_files_copied: Optional[List[str]] = None,
    output_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    auto_output_name: Optional[bool] = None,
    input_manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    total_frames = len(frame_results)
    valid_frames = sum(1 for frame in frame_results if frame.frame_valid)
    fit_attempts = sum(frame.fit_attempts for frame in frame_results)
    spot_successes = sum(
        int(frame.fit1.success) + int(frame.fit2.success) for frame in frame_results
    )
    valid_blocks = _valid_blocks_with_finite_seeing(block_results)
    seeing_values = np.asarray(
        [
            block.seeing_mean_zenith_arcsec
            for block in valid_blocks
            if _is_finite_number(block.seeing_mean_zenith_arcsec)
        ],
        dtype=float,
    )
    r0_values = np.asarray(
        [
            block.r0_mean_zenith_m
            for block in valid_blocks
            if _is_finite_number(block.r0_mean_zenith_m)
        ],
        dtype=float,
    )
    representative = _representative_block_values(valid_blocks)
    sanity_check = _sanity_check_from_r0(
        r0_zenith_m=representative.get("r0_zenith_m"),
        seeing_zenith_arcsec=representative.get("seeing_zenith_arcsec"),
        wavelength_m=config.instrument.wavelength_m,
    )
    zenith_factor = (
        zenith_correction_factor(config.instrument.zenith_deg)
        if config.instrument.zenith_deg is not None
        else None
    )
    saturation_stats = _saturation_summary(
        frame_results=frame_results,
        saturation_level=saturation_level,
        saturation_margin=config.quality.saturation_margin,
        saturation_core_min_pixels=config.quality.saturation_core_min_pixels,
    )
    relative_motion_count = int(reject_reason_counts.get("bad_relative_motion", 0))
    orientation_valid_block_count = len(valid_blocks)
    orientation_mismatch_score = _orientation_mismatch_score(orientation_scan_rows)
    orientation_reliable = (
        orientation.angle_deg is not None
        and orientation_valid_block_count >= 3
        and not orientation.fallback_used
    )
    warnings = list(dict.fromkeys(warnings_out))
    if orientation_valid_block_count < 3:
        warnings.append(
            "valid block が 3 未満です。orientation と seeing は参考値として扱ってください。"
        )
    elif orientation_valid_block_count < 5:
        warnings.append(
            "valid block が 5 未満です。auto_consistency orientation の安定性を確認してください。"
        )
    quality_flags = {
        "too_few_valid_blocks": orientation_valid_block_count < 3,
        "low_fit_success_rate": ratio(valid_frames, total_frames) < 0.75,
        "high_saturation_fraction": saturation_stats["saturated_frame_fraction"] > 0.05,
        "orientation_unreliable": not orientation_reliable,
        "relative_motion_outliers_detected": relative_motion_count > 0,
        "roi_too_close_to_edge": roi_safety_summary.get("roi_safety_status") == "unsafe",
        "roi_safety_warning": roi_safety_summary.get("roi_safety_status") == "warning",
        "roi_auto_shrunk": bool(roi_safety_summary.get("roi_auto_shrunk")),
    }
    result_reliability = _result_reliability(
        valid_block_count=orientation_valid_block_count,
        frame_fit_success_rate=ratio(valid_frames, total_frames),
        saturated_frame_fraction=saturation_stats["saturated_frame_fraction"],
        orientation_reliable=orientation_reliable,
        relative_motion_outlier_count=relative_motion_count,
        roi_safety_status=str(roi_safety_summary.get("roi_safety_status")),
        total_frames=total_frames,
    )
    valid_rejected_stats = _valid_rejected_medians(frame_results)
    resolved_ser = input_path_resolved_ser if input_path_resolved_ser is not None else input_path
    original_input = input_path_original if input_path_original is not None else input_path
    output_root_value = output_root if output_root is not None else output_dir
    return {
        "input_file": str(input_path) if input_path is not None else None,
        "input_path_original": str(original_input) if original_input is not None else None,
        "input_path_resolved_ser": str(resolved_ser) if resolved_ser is not None else None,
        "input_was_directory": bool(input_was_directory),
        "ser_selection_mode": ser_selection_mode,
        "companion_files_copied": list(companion_files_copied or []),
        "input_ser_path": str(resolved_ser) if resolved_ser is not None else None,
        "input_ser_filename": resolved_ser.name if resolved_ser is not None else None,
        "input_ser_stem": resolved_ser.stem if resolved_ser is not None else None,
        "output_root": str(output_root_value) if output_root_value is not None else None,
        "output_dir": str(output_dir) if output_dir is not None else None,
        "auto_output_name": auto_output_name,
        "input_manifest_path": str(input_manifest_path)
        if input_manifest_path is not None
        else None,
        "ser_metadata": ser_metadata.to_dict() if ser_metadata is not None else None,
        "config_used": model_to_dict(config),
        "effective_pixel_scale_arcsec_per_px": pixel_scale_arcsec_per_px,
        "binning": config.camera.binning,
        "roi_size_px": config.roi.size_px,
        "roi_fallback_size_px": config.roi.fallback_size_px,
        **roi_safety_summary,
        "zenith_deg": config.instrument.zenith_deg,
        "zenith_correction_factor": zenith_factor,
        "time_source": time_source,
        "fps": fps,
        "estimated_fps": _estimated_fps(frame_results, ser_metadata, fps),
        "total_frames": total_frames,
        "valid_frames": valid_frames,
        "frame_fit_success_rate": ratio(valid_frames, total_frames),
        "spot_fit_success_rate": ratio(spot_successes, fit_attempts),
        "orientation_mode": orientation.mode,
        "orientation_angle_deg": orientation.angle_deg,
        "estimated_mask_angle_deg": orientation.angle_deg,
        "orientation_confidence": orientation.confidence,
        "orientation_reliable": orientation_reliable,
        "orientation_valid_block_count": orientation_valid_block_count,
        "orientation_mismatch_score": orientation_mismatch_score,
        "number_of_valid_blocks": len(valid_blocks),
        "median_seeing_observed_arcsec": _nanmedian(
            [block.seeing_mean_observed_arcsec for block in valid_blocks]
        ),
        "median_seeing_zenith_corrected_arcsec": _nanmedian(seeing_values),
        "seeing_observed_arcsec": representative.get("seeing_observed_arcsec"),
        "seeing_zenith_arcsec": representative.get("seeing_zenith_arcsec"),
        "mean_seeing_zenith_corrected_arcsec": _nanmean(seeing_values),
        "std_seeing_zenith_corrected_arcsec": _nanstd(seeing_values),
        "r0_observed_m": representative.get("r0_observed_m"),
        "r0_zenith_m": representative.get("r0_zenith_m"),
        "median_r0_zenith_m": _nanmedian(r0_values),
        "var_L_px2": representative.get("var_L_px2"),
        "var_T_px2": representative.get("var_T_px2"),
        "var_L_rad2": representative.get("var_L_rad2"),
        "var_T_rad2": representative.get("var_T_rad2"),
        "seeing_L_observed_arcsec": representative.get("seeing_L_observed_arcsec"),
        "seeing_T_observed_arcsec": representative.get("seeing_T_observed_arcsec"),
        "seeing_L_zenith_arcsec": representative.get("seeing_L_zenith_arcsec"),
        "seeing_T_zenith_arcsec": representative.get("seeing_T_zenith_arcsec"),
        **saturation_stats,
        **valid_rejected_stats,
        "reject_reason_counts": reject_reason_counts,
        "frame_distribution_summary": list(frame_distribution_rows),
        "quality_flags": quality_flags,
        "result_reliability": result_reliability,
        "sanity_check": sanity_check,
        "warnings": list(dict.fromkeys(warnings)),
    }


def _estimated_fps(
    frame_results: Sequence[FrameResult],
    ser_metadata: Optional[SERMetadata],
    fps: Optional[float],
) -> Optional[float]:
    if ser_metadata is not None and ser_metadata.estimated_fps is not None:
        return ser_metadata.estimated_fps
    if fps is not None:
        return fps
    timed = [frame for frame in frame_results if frame.time_sec is not None]
    if len(timed) < 2:
        return None
    elapsed = timed[-1].time_sec - timed[0].time_sec  # type: ignore[operator]
    if elapsed is None or elapsed <= 0:
        return None
    return (len(timed) - 1) / elapsed


def _nanmedian(values) -> Optional[float]:  # type: ignore[no-untyped-def]
    array = np.asarray([value for value in values if value is not None], dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return float(np.nanmedian(array))


def _nanmean(values) -> Optional[float]:  # type: ignore[no-untyped-def]
    array = np.asarray([value for value in values if value is not None], dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return float(np.nanmean(array))


def _nanstd(values) -> Optional[float]:  # type: ignore[no-untyped-def]
    array = np.asarray([value for value in values if value is not None], dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return float(np.nanstd(array))
