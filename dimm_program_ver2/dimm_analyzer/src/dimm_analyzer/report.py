"""CSV and JSON report writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .models import BlockResult, FrameResult

PER_FRAME_COLUMNS = [
    "frame_index",
    "time_sec",
    "x1",
    "y1",
    "x2",
    "y2",
    "dx_px",
    "dy_px",
    "dL_px",
    "dT_px",
    "separation_px",
    "candidate1_x",
    "candidate1_y",
    "candidate2_x",
    "candidate2_y",
    "assigned_spot1_x",
    "assigned_spot1_y",
    "assigned_spot2_x",
    "assigned_spot2_y",
    "assignment_distance",
    "assignment_swapped",
    "tracking_status",
    "relative_reference_dx_px",
    "relative_reference_dy_px",
    "relative_delta_dx_px",
    "relative_delta_dy_px",
    "fwhm_reference1",
    "fwhm_reference2",
    "fit_success_spot1",
    "fit_success_spot2",
    "frame_valid",
    "reject_reason",
    "amp1",
    "bg1",
    "sigma_x1",
    "sigma_y1",
    "fwhm_x1",
    "fwhm_y1",
    "fwhm_mean1",
    "residual_rms1",
    "flux1",
    "peak1",
    "peak_raw_max1",
    "peak_core_max1",
    "saturated_core_pixel_count1",
    "saturated_roi_pixel_count1",
    "hot_pixel_or_roi_outlier1",
    "amp2",
    "bg2",
    "sigma_x2",
    "sigma_y2",
    "fwhm_x2",
    "fwhm_y2",
    "fwhm_mean2",
    "residual_rms2",
    "flux2",
    "peak2",
    "peak_raw_max2",
    "peak_core_max2",
    "saturated_core_pixel_count2",
    "saturated_roi_pixel_count2",
    "hot_pixel_or_roi_outlier2",
]

BLOCK_COLUMNS = [
    "block_index",
    "time_start_sec",
    "time_end_sec",
    "frame_start",
    "frame_end",
    "n_total",
    "n_valid",
    "frame_fit_success_rate",
    "spot_fit_success_rate",
    "var_L_px2",
    "var_T_px2",
    "var_L_rad2",
    "var_T_rad2",
    "r0_L_observed_m",
    "r0_T_observed_m",
    "r0_mean_observed_m",
    "r0_L_zenith_m",
    "r0_T_zenith_m",
    "r0_mean_zenith_m",
    "seeing_L_observed_arcsec",
    "seeing_T_observed_arcsec",
    "seeing_mean_observed_arcsec",
    "seeing_L_zenith_arcsec",
    "seeing_T_zenith_arcsec",
    "seeing_mean_zenith_arcsec",
]

ORIENTATION_SCAN_COLUMNS = [
    "angle_deg",
    "var_L_px2",
    "var_T_px2",
    "var_L_rad2",
    "var_T_rad2",
    "seeing_L_arcsec",
    "seeing_T_arcsec",
    "seeing_mean_arcsec",
    "seeing_mean_zenith_arcsec",
    "r0_mean_zenith_m",
    "mismatch_abs_log_ratio",
]

REJECTION_SUMMARY_COLUMNS = [
    "reject_reason",
    "count",
    "fraction_total",
    "fraction_of_total_frames",
]

FRAME_DISTRIBUTION_COLUMNS = [
    "group",
    "metric",
    "count",
    "mean",
    "std",
    "median",
    "p05",
    "p95",
]

COMPARISON_SUMMARY_COLUMNS = [
    "variant",
    "orientation_mode",
    "orientation_angle_deg",
    "roi_size_px",
    "large_jump_rejection",
    "seeing_zenith_arcsec",
    "r0_zenith_m",
    "valid_frames",
    "frame_fit_success_rate",
    "reject_reason_counts",
]

BATCH_SUMMARY_COLUMNS = [
    "ser_file",
    "ser_path",
    "output_dir",
    "summary_json",
    "block_results_csv",
    "per_frame_fits_csv",
    "total_frames",
    "valid_frames",
    "frame_fit_success_rate",
    "number_of_valid_blocks",
    "median_seeing_zenith_corrected_arcsec",
    "median_r0_zenith_m",
    "result_reliability",
    "saturated_frame_fraction",
    "roi_safety_status",
    "orientation_angle_deg",
]

SPOT_ASSIGNMENT_DEBUG_COLUMNS = [
    "frame_index",
    "candidate1_x",
    "candidate1_y",
    "candidate2_x",
    "candidate2_y",
    "assigned_spot1_x",
    "assigned_spot1_y",
    "assigned_spot2_x",
    "assigned_spot2_y",
    "assignment_distance",
    "assignment_swapped",
    "tracking_status",
]

RELATIVE_MOTION_OUTLIER_COLUMNS = [
    "frame_index",
    "time_sec",
    "dx_px",
    "dy_px",
    "median_dx_px",
    "median_dy_px",
    "delta_dx_px",
    "delta_dy_px",
    "threshold_px",
    "reason",
]

FWHM_OUTLIER_COLUMNS = [
    "frame_index",
    "time_sec",
    "fwhm_mean1",
    "fwhm_mean2",
    "fwhm_reference1",
    "fwhm_reference2",
    "reason",
]

ROI_SAFETY_COLUMNS = [
    "frame_index",
    "time_sec",
    "spot_id",
    "x",
    "y",
    "left_margin_px",
    "right_margin_px",
    "top_margin_px",
    "bottom_margin_px",
    "min_margin_px",
    "roi_size_px",
    "half_roi_px",
    "safety_margin_px",
    "roi_safe",
    "included_in_reliable_population",
    "below_half_roi",
    "below_required_margin",
    "suspected_outlier",
]


def write_per_frame_csv(path: Path, frames: Iterable[FrameResult]) -> None:
    _write_csv(path, PER_FRAME_COLUMNS, [frame.to_csv_dict() for frame in frames])


def write_block_csv(path: Path, blocks: Iterable[BlockResult]) -> None:
    _write_csv(path, BLOCK_COLUMNS, [block.to_csv_dict() for block in blocks])


def write_orientation_scan_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    _write_csv(path, ORIENTATION_SCAN_COLUMNS, rows)


def write_rejection_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    _write_csv(path, REJECTION_SUMMARY_COLUMNS, rows)


def write_frame_distribution_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    _write_csv(path, FRAME_DISTRIBUTION_COLUMNS, rows)


def write_spot_assignment_debug_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    _write_csv(path, SPOT_ASSIGNMENT_DEBUG_COLUMNS, rows)


def write_relative_motion_outliers_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    _write_csv(path, RELATIVE_MOTION_OUTLIER_COLUMNS, rows)


def write_fwhm_outliers_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    _write_csv(path, FWHM_OUTLIER_COLUMNS, rows)


def write_roi_safety_report_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    _write_csv(path, ROI_SAFETY_COLUMNS, rows)


def write_comparison_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    _write_csv(path, COMPARISON_SUMMARY_COLUMNS, rows)


def write_batch_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    _write_csv(path, BATCH_SUMMARY_COLUMNS, rows)


def write_batch_manifest_json(path: Path, manifest: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, default=_json_default)
        handle.write("\n")


def write_input_manifest_json(path: Path, manifest: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, default=_json_default)
        handle.write("\n")


def write_summary_json(path: Path, summary: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=_json_default)
        handle.write("\n")


def _write_csv(path: Path, columns: List[str], rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in columns})


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    return value


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} は JSON に変換できません")
