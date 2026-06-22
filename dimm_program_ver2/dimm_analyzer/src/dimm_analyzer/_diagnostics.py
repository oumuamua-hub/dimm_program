"""Post-fit filtering and diagnostic table construction."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import AnalysisConfig
from .dimm_math import compute_dimm_block
from .models import FrameResult
from .orientation import rotate_motion
from .quality import ratio
from .roi_safety import build_roi_safety_point

REJECT_REASON_ORDER = [
    "not_enough_spots",
    "bad_separation",
    "roi_out_of_bounds",
    "fit_failed_spot1",
    "fit_failed_spot2",
    "saturated",
    "hot_pixel_or_roi_outlier",
    "low_flux",
    "bad_fwhm",
    "fwhm_outlier",
    "large_jump",
    "spot_tracking_failed",
    "bad_relative_motion",
    "nan_result",
    "unknown_error",
]

FRAME_DIAGNOSTIC_METRICS = [
    "separation_px",
    "fwhm_mean1",
    "fwhm_mean2",
    "peak1",
    "peak2",
    "flux1",
    "flux2",
]


def apply_fwhm_outlier_filter(
    frame_results: Sequence[FrameResult],
    config: AnalysisConfig,
) -> List[Dict[str, Any]]:
    if not config.quality.reject_fwhm_outlier:
        return []
    valid = [frame for frame in frame_results if frame.frame_valid]
    fwhm1 = np.asarray([_finite_or_nan(frame.fit1.fwhm_mean) for frame in valid], dtype=float)
    fwhm2 = np.asarray([_finite_or_nan(frame.fit2.fwhm_mean) for frame in valid], dtype=float)
    ref1 = _reference_series(
        fwhm1,
        config.quality.fwhm_reference,
        config.quality.fwhm_window_frames,
    )
    ref2 = _reference_series(
        fwhm2,
        config.quality.fwhm_reference,
        config.quality.fwhm_window_frames,
    )
    rows: List[Dict[str, Any]] = []
    max_deviation = config.quality.max_fwhm_relative_deviation
    for idx, frame in enumerate(valid):
        frame.fwhm_reference1 = _none_if_nan(ref1[idx])
        frame.fwhm_reference2 = _none_if_nan(ref2[idx])
        value1 = fwhm1[idx]
        value2 = fwhm2[idx]
        bad1 = _is_relative_outlier(value1, ref1[idx], max_deviation)
        bad2 = _is_relative_outlier(value2, ref2[idx], max_deviation)
        if not (bad1 or bad2):
            continue
        rows.append(
            {
                "frame_index": frame.frame_index,
                "time_sec": frame.time_sec,
                "fwhm_mean1": _none_if_nan(value1),
                "fwhm_mean2": _none_if_nan(value2),
                "fwhm_reference1": frame.fwhm_reference1,
                "fwhm_reference2": frame.fwhm_reference2,
                "reason": "fwhm_outlier",
            }
        )
        frame.frame_valid = False
        frame.reject_reason = "fwhm_outlier"
    return rows


def apply_relative_motion_filter(
    frame_results: Sequence[FrameResult],
    config: AnalysisConfig,
) -> List[Dict[str, Any]]:
    if not config.quality.reject_bad_relative_motion:
        return []
    valid = [frame for frame in frame_results if frame.frame_valid]
    dx = np.asarray([_finite_or_nan(frame.dx_px) for frame in valid], dtype=float)
    dy = np.asarray([_finite_or_nan(frame.dy_px) for frame in valid], dtype=float)
    ref_dx = _reference_series(
        dx,
        config.quality.relative_motion_reference,
        config.quality.relative_motion_window_frames,
    )
    ref_dy = _reference_series(
        dy,
        config.quality.relative_motion_reference,
        config.quality.relative_motion_window_frames,
    )
    threshold = config.quality.max_relative_motion_deviation_px
    rows: List[Dict[str, Any]] = []
    for idx, frame in enumerate(valid):
        delta_dx = (
            abs(dx[idx] - ref_dx[idx])
            if np.isfinite(dx[idx]) and np.isfinite(ref_dx[idx])
            else np.nan
        )
        delta_dy = (
            abs(dy[idx] - ref_dy[idx])
            if np.isfinite(dy[idx]) and np.isfinite(ref_dy[idx])
            else np.nan
        )
        frame.relative_reference_dx_px = _none_if_nan(ref_dx[idx])
        frame.relative_reference_dy_px = _none_if_nan(ref_dy[idx])
        frame.relative_delta_dx_px = _none_if_nan(delta_dx)
        frame.relative_delta_dy_px = _none_if_nan(delta_dy)
        if not (
            (np.isfinite(delta_dx) and delta_dx > threshold)
            or (np.isfinite(delta_dy) and delta_dy > threshold)
        ):
            continue
        rows.append(
            {
                "frame_index": frame.frame_index,
                "time_sec": frame.time_sec,
                "dx_px": _none_if_nan(dx[idx]),
                "dy_px": _none_if_nan(dy[idx]),
                "median_dx_px": frame.relative_reference_dx_px,
                "median_dy_px": frame.relative_reference_dy_px,
                "delta_dx_px": frame.relative_delta_dx_px,
                "delta_dy_px": frame.relative_delta_dy_px,
                "threshold_px": threshold,
                "reason": "bad_relative_motion",
            }
        )
        frame.frame_valid = False
        frame.reject_reason = "bad_relative_motion"
    return rows


def build_spot_assignment_rows(frame_results: Sequence[FrameResult]) -> List[Dict[str, Any]]:
    return [
        {
            "frame_index": frame.frame_index,
            "candidate1_x": frame.candidate1_x,
            "candidate1_y": frame.candidate1_y,
            "candidate2_x": frame.candidate2_x,
            "candidate2_y": frame.candidate2_y,
            "assigned_spot1_x": frame.assigned_spot1_x,
            "assigned_spot1_y": frame.assigned_spot1_y,
            "assigned_spot2_x": frame.assigned_spot2_x,
            "assigned_spot2_y": frame.assigned_spot2_y,
            "assignment_distance": frame.assignment_distance,
            "assignment_swapped": frame.assignment_swapped,
            "tracking_status": frame.tracking_status,
        }
        for frame in frame_results
        if frame.candidate1_x is not None or frame.tracking_status
    ]


def build_roi_safety_points_from_frame_results(
    frame_results: Sequence[FrameResult],
) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for frame in frame_results:
        if frame.frame_width is None or frame.frame_height is None:
            continue
        reliable = _frame_in_reliable_roi_population(frame)
        spot_positions = [
            _frame_spot_position(frame, 1),
            _frame_spot_position(frame, 2),
        ]
        for spot_id, (x, y) in enumerate(spot_positions, start=1):
            point = build_roi_safety_point(
                frame_index=frame.frame_index,
                time_sec=frame.time_sec,
                spot_id=spot_id,
                x=x,
                y=y,
                width=frame.frame_width,
                height=frame.frame_height,
            )
            if point is not None:
                point["included_in_reliable_population"] = reliable
                points.append(point)
    return points


def _frame_in_reliable_roi_population(frame: FrameResult) -> bool:
    excluded_reasons = {"spot_tracking_failed", "bad_relative_motion", "nan_result"}
    return bool(
        frame.candidate1_x is not None
        and frame.candidate2_x is not None
        and frame.fit1.success
        and frame.fit2.success
        and frame.reject_reason not in excluded_reasons
    )


def _frame_spot_position(
    frame: FrameResult,
    spot_id: int,
) -> Tuple[Optional[float], Optional[float]]:
    if spot_id == 1:
        if frame.x1 is not None and frame.y1 is not None:
            return frame.x1, frame.y1
        if frame.assigned_spot1_x is not None and frame.assigned_spot1_y is not None:
            return frame.assigned_spot1_x, frame.assigned_spot1_y
        return frame.candidate1_x, frame.candidate1_y
    if frame.x2 is not None and frame.y2 is not None:
        return frame.x2, frame.y2
    if frame.assigned_spot2_x is not None and frame.assigned_spot2_y is not None:
        return frame.assigned_spot2_x, frame.assigned_spot2_y
    return frame.candidate2_x, frame.candidate2_y


def build_orientation_scan_rows(
    *,
    dx: np.ndarray,
    dy: np.ndarray,
    config: AnalysisConfig,
    pixel_scale_arcsec_per_px: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    step = (
        config.orientation.angle_grid_step_deg
        if config.orientation.angle_grid_step_deg > 0
        else 1.0
    )
    angles = np.arange(0.0, 180.0 + step * 0.5, step)
    for angle in angles:
        row: Dict[str, Any] = {"angle_deg": float(angle)}
        if len(dx) < 2:
            rows.append(row)
            continue
        d_l, d_t = rotate_motion(dx, dy, float(angle))
        var_l = float(np.var(d_l, ddof=1))
        var_t = float(np.var(d_t, ddof=1))
        row["var_L_px2"] = var_l
        row["var_T_px2"] = var_t
        try:
            values = compute_dimm_block(
                var_L_px2=var_l,
                var_T_px2=var_t,
                pixel_scale_arcsec_per_px=pixel_scale_arcsec_per_px,
                aperture_diameter_m=config.instrument.aperture_diameter_m,
                baseline_m=config.instrument.baseline_m,
                wavelength_m=config.instrument.wavelength_m,
                zenith_deg=config.instrument.zenith_deg,
                zenith_correction=config.instrument.zenith_correction,
            )
        except Exception:
            rows.append(row)
            continue
        seeing_l = values["seeing_L_observed_arcsec"]
        seeing_t = values["seeing_T_observed_arcsec"]
        row.update(
            {
                "var_L_rad2": values["var_L_rad2"],
                "var_T_rad2": values["var_T_rad2"],
                "seeing_L_arcsec": seeing_l,
                "seeing_T_arcsec": seeing_t,
                "seeing_mean_arcsec": values["seeing_mean_observed_arcsec"],
                "seeing_mean_zenith_arcsec": values["seeing_mean_zenith_arcsec"],
                "r0_mean_zenith_m": values["r0_mean_zenith_m"],
                "mismatch_abs_log_ratio": abs(math.log(seeing_l / seeing_t))
                if seeing_l > 0 and seeing_t > 0
                else None,
            }
        )
        rows.append(row)
    return rows


def reject_reason_counts(frame_results: Sequence[FrameResult]) -> Dict[str, int]:
    counts = Counter(frame.reject_reason for frame in frame_results if not frame.frame_valid)
    return {reason: int(counts.get(reason, 0)) for reason in REJECT_REASON_ORDER}


def build_rejection_summary_rows(frame_results: Sequence[FrameResult]) -> List[Dict[str, Any]]:
    counts = reject_reason_counts(frame_results)
    total = len(frame_results)
    return [
        {
            "reject_reason": reason,
            "count": count,
            "fraction_total": ratio(count, total),
            "fraction_of_total_frames": ratio(count, total),
        }
        for reason, count in counts.items()
    ]


def build_frame_distribution_rows(frame_results: Sequence[FrameResult]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    groups = {
        "valid": [frame for frame in frame_results if frame.frame_valid],
        "rejected": [frame for frame in frame_results if not frame.frame_valid],
    }
    for group_name, frames in groups.items():
        for metric in FRAME_DIAGNOSTIC_METRICS:
            values = np.asarray(
                [
                    value
                    for value in (_frame_metric_value(frame, metric) for frame in frames)
                    if value is not None and np.isfinite(value)
                ],
                dtype=float,
            )
            rows.append(_distribution_row(group_name, metric, values))
    return rows


def _frame_metric_value(frame: FrameResult, metric: str) -> Optional[float]:
    if metric == "separation_px":
        return frame.separation_px
    if metric.endswith("1"):
        fit = frame.fit1
        attr = metric[:-1]
    elif metric.endswith("2"):
        fit = frame.fit2
        attr = metric[:-1]
    else:
        return None
    return getattr(fit, attr, None)


def _distribution_row(group: str, metric: str, values: np.ndarray) -> Dict[str, Any]:
    if values.size == 0:
        return {
            "group": group,
            "metric": metric,
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "p05": None,
            "p95": None,
        }
    return {
        "group": group,
        "metric": metric,
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def _reference_series(values: np.ndarray, mode: str, window_frames: int) -> np.ndarray:
    refs = np.full(values.shape, np.nan, dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return refs
    if mode == "global_median":
        refs[:] = float(np.median(finite_values))
        return refs

    window = max(1, int(window_frames))
    half = window // 2
    for idx in range(values.size):
        start = max(0, idx - half)
        stop = min(values.size, idx + half + 1)
        local = values[start:stop]
        local = local[np.isfinite(local)]
        if local.size == 0:
            refs[idx] = float(np.median(finite_values))
        else:
            refs[idx] = float(np.median(local))
    return refs


def _is_relative_outlier(value: float, reference: float, max_relative_deviation: float) -> bool:
    if not (np.isfinite(value) and np.isfinite(reference) and reference > 0):
        return False
    return bool(abs(value - reference) > abs(reference) * max_relative_deviation)


def _finite_or_nan(value: Optional[float]) -> float:
    if value is None:
        return float("nan")
    return float(value) if np.isfinite(value) else float("nan")


def _none_if_nan(value: float) -> Optional[float]:
    return float(value) if np.isfinite(value) else None
