"""ROI edge-margin diagnostics."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

ROI_SAFETY_NOT_MEASURED_WARNING = "ROI safety could not be measured"


def roi_half_size(size_px: int) -> float:
    return (float(size_px) - 1.0) / 2.0


def recommended_max_roi_size(
    reference_edge_margin_px: Optional[float],
    safety_margin_px: float,
    max_allowed_size_px: int,
) -> Optional[int]:
    if reference_edge_margin_px is None:
        return None
    raw_size = 2 * math.floor(float(reference_edge_margin_px) - float(safety_margin_px)) + 1
    return _normalize_odd_size(raw_size, minimum=1, maximum=max_allowed_size_px)


def roi_safety_status(
    min_edge_margin_px: Optional[float],
    *,
    roi_size_px: int,
    safety_margin_px: float,
) -> str:
    if min_edge_margin_px is None:
        return "warning"
    half_roi_px = roi_half_size(roi_size_px)
    if min_edge_margin_px < half_roi_px:
        return "unsafe"
    if min_edge_margin_px < half_roi_px + safety_margin_px:
        return "warning"
    return "safe"


def build_roi_safety_point(
    *,
    frame_index: int,
    time_sec: Optional[float],
    spot_id: int,
    x: Optional[float],
    y: Optional[float],
    width: int,
    height: int,
) -> Optional[Dict[str, Any]]:
    if x is None or y is None:
        return None
    if not (math.isfinite(float(x)) and math.isfinite(float(y))):
        return None
    return {
        "frame_index": frame_index,
        "time_sec": time_sec,
        "spot_id": spot_id,
        "x": float(x),
        "y": float(y),
        "width": int(width),
        "height": int(height),
    }


def finalize_roi_safety_rows(
    points: Iterable[Dict[str, Any]],
    *,
    roi_size_px: int,
    safety_margin_px: float,
) -> List[Dict[str, Any]]:
    half_roi_px = roi_half_size(roi_size_px)
    required_margin_px = half_roi_px + safety_margin_px
    rows: List[Dict[str, Any]] = []
    for point in points:
        x = float(point["x"])
        y = float(point["y"])
        width = int(point["width"])
        height = int(point["height"])
        left_margin = x
        right_margin = float(width - 1) - x
        top_margin = y
        bottom_margin = float(height - 1) - y
        min_margin = min(left_margin, right_margin, top_margin, bottom_margin)
        rows.append(
            {
                "frame_index": point["frame_index"],
                "time_sec": point.get("time_sec"),
                "spot_id": point["spot_id"],
                "x": x,
                "y": y,
                "left_margin_px": left_margin,
                "right_margin_px": right_margin,
                "top_margin_px": top_margin,
                "bottom_margin_px": bottom_margin,
                "min_margin_px": min_margin,
                "roi_size_px": roi_size_px,
                "half_roi_px": half_roi_px,
                "safety_margin_px": safety_margin_px,
                "roi_safe": bool(min_margin >= required_margin_px),
                "included_in_reliable_population": bool(
                    point.get("included_in_reliable_population", True)
                ),
                "below_half_roi": bool(min_margin < half_roi_px),
                "below_required_margin": bool(min_margin < required_margin_px),
                "suspected_outlier": False,
            }
        )
    return rows


def summarize_roi_safety(
    rows: Iterable[Dict[str, Any]],
    *,
    roi_size_px: int,
    roi_fallback_size_px: int,
    safety_margin_px: float,
    max_allowed_size_px: int,
    enabled: bool,
    roi_auto_shrunk: bool = False,
    roi_original_size_px: Optional[int] = None,
    roi_original_fallback_size_px: Optional[int] = None,
    safety_reference_percentile: float = 5.0,
    unsafe_fraction_threshold: float = 0.01,
    warning_fraction_threshold: float = 0.05,
    total_frame_count: Optional[int] = None,
    roi_out_of_bounds_count: int = 0,
) -> Tuple[Dict[str, Any], List[str]]:
    row_list = list(rows)
    half_roi_px = roi_half_size(roi_size_px)
    required_margin_px = half_roi_px + safety_margin_px
    if not enabled:
        return (
            {
                "roi_size_px": roi_size_px,
                "roi_fallback_size_px": roi_fallback_size_px,
                "roi_half_size_px": half_roi_px,
                "roi_safety_margin_px": safety_margin_px,
                "roi_required_margin_px": required_margin_px,
                "min_edge_margin_px": None,
                "edge_margin_all_min_px": None,
                "edge_margin_min_frame_index": None,
                "edge_margin_min_spot_id": None,
                "edge_margin_all_p01_px": None,
                "edge_margin_all_p05_px": None,
                "edge_margin_all_median_px": None,
                "edge_margin_reliable_p01_px": None,
                "edge_margin_reliable_p05_px": None,
                "edge_margin_reliable_median_px": None,
                "recommended_max_roi_size_px_from_min": None,
                "recommended_max_roi_size_px_from_p01": None,
                "recommended_max_roi_size_px_from_p05": None,
                "recommended_max_roi_size_px": None,
                "roi_safety_status": "not_checked",
                "roi_out_of_bounds_count": int(roi_out_of_bounds_count),
                "roi_out_of_bounds_fraction": _safe_ratio(
                    roi_out_of_bounds_count,
                    total_frame_count,
                ),
                "edge_margin_below_half_roi_count": 0,
                "edge_margin_below_half_roi_fraction": 0.0,
                "edge_margin_below_required_count": 0,
                "edge_margin_below_required_fraction": 0.0,
                "edge_margin_absolute_outlier_detected": False,
                "roi_auto_shrunk": False,
                "roi_original_size_px": roi_original_size_px or roi_size_px,
                "roi_original_fallback_size_px": (
                    roi_original_fallback_size_px or roi_fallback_size_px
                ),
            },
            [],
        )

    if not row_list:
        return (
            {
                "roi_size_px": roi_size_px,
                "roi_fallback_size_px": roi_fallback_size_px,
                "roi_half_size_px": half_roi_px,
                "roi_safety_margin_px": safety_margin_px,
                "roi_required_margin_px": required_margin_px,
                "min_edge_margin_px": None,
                "edge_margin_all_min_px": None,
                "edge_margin_min_frame_index": None,
                "edge_margin_min_spot_id": None,
                "edge_margin_all_p01_px": None,
                "edge_margin_all_p05_px": None,
                "edge_margin_all_median_px": None,
                "edge_margin_reliable_p01_px": None,
                "edge_margin_reliable_p05_px": None,
                "edge_margin_reliable_median_px": None,
                "recommended_max_roi_size_px_from_min": None,
                "recommended_max_roi_size_px_from_p01": None,
                "recommended_max_roi_size_px_from_p05": None,
                "recommended_max_roi_size_px": None,
                "roi_safety_status": "warning",
                "roi_out_of_bounds_count": int(roi_out_of_bounds_count),
                "roi_out_of_bounds_fraction": _safe_ratio(
                    roi_out_of_bounds_count,
                    total_frame_count,
                ),
                "edge_margin_below_half_roi_count": 0,
                "edge_margin_below_half_roi_fraction": 0.0,
                "edge_margin_below_required_count": 0,
                "edge_margin_below_required_fraction": 0.0,
                "edge_margin_absolute_outlier_detected": False,
                "roi_auto_shrunk": roi_auto_shrunk,
                "roi_original_size_px": roi_original_size_px or roi_size_px,
                "roi_original_fallback_size_px": (
                    roi_original_fallback_size_px or roi_fallback_size_px
                ),
            },
            [ROI_SAFETY_NOT_MEASURED_WARNING],
        )

    all_values = _margin_values(row_list)
    reliable_rows = [
        row for row in row_list if bool(row.get("included_in_reliable_population", True))
    ]
    reliable_values = _margin_values(reliable_rows)
    stats_values = reliable_values if reliable_values.size else all_values

    min_row = min(row_list, key=lambda row: float(row["min_margin_px"]))
    all_min = float(min_row["min_margin_px"])
    all_p01 = _percentile(all_values, 1)
    all_p05 = _percentile(all_values, 5)
    all_median = _percentile(all_values, 50)
    reliable_p01 = _percentile(reliable_values, 1)
    reliable_p05 = _percentile(reliable_values, 5)
    reliable_median = _percentile(reliable_values, 50)
    reference_p01 = reliable_p01 if reliable_p01 is not None else all_p01
    reference_p05 = reliable_p05 if reliable_p05 is not None else all_p05

    recommended_from_min = recommended_max_roi_size(all_min, safety_margin_px, max_allowed_size_px)
    recommended_from_p01 = recommended_max_roi_size(
        reference_p01,
        safety_margin_px,
        max_allowed_size_px,
    )
    recommended_from_p05 = recommended_max_roi_size(
        reference_p05,
        safety_margin_px,
        max_allowed_size_px,
    )
    # Use robust p05 for recommendations so one edge outlier does not collapse the ROI size.
    recommended = recommended_from_p05

    below_half_count = int(np.sum(stats_values < half_roi_px))
    below_required_count = int(np.sum(stats_values < required_margin_px))
    reference_count = int(stats_values.size)
    below_half_fraction = _safe_ratio(below_half_count, reference_count)
    below_required_fraction = _safe_ratio(below_required_count, reference_count)
    roi_out_of_bounds_fraction = _safe_ratio(roi_out_of_bounds_count, total_frame_count)
    absolute_outlier = bool(all_min < half_roi_px)

    if all_p01 is not None:
        for row in row_list:
            row["suspected_outlier"] = bool(
                float(row["min_margin_px"]) <= all_p01
                or float(row["min_margin_px"]) < half_roi_px
            )

    status = _robust_safety_status(
        edge_margin_reliable_p05_px=reference_p05,
        half_roi_px=half_roi_px,
        required_margin_px=required_margin_px,
        roi_out_of_bounds_fraction=roi_out_of_bounds_fraction,
        below_half_fraction=below_half_fraction,
        below_required_fraction=below_required_fraction,
        unsafe_fraction_threshold=unsafe_fraction_threshold,
        warning_fraction_threshold=warning_fraction_threshold,
    )

    warnings: List[str] = []
    if status == "unsafe":
        warnings.append(
            "ROI safety is unsafe: robust edge-margin statistics indicate repeated edge risk."
        )
    elif status == "warning":
        warnings.append(
            "ROI safety warning: edge margin is usually acceptable, "
            "but outliers or p05 need review."
        )
    if absolute_outlier:
        warnings.append(
            "ROI edge-margin absolute outlier detected; recommended ROI uses robust p05."
        )
    return (
        {
            "roi_size_px": roi_size_px,
            "roi_fallback_size_px": roi_fallback_size_px,
            "roi_half_size_px": half_roi_px,
            "roi_safety_margin_px": safety_margin_px,
            "roi_required_margin_px": required_margin_px,
            "roi_safety_reference_percentile": safety_reference_percentile,
            "roi_unsafe_fraction_threshold": unsafe_fraction_threshold,
            "roi_warning_fraction_threshold": warning_fraction_threshold,
            "min_edge_margin_px": all_min,
            "edge_margin_all_min_px": all_min,
            "edge_margin_min_frame_index": min_row.get("frame_index"),
            "edge_margin_min_spot_id": min_row.get("spot_id"),
            "edge_margin_all_p01_px": all_p01,
            "edge_margin_all_p05_px": all_p05,
            "edge_margin_all_median_px": all_median,
            "edge_margin_reliable_p01_px": reliable_p01,
            "edge_margin_reliable_p05_px": reliable_p05,
            "edge_margin_reliable_median_px": reliable_median,
            "recommended_max_roi_size_px_from_min": recommended_from_min,
            "recommended_max_roi_size_px_from_p01": recommended_from_p01,
            "recommended_max_roi_size_px_from_p05": recommended_from_p05,
            "recommended_max_roi_size_px": recommended,
            "roi_safety_status": status,
            "roi_out_of_bounds_count": int(roi_out_of_bounds_count),
            "roi_out_of_bounds_fraction": roi_out_of_bounds_fraction,
            "edge_margin_below_half_roi_count": below_half_count,
            "edge_margin_below_half_roi_fraction": below_half_fraction,
            "edge_margin_below_required_count": below_required_count,
            "edge_margin_below_required_fraction": below_required_fraction,
            "edge_margin_absolute_outlier_detected": absolute_outlier,
            "roi_auto_shrunk": roi_auto_shrunk,
            "roi_original_size_px": roi_original_size_px or roi_size_px,
            "roi_original_fallback_size_px": roi_original_fallback_size_px
            or roi_fallback_size_px,
        },
        warnings,
    )


def shrink_roi_sizes_for_safety(
    *,
    requested_size_px: int,
    requested_fallback_size_px: int,
    recommended_size_px: Optional[int],
    min_allowed_size_px: int,
    max_allowed_size_px: int,
) -> Tuple[int, int]:
    if recommended_size_px is None:
        return requested_size_px, requested_fallback_size_px
    safe_cap = max(min_allowed_size_px, min(recommended_size_px, max_allowed_size_px))
    safe_cap = _normalize_odd_size(
        safe_cap,
        minimum=min_allowed_size_px,
        maximum=max_allowed_size_px,
    )
    size_px = _normalize_odd_size(
        min(requested_size_px, safe_cap),
        minimum=min_allowed_size_px,
        maximum=max_allowed_size_px,
    )
    fallback_size_px = _normalize_odd_size(
        min(requested_fallback_size_px, safe_cap),
        minimum=size_px,
        maximum=max_allowed_size_px,
    )
    return size_px, fallback_size_px


def _normalize_odd_size(value: int, *, minimum: int, maximum: int) -> int:
    value = int(value)
    value = max(minimum, min(value, maximum))
    if value % 2 == 0:
        value -= 1
    if value < minimum:
        value += 2
    if value > maximum:
        value -= 2
    return max(minimum, min(value, maximum))


def _margin_values(rows: Iterable[Dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            float(row["min_margin_px"])
            for row in rows
            if row.get("min_margin_px") is not None
            and math.isfinite(float(row["min_margin_px"]))
        ],
        dtype=float,
    )


def _percentile(values: np.ndarray, percentile: float) -> Optional[float]:
    if values.size == 0:
        return None
    return float(np.percentile(values, percentile))


def _safe_ratio(numerator: int, denominator: Optional[int]) -> float:
    if denominator is None or denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _robust_safety_status(
    *,
    edge_margin_reliable_p05_px: Optional[float],
    half_roi_px: float,
    required_margin_px: float,
    roi_out_of_bounds_fraction: float,
    below_half_fraction: float,
    below_required_fraction: float,
    unsafe_fraction_threshold: float,
    warning_fraction_threshold: float,
) -> str:
    if edge_margin_reliable_p05_px is None:
        return "warning"
    if (
        edge_margin_reliable_p05_px < half_roi_px
        or roi_out_of_bounds_fraction > warning_fraction_threshold
        or below_half_fraction > warning_fraction_threshold
        or below_required_fraction > warning_fraction_threshold
    ):
        return "unsafe"
    if (
        edge_margin_reliable_p05_px < required_margin_px
        or roi_out_of_bounds_fraction > unsafe_fraction_threshold
        or below_half_fraction > 0
        or below_required_fraction > 0
    ):
        return "warning"
    return "safe"
