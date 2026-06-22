"""Turn one image frame into a fitted and quality-classified frame result."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import AnalysisConfig
from .detection import detect_sources, extract_roi, select_spot_pair
from .gaussian_fit import fit_gaussian_2d
from .models import FitResult, FrameResult
from .quality import frame_reject_reason_for_fit_fail

PreviousCenters = Tuple[float, float, float, float]
ProcessFrameReturn = Tuple[
    FrameResult,
    Optional[PreviousCenters],
    Optional[np.ndarray],
    Optional[np.ndarray],
]


def _process_frame(
    *,
    frame_index: int,
    time_sec: Optional[float],
    frame: np.ndarray,
    config: AnalysisConfig,
    saturation_level: Optional[float],
    previous_centers: Optional[PreviousCenters],
) -> ProcessFrameReturn:
    height, width = frame.shape[:2]
    result = FrameResult(
        frame_index=frame_index,
        time_sec=time_sec,
        frame_width=int(width),
        frame_height=int(height),
    )
    success_roi: Optional[np.ndarray] = None
    failed_roi: Optional[np.ndarray] = None
    try:
        sources = detect_sources(frame, config.detection)
        pair = select_spot_pair(
            sources,
            expected_separation_px=config.spots.expected_separation_px,
            separation_tolerance_px=config.spots.separation_tolerance_px,
        )
        # Early returns below encode rejection precedence; keep their order stable.
        if pair is None:
            result.reject_reason = "not_enough_spots" if len(sources) < 2 else "bad_separation"
            return result, previous_centers, None, None
        result.candidate1_x = pair[0].x
        result.candidate1_y = pair[0].y
        result.candidate2_x = pair[1].x
        result.candidate2_y = pair[1].y

        fit1, attempts1, roi1, failed1 = _fit_source(pair[0], frame, config, saturation_level)
        fit2, attempts2, roi2, failed2 = _fit_source(pair[1], frame, config, saturation_level)
        result.fit1 = fit1
        result.fit2 = fit2
        result.fit_attempts = attempts1 + attempts2
        if fit1.success and roi1 is not None:
            success_roi = roi1
        if failed1 is not None:
            failed_roi = failed1
        elif failed2 is not None:
            failed_roi = failed2

        if fit1.failure_reason == "roi_out_of_bounds" or fit2.failure_reason == "roi_out_of_bounds":
            result.reject_reason = "roi_out_of_bounds"
            return result, previous_centers, success_roi, failed_roi
        if not (fit1.success and fit2.success):
            result.reject_reason = frame_reject_reason_for_fit_fail(fit1, fit2)
            return result, previous_centers, success_roi, failed_roi

        if not _assign_spot_ids(result, previous_centers, config):
            result.set_from_fits()
            result.reject_reason = "spot_tracking_failed"
            return result, previous_centers, success_roi, failed_roi
        result.set_from_fits()
        if result.separation_px is None:
            result.reject_reason = "nan_result"
            return result, previous_centers, success_roi, failed_roi
        if config.quality.reject_bad_separation:
            delta = abs(result.separation_px - config.spots.expected_separation_px)
            if delta > config.spots.separation_tolerance_px:
                result.reject_reason = "bad_separation"
                return result, previous_centers, success_roi, failed_roi
        if (
            config.quality.reject_large_jump
            and previous_centers is not None
            and _large_center_jump(result, previous_centers, config)
        ):
            result.reject_reason = "large_jump"
            return result, previous_centers, success_roi, failed_roi

        result.frame_valid = True
        result.reject_reason = ""
        new_centers = (result.x1, result.y1, result.x2, result.y2)
        return result, new_centers, success_roi, failed_roi  # type: ignore[return-value]
    except Exception:
        result.reject_reason = "unknown_error"
        return result, previous_centers, success_roi, failed_roi


def _fit_source(source, frame, config, saturation_level):  # type: ignore[no-untyped-def]
    attempts = 0
    first_failed_roi = None
    roi_result = extract_roi(
        frame,
        center_x=source.x,
        center_y=source.y,
        size_px=config.roi.size_px,
    )
    if roi_result is None:
        return FitResult.failed("roi_out_of_bounds"), attempts, None, None
    roi, origin_x, origin_y = roi_result
    fit = fit_gaussian_2d(
        roi,
        origin_x=origin_x,
        origin_y=origin_y,
        guess_x=source.x,
        guess_y=source.y,
        fit_config=config.gaussian_fit,
        quality_config=config.quality,
        saturation_level=saturation_level,
    )
    attempts += 1
    if fit.success:
        return fit, attempts, roi, None
    first_failed_roi = roi

    # Data-quality failures are final; a larger ROI cannot make them trustworthy.
    if (
        config.roi.fallback_size_px != config.roi.size_px
        and fit.failure_reason not in {"saturated", "hot_pixel_or_roi_outlier", "nan_result"}
    ):
        fallback = extract_roi(
            frame,
            center_x=source.x,
            center_y=source.y,
            size_px=config.roi.fallback_size_px,
        )
        if fallback is not None:
            roi, origin_x, origin_y = fallback
            fallback_fit = fit_gaussian_2d(
                roi,
                origin_x=origin_x,
                origin_y=origin_y,
                guess_x=source.x,
                guess_y=source.y,
                fit_config=config.gaussian_fit,
                quality_config=config.quality,
                saturation_level=saturation_level,
            )
            attempts += 1
            if fallback_fit.success:
                return fallback_fit, attempts, roi, first_failed_roi
            fit = fallback_fit
            first_failed_roi = roi
    return fit, attempts, None, first_failed_roi


def _large_center_jump(
    result: FrameResult,
    previous_centers: Tuple[float, float, float, float],
    config: AnalysisConfig,
) -> bool:
    x1, y1, x2, y2 = previous_centers
    jump1 = np.hypot(result.x1 - x1, result.y1 - y1)  # type: ignore[operator]
    jump2 = np.hypot(result.x2 - x2, result.y2 - y2)  # type: ignore[operator]
    return bool(max(jump1, jump2) > config.quality.max_center_jump_px)


def _assign_spot_ids(
    result: FrameResult,
    previous_centers: Optional[PreviousCenters],
    config: AnalysisConfig,
) -> bool:
    if None in (
        result.fit1.x0_global,
        result.fit1.y0_global,
        result.fit2.x0_global,
        result.fit2.y0_global,
    ):
        result.tracking_status = "missing_fit_center"
        return False
    if previous_centers is None:
        result.assignment_distance = 0.0
        result.assignment_swapped = False
        result.tracking_status = "initialized"
        return True

    prev_x1, prev_y1, prev_x2, prev_y2 = previous_centers
    x1 = float(result.fit1.x0_global)
    y1 = float(result.fit1.y0_global)
    x2 = float(result.fit2.x0_global)
    y2 = float(result.fit2.y0_global)
    distance_same = float(
        np.hypot(x1 - prev_x1, y1 - prev_y1)
        + np.hypot(x2 - prev_x2, y2 - prev_y2)
    )
    distance_swapped = float(
        np.hypot(x1 - prev_x2, y1 - prev_y2) + np.hypot(x2 - prev_x1, y2 - prev_y1)
    )
    if distance_swapped < distance_same:
        result.fit1, result.fit2 = result.fit2, result.fit1
        result.assignment_distance = distance_swapped
        result.assignment_swapped = True
        result.tracking_status = "swapped"
    else:
        result.assignment_distance = distance_same
        result.assignment_swapped = False
        result.tracking_status = "ok"

    max_total_distance = 2.0 * config.quality.max_spot_tracking_distance_px
    if (
        config.quality.reject_spot_tracking
        and result.assignment_distance is not None
        and result.assignment_distance > max_total_distance
    ):
        result.tracking_status = "spot_tracking_failed"
        return False
    return True
