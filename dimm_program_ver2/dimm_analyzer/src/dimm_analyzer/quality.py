"""Quality helpers and rejection reason normalization."""

from __future__ import annotations

from .models import FitResult

FIT_FAILURE_TO_FRAME_REASON = {
    "saturated": "saturated",
    "low_flux": "low_flux",
    "bad_fwhm": "bad_fwhm",
    "nan_result": "nan_result",
    "hot_pixel_or_roi_outlier": "hot_pixel_or_roi_outlier",
}


def frame_reject_reason_for_fit_fail(fit1: FitResult, fit2: FitResult) -> str:
    for fit in (fit1, fit2):
        if fit.failure_reason in FIT_FAILURE_TO_FRAME_REASON:
            return FIT_FAILURE_TO_FRAME_REASON[fit.failure_reason]  # type: ignore[index]
    if not fit1.success:
        return "fit_failed_spot1"
    if not fit2.success:
        return "fit_failed_spot2"
    return "unknown_error"


def ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)
