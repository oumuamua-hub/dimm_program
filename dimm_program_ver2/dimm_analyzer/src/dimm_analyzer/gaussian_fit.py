"""Bounded non-rotated elliptical 2D Gaussian fitting."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from .models import FitResult

FWHM_FACTOR = 2.3548200450309493


def fit_gaussian_2d(
    roi: np.ndarray,
    *,
    origin_x: int,
    origin_y: int,
    guess_x: Optional[float],
    guess_y: Optional[float],
    fit_config,  # type: ignore[no-untyped-def]
    quality_config,  # type: ignore[no-untyped-def]
    saturation_level: Optional[float] = None,
) -> FitResult:
    data = np.asarray(roi, dtype=float)
    if data.ndim != 2 or data.size == 0:
        return FitResult.failed("nan_result")
    if not np.all(np.isfinite(data)):
        return FitResult.failed("nan_result")

    height, width = data.shape
    peak_data = float(np.max(data))
    y_bright, x_bright = np.unravel_index(int(np.argmax(data)), data.shape)
    x0 = float(x_bright if guess_x is None else np.clip(guess_x - origin_x, 0, width - 1))
    y0 = float(y_bright if guess_y is None else np.clip(guess_y - origin_y, 0, height - 1))
    initial_saturation = _saturation_diagnostics(
        data,
        center_x=x0,
        center_y=y0,
        saturation_level=saturation_level,
        saturation_margin=quality_config.saturation_margin,
        core_radius_px=quality_config.saturation_core_radius_px,
    )
    initial_saturation_reason = _saturation_failure_reason(
        initial_saturation,
        quality_config,
        saturation_level,
    )
    if initial_saturation_reason is not None:
        return FitResult(
            success=False,
            failure_reason=initial_saturation_reason,
            x0_global=x0 + origin_x,
            y0_global=y0 + origin_y,
            x0_roi=x0,
            y0_roi=y0,
            peak=initial_saturation["peak_raw_max"],
            **initial_saturation,
        )

    border = np.concatenate([data[0, :], data[-1, :], data[:, 0], data[:, -1]])
    background0 = max(0.0, float(np.median(border)))
    amplitude0 = max(1.0, peak_data - background0)
    sigma0 = np.clip(5.0 / FWHM_FACTOR, fit_config.fit_sigma_min_px, fit_config.fit_sigma_max_px)

    yy, xx = np.indices(data.shape, dtype=float)
    p0 = np.array([background0, amplitude0, x0, y0, sigma0, sigma0], dtype=float)
    upper_bg = max(float(np.max(data) * 2.0 + 1.0), 1.0)
    upper_amp = max(float(np.ptp(data) * 4.0 + 1.0), amplitude0 + 1.0)
    lower = np.array(
        [0.0, 0.0, 0.0, 0.0, fit_config.fit_sigma_min_px, fit_config.fit_sigma_min_px],
        dtype=float,
    )
    upper = np.array(
        [
            upper_bg,
            upper_amp,
            float(width - 1),
            float(height - 1),
            fit_config.fit_sigma_max_px,
            fit_config.fit_sigma_max_px,
        ],
        dtype=float,
    )
    p0 = np.minimum(np.maximum(p0, lower), upper)

    def residual(params: np.ndarray) -> np.ndarray:
        return (_model(xx, yy, params) - data).ravel()

    try:
        result = least_squares(
            residual,
            p0,
            bounds=(lower, upper),
            max_nfev=fit_config.max_iterations,
            loss=fit_config.robust_loss,
        )
    except Exception as exc:
        return FitResult.failed(f"optimizer_error:{exc}")

    params = result.x
    if not result.success:
        return FitResult.failed("optimizer_failed")
    if not np.all(np.isfinite(params)):
        return FitResult.failed("nan_result")

    background, amplitude, x_roi, y_roi, sigma_x, sigma_y = [float(value) for value in params]
    fwhm_x = FWHM_FACTOR * sigma_x
    fwhm_y = FWHM_FACTOR * sigma_y
    fwhm_mean = 0.5 * (fwhm_x + fwhm_y)
    flux = 2.0 * np.pi * amplitude * sigma_x * sigma_y
    residual_rms = float(np.sqrt(np.mean(residual(params) ** 2)))
    fit_saturation = _saturation_diagnostics(
        data,
        center_x=x_roi,
        center_y=y_roi,
        saturation_level=saturation_level,
        saturation_margin=quality_config.saturation_margin,
        core_radius_px=quality_config.saturation_core_radius_px,
    )

    failure_reason = _validate_fit(
        amplitude=amplitude,
        x_roi=x_roi,
        y_roi=y_roi,
        width=width,
        height=height,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        fwhm_x=fwhm_x,
        fwhm_y=fwhm_y,
        peak=fit_saturation["peak_raw_max"],
        flux=flux,
        fit_config=fit_config,
        quality_config=quality_config,
        saturation_level=saturation_level,
        saturation_diagnostics=fit_saturation,
    )
    if failure_reason is not None:
        return FitResult(
            success=False,
            failure_reason=failure_reason,
            x0_global=x_roi + origin_x,
            y0_global=y_roi + origin_y,
            x0_roi=x_roi,
            y0_roi=y_roi,
            background=background,
            amplitude=amplitude,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            fwhm_x=fwhm_x,
            fwhm_y=fwhm_y,
            fwhm_mean=fwhm_mean,
            residual_rms=residual_rms,
            peak=fit_saturation["peak_raw_max"],
            **fit_saturation,
            flux=flux,
            nfev=int(result.nfev),
        )

    return FitResult(
        success=True,
        x0_global=x_roi + origin_x,
        y0_global=y_roi + origin_y,
        x0_roi=x_roi,
        y0_roi=y_roi,
        background=background,
        amplitude=amplitude,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        fwhm_x=fwhm_x,
        fwhm_y=fwhm_y,
        fwhm_mean=fwhm_mean,
        residual_rms=residual_rms,
        peak=fit_saturation["peak_raw_max"],
        **fit_saturation,
        flux=flux,
        nfev=int(result.nfev),
    )


def _model(xx: np.ndarray, yy: np.ndarray, params: np.ndarray) -> np.ndarray:
    background, amplitude, x0, y0, sigma_x, sigma_y = params
    exponent = -(
        ((xx - x0) ** 2) / (2.0 * sigma_x**2)
        + ((yy - y0) ** 2) / (2.0 * sigma_y**2)
    )
    return background + amplitude * np.exp(exponent)


def _validate_fit(
    *,
    amplitude: float,
    x_roi: float,
    y_roi: float,
    width: int,
    height: int,
    sigma_x: float,
    sigma_y: float,
    fwhm_x: float,
    fwhm_y: float,
    peak: float,
    flux: float,
    fit_config,  # type: ignore[no-untyped-def]
    quality_config,  # type: ignore[no-untyped-def]
    saturation_level: Optional[float],
    saturation_diagnostics,
) -> Optional[str]:
    values = [amplitude, x_roi, y_roi, sigma_x, sigma_y, fwhm_x, fwhm_y, peak, flux]
    if not np.all(np.isfinite(values)):
        return "nan_result"
    if amplitude < fit_config.min_amplitude:
        return "low_flux"
    if not (0 <= x_roi < width and 0 <= y_roi < height):
        return "fit_center_outside_roi"
    if not (fit_config.fit_sigma_min_px <= sigma_x <= fit_config.fit_sigma_max_px):
        return "bad_fwhm"
    if not (fit_config.fit_sigma_min_px <= sigma_y <= fit_config.fit_sigma_max_px):
        return "bad_fwhm"
    if quality_config.reject_bad_fwhm and not (
        fit_config.fit_fwhm_min_px <= fwhm_x <= fit_config.fit_fwhm_max_px
        and fit_config.fit_fwhm_min_px <= fwhm_y <= fit_config.fit_fwhm_max_px
    ):
        return "bad_fwhm"
    saturation_reason = _saturation_failure_reason(
        saturation_diagnostics,
        quality_config,
        saturation_level,
    )
    if saturation_reason is not None:
        return saturation_reason
    if flux < quality_config.min_flux:
        return "low_flux"
    return None


def _saturation_diagnostics(
    data: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    saturation_level: Optional[float],
    saturation_margin: float,
    core_radius_px: float,
) -> dict:
    yy, xx = np.indices(data.shape, dtype=float)
    core_mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= core_radius_px**2
    if not np.any(core_mask):
        cx = int(np.clip(round(center_x), 0, data.shape[1] - 1))
        cy = int(np.clip(round(center_y), 0, data.shape[0] - 1))
        core_mask[cy, cx] = True
    core_values = data[core_mask]
    threshold = (
        saturation_level - saturation_margin
        if saturation_level is not None
        else None
    )
    if threshold is None:
        saturated_roi_pixel_count = 0
        saturated_core_pixel_count = 0
    else:
        saturated_roi_pixel_count = int(np.sum(data >= threshold))
        saturated_core_pixel_count = int(np.sum(core_values >= threshold))
    hot_pixel_or_roi_outlier = (
        saturated_roi_pixel_count > 0 and saturated_core_pixel_count == 0
    )
    return {
        "peak_raw_max": float(np.max(data)),
        "peak_core_max": float(np.max(core_values)),
        "saturated_core_pixel_count": saturated_core_pixel_count,
        "saturated_roi_pixel_count": saturated_roi_pixel_count,
        "hot_pixel_or_roi_outlier": hot_pixel_or_roi_outlier,
    }


def _saturation_failure_reason(
    diagnostics: dict,
    quality_config,  # type: ignore[no-untyped-def]
    saturation_level: Optional[float],
) -> Optional[str]:
    if saturation_level is None:
        return None
    # A saturated core invalidates the stellar profile.
    # Isolated ROI pixels remain a separate diagnostic class.
    if (
        quality_config.reject_saturated
        and quality_config.reject_saturated_core
        and diagnostics["saturated_core_pixel_count"]
        >= quality_config.saturation_core_min_pixels
    ):
        return "saturated"
    if (
        quality_config.classify_roi_saturated_pixel_as_hot_pixel
        and diagnostics["hot_pixel_or_roi_outlier"]
    ):
        return "hot_pixel_or_roi_outlier"
    return None
