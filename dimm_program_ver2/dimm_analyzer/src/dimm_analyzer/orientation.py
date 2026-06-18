"""Orientation estimation for longitudinal/transverse DIMM motion."""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .dimm_math import compute_dimm_block
from .models import OrientationResult


def rotate_motion(
    dx: np.ndarray,
    dy: np.ndarray,
    angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    d_l = dx * cos_t + dy * sin_t
    d_t = -dx * sin_t + dy * cos_t
    return d_l, d_t


def estimate_orientation(
    *,
    mode: str,
    dx: np.ndarray,
    dy: np.ndarray,
    block_masks: Sequence[np.ndarray],
    orientation_config,  # type: ignore[no-untyped-def]
    instrument_config,  # type: ignore[no-untyped-def]
    pixel_scale_arcsec_per_px: float,
    warnings_out: List[str],
) -> OrientationResult:
    if len(dx) == 0:
        warning = "orientation 推定に使える valid frame がありません。"
        warnings_out.append(warning)
        return OrientationResult(mode=mode, angle_deg=None, confidence=0.0, warning=warning)

    mode = mode.lower()
    if mode == "manual":
        if orientation_config.mask_angle_deg is None:
            raise ValueError(
                "orientation.mode='manual' の場合は orientation.mask_angle_deg が必要です。"
            )
        return OrientationResult(
            mode=mode,
            angle_deg=_normalize_180(orientation_config.mask_angle_deg),
        )

    if mode == "auto_pair":
        warning = (
            "orientation.auto_pair は暫定推定です。星像ペア角が物理的な開口ベースライン角と"
            "一致するとは限りません。"
        )
        warnings_out.append(warning)
        return OrientationResult(
            mode=mode,
            angle_deg=_mean_pair_angle(dx, dy),
            confidence=None,
            warning=warning,
        )

    if mode != "auto_consistency":
        warning = f"未知の orientation.mode={mode!r} です。auto_pair に fallback します。"
        warnings_out.append(warning)
        return OrientationResult(
            mode="auto_pair",
            angle_deg=_mean_pair_angle(dx, dy),
            confidence=None,
            fallback_used=True,
            warning=warning,
        )

    candidate = _estimate_by_consistency(
        dx=dx,
        dy=dy,
        block_masks=block_masks,
        step_deg=orientation_config.angle_grid_step_deg,
        instrument_config=instrument_config,
        pixel_scale_arcsec_per_px=pixel_scale_arcsec_per_px,
    )
    if candidate is None:
        warning = "auto_consistency で評価可能な valid seeing block がありません。"
        warnings_out.append(warning)
        if orientation_config.allow_pair_angle_as_fallback:
            return OrientationResult(
                mode="auto_pair",
                angle_deg=_mean_pair_angle(dx, dy),
                confidence=0.0,
                fallback_used=True,
                warning=warning,
            )
        return OrientationResult(mode=mode, angle_deg=None, confidence=0.0, warning=warning)

    angle, mismatch = candidate
    confidence = float(math.exp(-mismatch))
    warning = None
    fallback_used = False
    if confidence < 0.6:
        warning = (
            "auto_consistency の orientation 信頼度が低いです。"
            "物理的な mask angle を確認してください。"
        )
        warnings_out.append(warning)
        if orientation_config.allow_pair_angle_as_fallback:
            fallback_used = True
            angle = _mean_pair_angle(dx, dy)
    return OrientationResult(
        mode=mode,
        angle_deg=_normalize_180(angle),
        confidence=confidence,
        fallback_used=fallback_used,
        warning=warning,
    )


def _estimate_by_consistency(
    *,
    dx: np.ndarray,
    dy: np.ndarray,
    block_masks: Sequence[np.ndarray],
    step_deg: float,
    instrument_config,  # type: ignore[no-untyped-def]
    pixel_scale_arcsec_per_px: float,
) -> Optional[Tuple[float, float]]:
    if step_deg <= 0:
        step_deg = 1.0
    best_angle = None
    best_mismatch = float("inf")
    for angle in np.arange(0.0, 180.0, step_deg):
        d_l, d_t = rotate_motion(dx, dy, float(angle))
        mismatches: List[float] = []
        for mask in block_masks:
            if int(np.sum(mask)) < 2:
                continue
            var_l = float(np.var(d_l[mask], ddof=1))
            var_t = float(np.var(d_t[mask], ddof=1))
            try:
                values = compute_dimm_block(
                    var_L_px2=var_l,
                    var_T_px2=var_t,
                    pixel_scale_arcsec_per_px=pixel_scale_arcsec_per_px,
                    aperture_diameter_m=instrument_config.aperture_diameter_m,
                    baseline_m=instrument_config.baseline_m,
                    wavelength_m=instrument_config.wavelength_m,
                    zenith_deg=instrument_config.zenith_deg,
                    zenith_correction=instrument_config.zenith_correction,
                )
            except Exception:
                continue
            seeing_l = values["seeing_L_observed_arcsec"]
            seeing_t = values["seeing_T_observed_arcsec"]
            if seeing_l > 0 and seeing_t > 0:
                mismatches.append(abs(math.log(seeing_l / seeing_t)))
        if not mismatches:
            continue
        mismatch = float(np.median(mismatches))
        if mismatch < best_mismatch:
            best_mismatch = mismatch
            best_angle = float(angle)
    if best_angle is None:
        return None
    return best_angle, best_mismatch


def _mean_pair_angle(dx: Iterable[float], dy: Iterable[float]) -> float:
    angles = np.arctan2(np.asarray(list(dy), dtype=float), np.asarray(list(dx), dtype=float))
    if len(angles) == 0:
        return 0.0
    doubled = 2.0 * angles
    mean_angle = 0.5 * math.atan2(float(np.mean(np.sin(doubled))), float(np.mean(np.cos(doubled))))
    return _normalize_180(math.degrees(mean_angle))


def _normalize_180(angle_deg: float) -> float:
    value = float(angle_deg) % 180.0
    if value < 0:
        value += 180.0
    return value
