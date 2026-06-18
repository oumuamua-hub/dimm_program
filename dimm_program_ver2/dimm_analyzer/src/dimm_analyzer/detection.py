"""Spot detection and ROI extraction."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, List, Optional, Tuple

import numpy as np

from .models import Source


def estimate_background(frame: np.ndarray) -> Tuple[float, float]:
    try:
        from astropy.stats import sigma_clipped_stats

        mean, median, std = sigma_clipped_stats(frame, sigma=3.0, maxiters=5)
        _ = mean
        return float(median), float(std)
    except Exception:
        median = float(np.nanmedian(frame))
        std = float(np.nanstd(frame))
        return median, std


def detect_sources(frame: np.ndarray, detection_config) -> List[Source]:  # type: ignore[no-untyped-def]
    background, background_rms = estimate_background(frame)
    threshold = background + detection_config.threshold_sigma * max(background_rms, 1e-12)

    if detection_config.method == "DAOStarFinder":
        sources = _detect_with_daofinder(frame, background, background_rms, detection_config)
        if sources:
            return sources[: detection_config.max_sources]

    return _detect_with_local_maxima(frame, threshold, detection_config.max_sources)


def select_spot_pair(
    sources: Iterable[Source],
    *,
    expected_separation_px: float,
    separation_tolerance_px: float,
) -> Optional[Tuple[Source, Source]]:
    candidates = list(sources)
    best_pair: Optional[Tuple[Source, Source]] = None
    best_delta = float("inf")
    for left, right in combinations(candidates, 2):
        separation = float(np.hypot(right.x - left.x, right.y - left.y))
        delta = abs(separation - expected_separation_px)
        if delta <= separation_tolerance_px and delta < best_delta:
            best_delta = delta
            best_pair = (left, right)
    if best_pair is None:
        return None
    return tuple(sorted(best_pair, key=lambda source: (source.x, source.y)))  # type: ignore[return-value]


def extract_roi(
    frame: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    size_px: int,
) -> Optional[Tuple[np.ndarray, int, int]]:
    if size_px % 2 == 0:
        raise ValueError("ROI size must be an odd integer.")
    half = size_px // 2
    cx = int(round(center_x))
    cy = int(round(center_y))
    x0 = cx - half
    y0 = cy - half
    x1 = x0 + size_px
    y1 = y0 + size_px
    if x0 < 0 or y0 < 0 or x1 > frame.shape[1] or y1 > frame.shape[0]:
        return None
    return frame[y0:y1, x0:x1].copy(), x0, y0


def _detect_with_daofinder(
    frame: np.ndarray,
    background: float,
    background_rms: float,
    detection_config,  # type: ignore[no-untyped-def]
) -> List[Source]:
    try:
        from photutils.detection import DAOStarFinder
    except Exception:
        return []

    try:
        finder = DAOStarFinder(
            fwhm=detection_config.fwhm_guess_px,
            threshold=detection_config.threshold_sigma * max(background_rms, 1e-12),
        )
        table = finder(frame - background)
    except Exception:
        return []
    if table is None or len(table) == 0:
        return []

    sources: List[Source] = []
    for row in table:
        try:
            x = float(row["xcentroid"])
            y = float(row["ycentroid"])
            flux = float(row["flux"]) if "flux" in table.colnames else 0.0
            peak = float(row["peak"]) if "peak" in table.colnames else float(frame[int(y), int(x)])
        except Exception:
            continue
        if np.isfinite(x) and np.isfinite(y):
            sources.append(Source(x=x, y=y, flux=flux, peak=peak))
    sources.sort(key=lambda source: source.peak, reverse=True)
    return sources


def _detect_with_local_maxima(
    frame: np.ndarray,
    threshold: float,
    max_sources: int,
) -> List[Source]:
    try:
        from scipy.ndimage import maximum_filter

        local_max = frame == maximum_filter(frame, size=3, mode="nearest")
    except Exception:
        padded = np.pad(frame, 1, mode="edge")
        local_max = np.ones_like(frame, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                local_max &= frame >= padded[dy : dy + frame.shape[0], dx : dx + frame.shape[1]]

    mask = local_max & np.isfinite(frame) & (frame > threshold)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []
    peaks = frame[ys, xs]
    order = np.argsort(peaks)[::-1]
    sources: List[Source] = []
    for idx in order[: max_sources * 4]:
        x = int(xs[idx])
        y = int(ys[idx])
        x0 = max(0, x - 2)
        x1 = min(frame.shape[1], x + 3)
        y0 = max(0, y - 2)
        y1 = min(frame.shape[0], y + 3)
        patch = frame[y0:y1, x0:x1]
        weights = np.clip(patch - np.nanmedian(frame), 0, None)
        total = float(np.sum(weights))
        if total > 0:
            yy, xx = np.indices(patch.shape)
            cx = float(np.sum((xx + x0) * weights) / total)
            cy = float(np.sum((yy + y0) * weights) / total)
        else:
            cx = float(x)
            cy = float(y)
        sources.append(Source(x=cx, y=cy, flux=total, peak=float(frame[y, x])))
        if len(sources) >= max_sources:
            break
    return sources
