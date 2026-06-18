"""Data structures used by the DIMM analysis pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SERMetadata:
    path: Path
    width: int
    height: int
    frame_count: int
    pixel_depth: int
    color_id: int
    color_mode: str
    little_endian: bool
    observer: str = ""
    instrument: str = ""
    telescope: str = ""
    date_time_local_raw: int = 0
    date_time_utc_raw: int = 0
    timestamps_available: bool = False
    timestamp_count: int = 0
    estimated_fps: Optional[float] = None
    header_size: int = 178

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


@dataclass
class Source:
    x: float
    y: float
    flux: float
    peak: float


@dataclass
class FitResult:
    success: bool
    failure_reason: Optional[str] = None
    x0_global: Optional[float] = None
    y0_global: Optional[float] = None
    x0_roi: Optional[float] = None
    y0_roi: Optional[float] = None
    background: Optional[float] = None
    amplitude: Optional[float] = None
    sigma_x: Optional[float] = None
    sigma_y: Optional[float] = None
    fwhm_x: Optional[float] = None
    fwhm_y: Optional[float] = None
    fwhm_mean: Optional[float] = None
    residual_rms: Optional[float] = None
    peak: Optional[float] = None
    peak_raw_max: Optional[float] = None
    peak_core_max: Optional[float] = None
    saturated_core_pixel_count: int = 0
    saturated_roi_pixel_count: int = 0
    hot_pixel_or_roi_outlier: bool = False
    flux: Optional[float] = None
    nfev: int = 0

    @classmethod
    def failed(
        cls,
        reason: str,
        *,
        peak_raw_max: Optional[float] = None,
        peak_core_max: Optional[float] = None,
        saturated_core_pixel_count: int = 0,
        saturated_roi_pixel_count: int = 0,
        hot_pixel_or_roi_outlier: bool = False,
    ) -> "FitResult":
        return cls(
            success=False,
            failure_reason=reason,
            peak_raw_max=peak_raw_max,
            peak_core_max=peak_core_max,
            saturated_core_pixel_count=saturated_core_pixel_count,
            saturated_roi_pixel_count=saturated_roi_pixel_count,
            hot_pixel_or_roi_outlier=hot_pixel_or_roi_outlier,
        )

    def prefixed_dict(self, prefix: str) -> Dict[str, Any]:
        return {
            f"amp{prefix}": self.amplitude,
            f"bg{prefix}": self.background,
            f"sigma_x{prefix}": self.sigma_x,
            f"sigma_y{prefix}": self.sigma_y,
            f"fwhm_x{prefix}": self.fwhm_x,
            f"fwhm_y{prefix}": self.fwhm_y,
            f"fwhm_mean{prefix}": self.fwhm_mean,
            f"residual_rms{prefix}": self.residual_rms,
            f"flux{prefix}": self.flux,
            f"peak{prefix}": self.peak,
            f"peak_raw_max{prefix}": self.peak_raw_max,
            f"peak_core_max{prefix}": self.peak_core_max,
            f"saturated_core_pixel_count{prefix}": self.saturated_core_pixel_count,
            f"saturated_roi_pixel_count{prefix}": self.saturated_roi_pixel_count,
            f"hot_pixel_or_roi_outlier{prefix}": self.hot_pixel_or_roi_outlier,
        }


@dataclass
class FrameResult:
    frame_index: int
    time_sec: Optional[float]
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None
    fit1: FitResult = field(default_factory=lambda: FitResult.failed("not_attempted"))
    fit2: FitResult = field(default_factory=lambda: FitResult.failed("not_attempted"))
    fit_attempts: int = 0
    frame_valid: bool = False
    reject_reason: str = "unknown_error"
    x1: Optional[float] = None
    y1: Optional[float] = None
    x2: Optional[float] = None
    y2: Optional[float] = None
    dx_px: Optional[float] = None
    dy_px: Optional[float] = None
    dL_px: Optional[float] = None
    dT_px: Optional[float] = None
    separation_px: Optional[float] = None
    candidate1_x: Optional[float] = None
    candidate1_y: Optional[float] = None
    candidate2_x: Optional[float] = None
    candidate2_y: Optional[float] = None
    assigned_spot1_x: Optional[float] = None
    assigned_spot1_y: Optional[float] = None
    assigned_spot2_x: Optional[float] = None
    assigned_spot2_y: Optional[float] = None
    assignment_distance: Optional[float] = None
    assignment_swapped: Optional[bool] = None
    tracking_status: str = ""
    relative_reference_dx_px: Optional[float] = None
    relative_reference_dy_px: Optional[float] = None
    relative_delta_dx_px: Optional[float] = None
    relative_delta_dy_px: Optional[float] = None
    fwhm_reference1: Optional[float] = None
    fwhm_reference2: Optional[float] = None

    def set_from_fits(self) -> None:
        self.x1 = self.fit1.x0_global
        self.y1 = self.fit1.y0_global
        self.x2 = self.fit2.x0_global
        self.y2 = self.fit2.y0_global
        self.assigned_spot1_x = self.x1
        self.assigned_spot1_y = self.y1
        self.assigned_spot2_x = self.x2
        self.assigned_spot2_y = self.y2
        if None not in (self.x1, self.y1, self.x2, self.y2):
            self.dx_px = float(self.x2 - self.x1)  # type: ignore[operator]
            self.dy_px = float(self.y2 - self.y1)  # type: ignore[operator]
            self.separation_px = float(np.hypot(self.dx_px, self.dy_px))

    def to_csv_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "frame_index": self.frame_index,
            "time_sec": self.time_sec,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "dx_px": self.dx_px,
            "dy_px": self.dy_px,
            "dL_px": self.dL_px,
            "dT_px": self.dT_px,
            "separation_px": self.separation_px,
            "candidate1_x": self.candidate1_x,
            "candidate1_y": self.candidate1_y,
            "candidate2_x": self.candidate2_x,
            "candidate2_y": self.candidate2_y,
            "assigned_spot1_x": self.assigned_spot1_x,
            "assigned_spot1_y": self.assigned_spot1_y,
            "assigned_spot2_x": self.assigned_spot2_x,
            "assigned_spot2_y": self.assigned_spot2_y,
            "assignment_distance": self.assignment_distance,
            "assignment_swapped": self.assignment_swapped,
            "tracking_status": self.tracking_status,
            "relative_reference_dx_px": self.relative_reference_dx_px,
            "relative_reference_dy_px": self.relative_reference_dy_px,
            "relative_delta_dx_px": self.relative_delta_dx_px,
            "relative_delta_dy_px": self.relative_delta_dy_px,
            "fwhm_reference1": self.fwhm_reference1,
            "fwhm_reference2": self.fwhm_reference2,
            "fit_success_spot1": self.fit1.success,
            "fit_success_spot2": self.fit2.success,
            "frame_valid": self.frame_valid,
            "reject_reason": "" if self.frame_valid else self.reject_reason,
        }
        data.update(self.fit1.prefixed_dict("1"))
        data.update(self.fit2.prefixed_dict("2"))
        return data


@dataclass
class OrientationResult:
    mode: str
    angle_deg: Optional[float]
    confidence: Optional[float] = None
    fallback_used: bool = False
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BlockResult:
    block_index: int
    time_start_sec: Optional[float]
    time_end_sec: Optional[float]
    frame_start: int
    frame_end: int
    n_total: int
    n_valid: int
    frame_fit_success_rate: float
    spot_fit_success_rate: float
    var_L_px2: Optional[float] = None
    var_T_px2: Optional[float] = None
    var_L_rad2: Optional[float] = None
    var_T_rad2: Optional[float] = None
    r0_L_observed_m: Optional[float] = None
    r0_T_observed_m: Optional[float] = None
    r0_mean_observed_m: Optional[float] = None
    r0_L_zenith_m: Optional[float] = None
    r0_T_zenith_m: Optional[float] = None
    r0_mean_zenith_m: Optional[float] = None
    seeing_L_observed_arcsec: Optional[float] = None
    seeing_T_observed_arcsec: Optional[float] = None
    seeing_mean_observed_arcsec: Optional[float] = None
    seeing_L_zenith_arcsec: Optional[float] = None
    seeing_T_zenith_arcsec: Optional[float] = None
    seeing_mean_zenith_arcsec: Optional[float] = None
    valid: bool = False

    def to_csv_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("valid", None)
        return data


@dataclass
class AnalysisResult:
    frame_results: List[FrameResult]
    block_results: List[BlockResult]
    summary: Dict[str, Any]
    warnings: List[str]
    orientation: OrientationResult
    orientation_scan_rows: List[Dict[str, Any]] = field(default_factory=list)
    rejection_summary_rows: List[Dict[str, Any]] = field(default_factory=list)
    frame_distribution_rows: List[Dict[str, Any]] = field(default_factory=list)
    spot_assignment_rows: List[Dict[str, Any]] = field(default_factory=list)
    relative_motion_outlier_rows: List[Dict[str, Any]] = field(default_factory=list)
    fwhm_outlier_rows: List[Dict[str, Any]] = field(default_factory=list)
    roi_safety_rows: List[Dict[str, Any]] = field(default_factory=list)
    example_success_roi: Optional[np.ndarray] = None
    example_failed_roi: Optional[np.ndarray] = None
