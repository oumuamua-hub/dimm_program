"""Unit-aware DIMM calculations."""

from __future__ import annotations

import math
import warnings
from typing import Dict

ARCSEC_PER_RADIAN = 206265.0


def dimm_coefficients(aperture_diameter_m: float, baseline_m: float) -> Dict[str, float]:
    _validate_instrument(aperture_diameter_m, baseline_m, 5.0e-7)
    if baseline_m <= 2 * aperture_diameter_m:
        warnings.warn(
            "DIMM baseline は通常 2D より大きい必要があります。",
            RuntimeWarning,
            stacklevel=2,
        )
    c_l = 0.179 * aperture_diameter_m ** (-1.0 / 3.0) - 0.0968 * baseline_m ** (
        -1.0 / 3.0
    )
    c_t = 0.179 * aperture_diameter_m ** (-1.0 / 3.0) - 0.145 * baseline_m ** (-1.0 / 3.0)
    if c_l <= 0 or c_t <= 0:
        raise ValueError("DIMM 係数が不正です。aperture diameter と baseline を確認してください。")
    return {"L": c_l, "T": c_t}


def pixel_variance_to_rad2(var_px2: float, pixel_scale_arcsec_per_px: float) -> float:
    if var_px2 <= 0:
        raise ValueError("variance は正の値にしてください。")
    if pixel_scale_arcsec_per_px <= 0:
        raise ValueError("pixel scale は正の値にしてください。")
    return var_px2 * (pixel_scale_arcsec_per_px / ARCSEC_PER_RADIAN) ** 2


def r0_from_variance_rad2(variance_rad2: float, wavelength_m: float, coefficient: float) -> float:
    if variance_rad2 <= 0:
        raise ValueError("variance は正の値にしてください。")
    if wavelength_m <= 0:
        raise ValueError("wavelength は正の値にしてください。")
    if coefficient <= 0:
        raise ValueError("DIMM coefficient は正の値にしてください。")
    return (2.0 * wavelength_m**2 * coefficient / variance_rad2) ** (3.0 / 5.0)


def seeing_from_r0_arcsec(r0_m: float, wavelength_m: float) -> float:
    if r0_m <= 0:
        raise ValueError("r0 は正の値にしてください。")
    if wavelength_m <= 0:
        raise ValueError("wavelength は正の値にしてください。")
    return 0.98 * wavelength_m / r0_m * ARCSEC_PER_RADIAN


def zenith_correction_factor(zenith_deg: float) -> float:
    if not (0 <= zenith_deg < 90):
        raise ValueError("zenith angle は [0, 90) の範囲にしてください。")
    return math.cos(math.radians(zenith_deg)) ** (3.0 / 5.0)


def zenith_correct(
    *,
    r0_observed_m: float,
    seeing_observed_arcsec: float,
    zenith_deg: float,
) -> Dict[str, float]:
    cos_z_factor = zenith_correction_factor(zenith_deg)
    return {
        "r0_zenith_m": r0_observed_m / cos_z_factor,
        "seeing_zenith_arcsec": seeing_observed_arcsec * cos_z_factor,
    }


def compute_dimm_block(
    *,
    var_L_px2: float,
    var_T_px2: float,
    pixel_scale_arcsec_per_px: float,
    aperture_diameter_m: float,
    baseline_m: float,
    wavelength_m: float,
    zenith_deg: float,
    zenith_correction: bool = True,
) -> Dict[str, float]:
    _validate_instrument(aperture_diameter_m, baseline_m, wavelength_m)
    coefficients = dimm_coefficients(aperture_diameter_m, baseline_m)
    var_L_rad2 = pixel_variance_to_rad2(var_L_px2, pixel_scale_arcsec_per_px)
    var_T_rad2 = pixel_variance_to_rad2(var_T_px2, pixel_scale_arcsec_per_px)
    r0_L = r0_from_variance_rad2(var_L_rad2, wavelength_m, coefficients["L"])
    r0_T = r0_from_variance_rad2(var_T_rad2, wavelength_m, coefficients["T"])
    r0_mean = 0.5 * (r0_L + r0_T)
    seeing_L = seeing_from_r0_arcsec(r0_L, wavelength_m)
    seeing_T = seeing_from_r0_arcsec(r0_T, wavelength_m)
    seeing_mean = 0.5 * (seeing_L + seeing_T)

    if zenith_correction:
        cos_z_factor = zenith_correction_factor(zenith_deg)
        corrected = zenith_correct(
            r0_observed_m=r0_mean,
            seeing_observed_arcsec=seeing_mean,
            zenith_deg=zenith_deg,
        )
        r0_mean_zenith = corrected["r0_zenith_m"]
        seeing_mean_zenith = corrected["seeing_zenith_arcsec"]
        r0_L_zenith = r0_L / cos_z_factor
        r0_T_zenith = r0_T / cos_z_factor
        seeing_L_zenith = seeing_L * cos_z_factor
        seeing_T_zenith = seeing_T * cos_z_factor
    else:
        r0_mean_zenith = r0_mean
        seeing_mean_zenith = seeing_mean
        r0_L_zenith = r0_L
        r0_T_zenith = r0_T
        seeing_L_zenith = seeing_L
        seeing_T_zenith = seeing_T

    return {
        "var_L_rad2": var_L_rad2,
        "var_T_rad2": var_T_rad2,
        "r0_L_observed_m": r0_L,
        "r0_T_observed_m": r0_T,
        "r0_mean_observed_m": r0_mean,
        "r0_L_zenith_m": r0_L_zenith,
        "r0_T_zenith_m": r0_T_zenith,
        "r0_mean_zenith_m": r0_mean_zenith,
        "seeing_L_observed_arcsec": seeing_L,
        "seeing_T_observed_arcsec": seeing_T,
        "seeing_mean_observed_arcsec": seeing_mean,
        "seeing_L_zenith_arcsec": seeing_L_zenith,
        "seeing_T_zenith_arcsec": seeing_T_zenith,
        "seeing_mean_zenith_arcsec": seeing_mean_zenith,
    }


def _validate_instrument(
    aperture_diameter_m: float,
    baseline_m: float,
    wavelength_m: float,
) -> None:
    if aperture_diameter_m <= 0:
        raise ValueError("aperture diameter D は正の値にしてください。")
    if baseline_m <= 0:
        raise ValueError("baseline B は正の値にしてください。")
    if wavelength_m <= 0:
        raise ValueError("wavelength は正の値にしてください。")
