import pytest

from dimm_analyzer.dimm_math import (
    compute_dimm_block,
    pixel_variance_to_rad2,
    r0_from_variance_rad2,
    seeing_from_r0_arcsec,
    zenith_correct,
    zenith_correction_factor,
)


def test_r0_and_seeing_positive_variance():
    variance_rad2 = pixel_variance_to_rad2(0.04, 0.643)
    r0 = r0_from_variance_rad2(variance_rad2, 5.0e-7, 0.2)
    seeing = seeing_from_r0_arcsec(r0, 5.0e-7)
    assert r0 > 0
    assert seeing > 0


def test_block_computation_and_zenith_correction():
    values = compute_dimm_block(
        var_L_px2=0.04,
        var_T_px2=0.05,
        pixel_scale_arcsec_per_px=0.643,
        aperture_diameter_m=0.045,
        baseline_m=0.150,
        wavelength_m=5.0e-7,
        zenith_deg=30.0,
    )
    assert values["r0_mean_observed_m"] > 0
    assert values["seeing_mean_observed_arcsec"] > 0
    assert values["r0_mean_zenith_m"] > values["r0_mean_observed_m"]
    assert values["seeing_mean_zenith_arcsec"] < values["seeing_mean_observed_arcsec"]
    assert values["seeing_L_zenith_arcsec"] < values["seeing_L_observed_arcsec"]
    assert values["seeing_T_zenith_arcsec"] < values["seeing_T_observed_arcsec"]


def test_zenith_zero_is_identity():
    corrected = zenith_correct(r0_observed_m=0.1, seeing_observed_arcsec=1.0, zenith_deg=0)
    assert corrected["r0_zenith_m"] == pytest.approx(0.1)
    assert corrected["seeing_zenith_arcsec"] == pytest.approx(1.0)
    assert zenith_correction_factor(0) == pytest.approx(1.0)


def test_invalid_variance_raises():
    with pytest.raises(ValueError):
        pixel_variance_to_rad2(0.0, 0.643)
