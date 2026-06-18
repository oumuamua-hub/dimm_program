import csv
import json
import math
import struct

import numpy as np

from dimm_analyzer.config import AnalysisConfig
from dimm_analyzer.pipeline import analyze_frame_sequence, analyze_ser


def _two_spot_frame(index, size=96, y2_override=None):
    yy, xx = np.indices((size, size), dtype=float)
    shift = 0.25 * np.sin(index / 3.0)
    x1 = 25.0 + shift
    y1 = 47.0 + 0.12 * np.cos(index / 4.0)
    x2 = 75.0 - shift
    y2 = 47.5 - 0.10 * np.sin(index / 5.0) if y2_override is None else y2_override
    frame = 90.0
    frame += 2500.0 * np.exp(-(((xx - x1) ** 2) / (2 * 1.8**2) + ((yy - y1) ** 2) / (2 * 1.9**2)))
    frame += 2300.0 * np.exp(-(((xx - x2) ** 2) / (2 * 2.0**2) + ((yy - y2) ** 2) / (2 * 1.8**2)))
    rng = np.random.default_rng(index)
    return frame + rng.normal(0, 3.0, frame.shape)


def _write_mono16_ser(path, frames):
    frame_count = len(frames)
    height, width = frames[0].shape
    header = struct.pack(
        "<14s7I40s40s40sQQ",
        b"LUCAM-RECORDER",
        0,
        0,
        0,
        width,
        height,
        16,
        frame_count,
        b"Observer",
        b"Camera",
        b"Scope",
        0,
        0,
    )
    body = b"".join(np.asarray(frame, dtype="<u2").tobytes() for frame in frames)
    path.write_bytes(header + body)


def _edge_spot_frame(index, size=80):
    yy, xx = np.indices((size, size), dtype=float)
    x1 = 13.0
    y1 = 40.0 + 0.05 * np.sin(index)
    x2 = 63.0
    y2 = 40.0 - 0.05 * np.sin(index)
    frame = 100.0
    frame += 30000.0 * np.exp(-(((xx - x1) ** 2) / (2 * 1.8**2) + ((yy - y1) ** 2) / (2 * 1.8**2)))
    frame += 28000.0 * np.exp(-(((xx - x2) ** 2) / (2 * 1.9**2) + ((yy - y2) ** 2) / (2 * 1.8**2)))
    return np.clip(frame, 0, 65535).astype(np.uint16)


def test_synthetic_pipeline_writes_outputs(tmp_path):
    config = AnalysisConfig()
    config.instrument.zenith_deg = 12.0
    config.statistics.block_duration_sec = 1.0
    config.statistics.min_valid_frames_per_block = 3
    config.statistics.min_valid_fraction_per_block = 0.5
    config.quality.max_center_jump_px = 20

    frames = [(idx, _two_spot_frame(idx), idx / 10.0) for idx in range(30)]
    result = analyze_frame_sequence(
        frames=frames,
        config=config,
        output_dir=tmp_path,
        time_source="fps",
        fps=10.0,
        saturation_level=65535,
    )

    assert result.summary["valid_frames"] > 20
    assert (tmp_path / "per_frame_fits.csv").exists()
    assert (tmp_path / "block_results.csv").exists()
    assert (tmp_path / "orientation_scan.csv").exists()
    assert (tmp_path / "orientation_diagnostics.csv").exists()
    assert (tmp_path / "rejection_summary.csv").exists()
    assert (tmp_path / "frame_distribution_summary.csv").exists()
    assert (tmp_path / "spot_assignment_debug.csv").exists()
    assert (tmp_path / "relative_motion_outliers.csv").exists()
    assert (tmp_path / "fwhm_outliers.csv").exists()
    assert (tmp_path / "roi_safety_report.csv").exists()
    assert (tmp_path / "peak_timeseries.png").exists()
    assert (tmp_path / "peak_histogram.png").exists()
    assert (tmp_path / "relative_motion_outliers.png").exists()
    assert (tmp_path / "roi_safety_timeseries.png").exists()
    assert (tmp_path / "summary.json").exists()
    with (tmp_path / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["number_of_valid_blocks"] >= 1
    assert summary["effective_pixel_scale_arcsec_per_px"] == 0.643
    assert summary["roi_size_px"] == 25
    assert summary["roi_fallback_size_px"] == 31
    assert summary["roi_half_size_px"] == 12.0
    assert summary["roi_safety_margin_px"] == 5.0
    assert summary["roi_safety_status"] == "safe"
    assert summary["roi_auto_shrunk"] is False
    assert summary["min_edge_margin_px"] is not None
    assert summary["edge_margin_all_min_px"] is not None
    assert summary["edge_margin_min_frame_index"] is not None
    assert summary["edge_margin_min_spot_id"] in {1, 2}
    assert summary["edge_margin_all_p01_px"] is not None
    assert summary["edge_margin_all_p05_px"] is not None
    assert summary["edge_margin_all_median_px"] is not None
    assert summary["edge_margin_reliable_p01_px"] is not None
    assert summary["edge_margin_reliable_p05_px"] is not None
    assert summary["edge_margin_reliable_median_px"] is not None
    assert summary["recommended_max_roi_size_px_from_min"] is not None
    assert summary["recommended_max_roi_size_px_from_p01"] is not None
    assert summary["recommended_max_roi_size_px_from_p05"] is not None
    assert summary["recommended_max_roi_size_px"] == summary[
        "recommended_max_roi_size_px_from_p05"
    ]
    assert "edge_margin_below_half_roi_count" in summary
    assert "edge_margin_below_required_count" in summary
    assert "edge_margin_absolute_outlier_detected" in summary
    assert summary["zenith_correction_factor"] < 1.0
    assert summary["seeing_observed_arcsec"] is not None
    assert summary["seeing_zenith_arcsec"] is not None
    assert summary["var_L_px2"] is not None
    assert summary["var_T_rad2"] is not None
    assert "reject_reason_counts" in summary
    assert "quality_flags" in summary
    assert summary["quality_flags"]["roi_too_close_to_edge"] is False
    assert summary["quality_flags"]["roi_safety_warning"] is False
    assert summary["result_reliability"] in {"good", "caution", "bad"}
    assert summary["saturation_level"] == 65535
    assert summary["saturated_frame_fraction"] == 0.0
    assert summary["saturated_core_frame_count"] == 0
    assert summary["saturated_core_frame_fraction"] == 0.0
    assert summary["saturated_roi_pixel_outlier_count"] == 0
    assert summary["peak_raw_max1"] is not None
    assert summary["peak_raw_max2"] is not None
    assert summary["peak_core_max1"] is not None
    assert summary["peak_core_max2"] is not None
    assert summary["valid_peak1_median"] is not None
    assert summary["sanity_check"]["seeing_from_r0_arcsec"] is not None
    with (tmp_path / "per_frame_fits.csv").open("r", encoding="utf-8") as handle:
        per_frame_rows = list(csv.DictReader(handle))
    first_row = per_frame_rows[0]
    assert "peak_raw_max1" in first_row
    assert "peak_raw_max2" in first_row
    assert "peak_core_max1" in first_row
    assert "peak_core_max2" in first_row
    assert "saturated_core_pixel_count1" in first_row
    assert "saturated_core_pixel_count2" in first_row
    assert "saturated_roi_pixel_count1" in first_row
    assert "saturated_roi_pixel_count2" in first_row
    assert "hot_pixel_or_roi_outlier1" in first_row
    assert "hot_pixel_or_roi_outlier2" in first_row
    assert summary["total_frames"] == len(per_frame_rows)
    assert summary["valid_frames"] == sum(row["frame_valid"] == "True" for row in per_frame_rows)
    with (tmp_path / "block_results.csv").open("r", encoding="utf-8") as handle:
        block_rows = list(csv.DictReader(handle))
    valid_block_rows = [
        row
        for row in block_rows
        if row["seeing_mean_zenith_arcsec"]
        and math.isfinite(float(row["seeing_mean_zenith_arcsec"]))
    ]
    assert summary["number_of_valid_blocks"] == len(valid_block_rows)
    if valid_block_rows:
        seeing_values = sorted(float(row["seeing_mean_zenith_arcsec"]) for row in valid_block_rows)
        r0_values = sorted(float(row["r0_mean_zenith_m"]) for row in valid_block_rows)
        assert summary["median_seeing_zenith_corrected_arcsec"] == pytest_approx(
            _median(seeing_values)
        )
        assert summary["seeing_zenith_arcsec"] == pytest_approx(_median(seeing_values))
        assert summary["median_r0_zenith_m"] == pytest_approx(_median(r0_values))
        assert summary["r0_zenith_m"] == pytest_approx(_median(r0_values))
    with (tmp_path / "roi_safety_report.csv").open("r", encoding="utf-8") as handle:
        roi_row = next(csv.DictReader(handle))
    assert "included_in_reliable_population" in roi_row
    assert "below_half_roi" in roi_row
    assert "below_required_margin" in roi_row
    assert "suspected_outlier" in roi_row


def test_relative_motion_outlier_is_rejected(tmp_path):
    config = AnalysisConfig()
    config.instrument.zenith_deg = 10.0
    config.statistics.block_duration_sec = 1.0
    config.statistics.min_valid_frames_per_block = 3
    config.statistics.min_valid_fraction_per_block = 0.5
    config.quality.reject_spot_tracking = False
    config.quality.reject_large_jump = False
    config.quality.reject_fwhm_outlier = False
    config.quality.max_relative_motion_deviation_px = 10.0

    frames = []
    for idx in range(24):
        y2_override = 25.0 if idx == 12 else None
        frames.append((idx, _two_spot_frame(idx, y2_override=y2_override), idx / 10.0))

    result = analyze_frame_sequence(
        frames=frames,
        config=config,
        output_dir=tmp_path,
        time_source="fps",
        fps=10.0,
        saturation_level=65535,
    )

    counts = result.summary["reject_reason_counts"]
    assert counts["bad_relative_motion"] >= 1
    assert result.summary["quality_flags"]["relative_motion_outliers_detected"]
    assert result.relative_motion_outlier_rows


def test_roi_hot_pixel_outlier_is_not_saturated(tmp_path):
    config = AnalysisConfig()
    config.instrument.zenith_deg = 10.0
    config.statistics.block_duration_sec = 1.0
    config.statistics.min_valid_frames_per_block = 1
    config.statistics.min_valid_fraction_per_block = 0.0
    config.quality.reject_fwhm_outlier = False
    config.quality.reject_bad_relative_motion = False

    frame = _two_spot_frame(0)
    frame[36, 14] = 65535
    result = analyze_frame_sequence(
        frames=[(0, frame, 0.0)],
        config=config,
        output_dir=tmp_path,
        time_source="fps",
        fps=10.0,
        saturation_level=65535,
    )

    counts = result.summary["reject_reason_counts"]
    assert counts["hot_pixel_or_roi_outlier"] == 1
    assert counts["saturated"] == 0
    assert result.summary["saturated_core_frame_count"] == 0
    assert result.summary["saturated_roi_pixel_outlier_count"] == 1


def test_roi_safety_check_can_be_disabled(tmp_path):
    config = AnalysisConfig()
    config.instrument.zenith_deg = 12.0
    config.roi.auto_safety_check = False
    config.statistics.block_duration_sec = 1.0
    config.statistics.min_valid_frames_per_block = 3
    config.statistics.min_valid_fraction_per_block = 0.5

    frames = [(idx, _two_spot_frame(idx), idx / 10.0) for idx in range(10)]
    result = analyze_frame_sequence(
        frames=frames,
        config=config,
        output_dir=tmp_path,
        time_source="fps",
        fps=10.0,
        saturation_level=65535,
    )

    assert result.summary["roi_safety_status"] == "not_checked"
    assert result.roi_safety_rows == []
    assert not (tmp_path / "roi_safety_report.csv").exists()
    assert not (tmp_path / "roi_safety_timeseries.png").exists()


def test_no_valid_blocks_reports_null_seeing_values(tmp_path):
    config = AnalysisConfig()
    config.instrument.zenith_deg = 12.0
    config.statistics.block_duration_sec = 1.0
    config.statistics.min_valid_frames_per_block = 999
    config.statistics.min_valid_fraction_per_block = 1.0
    config.quality.reject_fwhm_outlier = False
    config.quality.reject_bad_relative_motion = False

    frames = [(idx, _two_spot_frame(idx), idx / 10.0) for idx in range(8)]
    result = analyze_frame_sequence(
        frames=frames,
        config=config,
        output_dir=tmp_path,
        time_source="fps",
        fps=10.0,
        saturation_level=65535,
    )

    assert result.summary["number_of_valid_blocks"] == 0
    assert result.summary["median_seeing_zenith_corrected_arcsec"] is None
    assert result.summary["seeing_zenith_arcsec"] is None
    assert result.summary["mean_seeing_zenith_corrected_arcsec"] is None
    assert result.summary["r0_zenith_m"] is None
    assert result.summary["median_r0_zenith_m"] is None


def test_auto_shrink_roi_if_unsafe_for_ser(tmp_path):
    ser_path = tmp_path / "edge.ser"
    output_dir = tmp_path / "results" / "edge"
    manifest_path = output_dir / "input_manifest.json"
    _write_mono16_ser(ser_path, [_edge_spot_frame(idx) for idx in range(4)])

    config = AnalysisConfig()
    config.instrument.zenith_deg = 12.0
    config.detection.method = "local"
    config.roi.size_px = 31
    config.roi.fallback_size_px = 31
    config.roi.auto_shrink_if_unsafe = True
    config.statistics.min_valid_frames_per_block = 1
    config.statistics.min_valid_fraction_per_block = 0.0
    config.quality.reject_fwhm_outlier = False
    config.quality.reject_bad_relative_motion = False

    result = analyze_ser(
        input_path=ser_path,
        config=config,
        output_dir=output_dir,
        max_frames=4,
        show_progress=False,
        input_path_original=tmp_path,
        input_was_directory=True,
        ser_selection_mode="newest",
        companion_files_copied=["CameraSettings.txt"],
        output_root=tmp_path / "results",
        auto_output_name=True,
        input_manifest_path=manifest_path,
    )

    assert result.summary["roi_auto_shrunk"] is True
    assert result.summary["roi_original_size_px"] == 31
    assert result.summary["roi_original_fallback_size_px"] == 31
    assert result.summary["roi_size_px"] == 21
    assert result.summary["roi_fallback_size_px"] == 21
    assert result.summary["input_path_original"] == str(tmp_path)
    assert result.summary["input_path_resolved_ser"] == str(ser_path)
    assert result.summary["input_was_directory"] is True
    assert result.summary["ser_selection_mode"] == "newest"
    assert result.summary["companion_files_copied"] == ["CameraSettings.txt"]
    assert result.summary["input_ser_path"] == str(ser_path)
    assert result.summary["input_ser_filename"] == "edge.ser"
    assert result.summary["input_ser_stem"] == "edge"
    assert result.summary["output_root"] == str(tmp_path / "results")
    assert result.summary["output_dir"] == str(output_dir)
    assert result.summary["auto_output_name"] is True
    assert result.summary["input_manifest_path"] == str(manifest_path)


def _median(values):
    count = len(values)
    middle = count // 2
    if count % 2:
        return values[middle]
    return 0.5 * (values[middle - 1] + values[middle])


def pytest_approx(*args, **kwargs):
    import pytest

    return pytest.approx(*args, **kwargs)
