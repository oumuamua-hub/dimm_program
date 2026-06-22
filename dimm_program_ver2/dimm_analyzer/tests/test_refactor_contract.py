import inspect
from dataclasses import fields

import numpy as np

from dimm_analyzer import cli as cli_module
from dimm_analyzer import pipeline as pipeline_module
from dimm_analyzer.config import AnalysisConfig
from dimm_analyzer.models import AnalysisResult, BlockResult, FitResult, FrameResult
from dimm_analyzer.pipeline import analyze_frame_sequence, analyze_ser
from dimm_analyzer.report import PER_FRAME_COLUMNS

ANALYZE_SER_PARAMETERS = (
    "input_path",
    "config",
    "output_dir",
    "start_frame",
    "end_frame",
    "max_frames",
    "preview",
    "show_progress",
    "input_path_original",
    "input_was_directory",
    "ser_selection_mode",
    "companion_files_copied",
    "output_root",
    "auto_output_name",
    "input_manifest_path",
)

ANALYZE_FRAME_SEQUENCE_PARAMETERS = (
    "frames",
    "config",
    "output_dir",
    "input_path",
    "ser_metadata",
    "dark_frame",
    "saturation_level",
    "time_source",
    "fps",
    "show_progress",
    "total_frames",
    "roi_safety_rows",
    "roi_safety_summary",
    "roi_original_size_px",
    "roi_original_fallback_size_px",
    "roi_auto_shrunk",
    "input_path_original",
    "input_path_resolved_ser",
    "input_was_directory",
    "ser_selection_mode",
    "companion_files_copied",
    "output_root",
    "auto_output_name",
    "input_manifest_path",
)

FIT_RESULT_FIELDS = (
    "success",
    "failure_reason",
    "x0_global",
    "y0_global",
    "x0_roi",
    "y0_roi",
    "background",
    "amplitude",
    "sigma_x",
    "sigma_y",
    "fwhm_x",
    "fwhm_y",
    "fwhm_mean",
    "residual_rms",
    "peak",
    "peak_raw_max",
    "peak_core_max",
    "saturated_core_pixel_count",
    "saturated_roi_pixel_count",
    "hot_pixel_or_roi_outlier",
    "flux",
    "nfev",
)

PIPELINE_RELOCATED_FUNCTIONS = (
    "_assign_spot_ids",
    "_fit_source",
    "_large_center_jump",
    "_process_frame",
    "_distribution_row",
    "_finite_or_nan",
    "_frame_in_reliable_roi_population",
    "_frame_metric_value",
    "_frame_spot_position",
    "_is_relative_outlier",
    "_none_if_nan",
    "_reference_series",
    "apply_fwhm_outlier_filter",
    "apply_relative_motion_filter",
    "build_frame_distribution_rows",
    "build_orientation_scan_rows",
    "build_rejection_summary_rows",
    "build_roi_safety_points_from_frame_results",
    "build_spot_assignment_rows",
    "reject_reason_counts",
    "_array_stat",
    "_estimated_fps",
    "_is_finite_number",
    "_nanmean",
    "_nanmedian",
    "_nanstd",
    "_optional_close",
    "_orientation_mismatch_score",
    "_representative_block_values",
    "_result_reliability",
    "_sanity_check_from_r0",
    "_saturation_summary",
    "_valid_blocks_with_finite_seeing",
    "_valid_rejected_medians",
    "build_summary",
    "validate_result_consistency",
)

CLI_RELOCATED_FUNCTIONS = (
    "_batch_error_row",
    "_batch_manifest_entry",
    "_batch_success_row",
    "_comparison_row",
    "_next_available_output_path",
    "_resolve_batch_output_root",
    "_resolve_collision",
    "_resolve_output_target",
    "_safe_output_stem",
    "_ser_output_dirs",
)


def test_pipeline_public_signatures_are_stable():
    assert tuple(inspect.signature(analyze_ser).parameters) == ANALYZE_SER_PARAMETERS
    assert (
        tuple(inspect.signature(analyze_frame_sequence).parameters)
        == ANALYZE_FRAME_SEQUENCE_PARAMETERS
    )


def test_fit_result_field_order_is_stable():
    assert tuple(field.name for field in fields(FitResult)) == FIT_RESULT_FIELDS


def test_result_models_remain_dataclasses():
    assert fields(FrameResult)
    assert fields(BlockResult)
    assert fields(AnalysisResult)


def test_per_frame_schema_matches_fit_diagnostics():
    for spot_id in (1, 2):
        assert f"peak_raw_max{spot_id}" in PER_FRAME_COLUMNS
        assert f"peak_core_max{spot_id}" in PER_FRAME_COLUMNS
        assert f"saturated_core_pixel_count{spot_id}" in PER_FRAME_COLUMNS
        assert f"saturated_roi_pixel_count{spot_id}" in PER_FRAME_COLUMNS
        assert f"hot_pixel_or_roi_outlier{spot_id}" in PER_FRAME_COLUMNS


def test_pipeline_relocated_functions_keep_pipeline_identity():
    for name in PIPELINE_RELOCATED_FUNCTIONS:
        function = getattr(pipeline_module, name)
        assert function.__module__ == pipeline_module.__name__, name


def test_pipeline_relocated_functions_use_pipeline_globals():
    for name in PIPELINE_RELOCATED_FUNCTIONS:
        function = getattr(pipeline_module, name)
        assert function.__globals__ is vars(pipeline_module), name


def test_pipeline_detect_sources_remains_a_process_frame_seam(monkeypatch):
    calls = []

    def fake_detect_sources(frame, config):
        calls.append((frame, config))
        return []

    config = AnalysisConfig()
    frame = np.zeros((16, 16), dtype=float)
    monkeypatch.setattr(pipeline_module, "detect_sources", fake_detect_sources)

    pipeline_module._process_frame(
        frame_index=0,
        time_sec=None,
        frame=frame,
        config=config,
        saturation_level=None,
        previous_centers=None,
    )

    assert calls == [(frame, config.detection)]


def test_cli_relocated_functions_keep_cli_identity():
    for name in CLI_RELOCATED_FUNCTIONS:
        function = getattr(cli_module, name)
        assert function.__module__ == cli_module.__name__, name


def test_cli_relocated_functions_use_cli_globals():
    for name in CLI_RELOCATED_FUNCTIONS:
        function = getattr(cli_module, name)
        assert function.__globals__ is vars(cli_module), name


def test_output_target_keeps_cli_identity():
    assert cli_module.OutputTarget.__module__ == cli_module.__name__
