import inspect
from dataclasses import fields

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
