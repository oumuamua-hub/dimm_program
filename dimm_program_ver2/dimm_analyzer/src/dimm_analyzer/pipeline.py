"""End-to-end DIMM analysis pipeline."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from . import _diagnostics, _frame_processing
from .config import (
    AnalysisConfig,
    config_warnings,
    effective_pixel_scale_arcsec_per_px,
    model_to_dict,
    validate_config,
)
from .detection import detect_sources, select_spot_pair
from .dimm_math import compute_dimm_block, seeing_from_r0_arcsec, zenith_correction_factor
from .exceptions import PipelineError, SERFormatError
from .models import (
    AnalysisResult,
    BlockResult,
    FrameResult,
    OrientationResult,
    SERMetadata,
)
from .orientation import estimate_orientation, rotate_motion
from .plotting import save_plots
from .quality import ratio
from .report import (
    write_block_csv,
    write_frame_distribution_csv,
    write_fwhm_outliers_csv,
    write_orientation_scan_csv,
    write_per_frame_csv,
    write_rejection_summary_csv,
    write_relative_motion_outliers_csv,
    write_roi_safety_report_csv,
    write_spot_assignment_debug_csv,
    write_summary_json,
)
from .roi_safety import (
    build_roi_safety_point,
    finalize_roi_safety_rows,
    shrink_roi_sizes_for_safety,
    summarize_roi_safety,
)
from .ser_reader import SERReader

FrameInput = Tuple[int, np.ndarray, Optional[float]]
PreviousCenters = _frame_processing.PreviousCenters
ProcessFrameReturn = _frame_processing.ProcessFrameReturn
_assign_spot_ids = _frame_processing._assign_spot_ids
_fit_source = _frame_processing._fit_source
_large_center_jump = _frame_processing._large_center_jump
_process_frame = _frame_processing._process_frame

FRAME_DIAGNOSTIC_METRICS = _diagnostics.FRAME_DIAGNOSTIC_METRICS
REJECT_REASON_ORDER = _diagnostics.REJECT_REASON_ORDER
_distribution_row = _diagnostics._distribution_row
_finite_or_nan = _diagnostics._finite_or_nan
_frame_in_reliable_roi_population = _diagnostics._frame_in_reliable_roi_population
_frame_metric_value = _diagnostics._frame_metric_value
_frame_spot_position = _diagnostics._frame_spot_position
_is_relative_outlier = _diagnostics._is_relative_outlier
_none_if_nan = _diagnostics._none_if_nan
_reference_series = _diagnostics._reference_series
apply_fwhm_outlier_filter = _diagnostics.apply_fwhm_outlier_filter
apply_relative_motion_filter = _diagnostics.apply_relative_motion_filter
build_frame_distribution_rows = _diagnostics.build_frame_distribution_rows
build_orientation_scan_rows = _diagnostics.build_orientation_scan_rows
build_rejection_summary_rows = _diagnostics.build_rejection_summary_rows
build_roi_safety_points_from_frame_results = (
    _diagnostics.build_roi_safety_points_from_frame_results
)
build_spot_assignment_rows = _diagnostics.build_spot_assignment_rows
reject_reason_counts = _diagnostics.reject_reason_counts


def analyze_ser(
    *,
    input_path: Path,
    config: AnalysisConfig,
    output_dir: Path,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    max_frames: Optional[int] = None,
    preview: bool = False,
    show_progress: bool = True,
    input_path_original: Optional[Path] = None,
    input_was_directory: bool = False,
    ser_selection_mode: Optional[str] = None,
    companion_files_copied: Optional[List[str]] = None,
    output_root: Optional[Path] = None,
    auto_output_name: Optional[bool] = None,
    input_manifest_path: Optional[Path] = None,
) -> AnalysisResult:
    validate_config(config)
    if preview:
        max_frames = 500 if max_frames is None else min(max_frames, 500)

    with SERReader(input_path, reject_color=config.input.reject_color_ser) as reader:
        time_source = _resolve_time_source(reader, config)
        fps = config.timing.fps if time_source == "fps" else None
        dark_frame = None
        if config.calibration.dark_path is not None or config.calibration.subtract_dark:
            dark_path = config.calibration.dark_path
            if dark_path is None:
                raise PipelineError(
                    "calibration.subtract_dark が true ですが calibration.dark_path が null です。"
                )
            dark_frame = build_median_dark_frame(dark_path, reader.metadata, config)

        total = _selected_frame_count(
            reader.metadata.frame_count,
            start_frame,
            end_frame,
            max_frames,
        )

        roi_original_size_px = config.roi.size_px
        roi_original_fallback_size_px = config.roi.fallback_size_px
        roi_auto_shrunk = False
        roi_safety_rows: Optional[List[Dict[str, Any]]] = None
        roi_safety_summary: Optional[Dict[str, Any]] = None

        if config.roi.auto_safety_check or config.roi.auto_shrink_if_unsafe:
            roi_points = _collect_ser_roi_safety_points(
                reader=reader,
                config=config,
                start_frame=start_frame,
                end_frame=end_frame,
                max_frames=max_frames,
                time_source=time_source,
                fps=fps,
                dark_frame=dark_frame,
            )
            requested_rows = finalize_roi_safety_rows(
                roi_points,
                roi_size_px=config.roi.size_px,
                safety_margin_px=config.roi.safety_margin_px,
            )
            requested_summary, _ = summarize_roi_safety(
                requested_rows,
                roi_size_px=config.roi.size_px,
                roi_fallback_size_px=config.roi.fallback_size_px,
                safety_margin_px=config.roi.safety_margin_px,
                max_allowed_size_px=config.roi.max_allowed_size_px,
                enabled=config.roi.auto_safety_check,
                roi_auto_shrunk=False,
                roi_original_size_px=roi_original_size_px,
                roi_original_fallback_size_px=roi_original_fallback_size_px,
                safety_reference_percentile=config.roi.safety_reference_percentile,
                unsafe_fraction_threshold=config.roi.unsafe_fraction_threshold,
                warning_fraction_threshold=config.roi.warning_fraction_threshold,
                total_frame_count=total,
                roi_out_of_bounds_count=0,
            )
            if (
                config.roi.auto_shrink_if_unsafe
                and requested_summary.get("roi_safety_status") == "unsafe"
            ):
                new_size, new_fallback_size = shrink_roi_sizes_for_safety(
                    requested_size_px=config.roi.size_px,
                    requested_fallback_size_px=config.roi.fallback_size_px,
                    recommended_size_px=requested_summary.get("recommended_max_roi_size_px"),
                    min_allowed_size_px=config.roi.min_allowed_size_px,
                    max_allowed_size_px=config.roi.max_allowed_size_px,
                )
                roi_auto_shrunk = (
                    new_size != config.roi.size_px
                    or new_fallback_size != config.roi.fallback_size_px
                )
                config.roi.size_px = new_size
                config.roi.fallback_size_px = new_fallback_size
                validate_config(config)

            roi_safety_rows = None
            roi_safety_summary = None

        def frame_iter() -> Iterator[FrameInput]:
            for frame_index, frame in reader.iter_frames(
                start=start_frame,
                end=end_frame,
                max_frames=max_frames,
            ):
                time_sec = _frame_time_seconds(reader, frame_index, time_source, fps)
                yield frame_index, frame, time_sec

        saturation_level = float(2**reader.metadata.pixel_depth - 1)
        return analyze_frame_sequence(
            frames=frame_iter(),
            config=config,
            output_dir=output_dir,
            input_path=input_path,
            ser_metadata=reader.metadata,
            dark_frame=dark_frame,
            saturation_level=saturation_level,
            time_source=time_source,
            fps=fps,
            show_progress=show_progress,
            total_frames=total,
            roi_safety_rows=roi_safety_rows,
            roi_safety_summary=roi_safety_summary,
            roi_original_size_px=roi_original_size_px,
            roi_original_fallback_size_px=roi_original_fallback_size_px,
            roi_auto_shrunk=roi_auto_shrunk,
            input_path_original=input_path_original,
            input_path_resolved_ser=input_path,
            input_was_directory=input_was_directory,
            ser_selection_mode=ser_selection_mode,
            companion_files_copied=companion_files_copied,
            output_root=output_root,
            auto_output_name=auto_output_name,
            input_manifest_path=input_manifest_path,
        )


def build_median_dark_frame(
    dark_path: Path,
    science_metadata: SERMetadata,
    config: AnalysisConfig,
) -> np.ndarray:
    with SERReader(Path(dark_path), reject_color=config.input.reject_color_ser) as dark_reader:
        dark_meta = dark_reader.metadata
        if (
            dark_meta.width != science_metadata.width
            or dark_meta.height != science_metadata.height
            or dark_meta.pixel_depth != science_metadata.pixel_depth
        ):
            raise SERFormatError(
                "Dark SER の画像サイズと bit depth は science SER と一致が必要です。"
            )
        frames: List[np.ndarray] = []
        for _, frame in dark_reader.iter_frames(max_frames=config.calibration.max_dark_frames):
            frames.append(frame.astype(float))
        if not frames:
            raise SERFormatError("Dark SER に読み取り可能な frame がありません。")
    return np.median(np.stack(frames, axis=0), axis=0)


def _collect_ser_roi_safety_points(
    *,
    reader: SERReader,
    config: AnalysisConfig,
    start_frame: int,
    end_frame: Optional[int],
    max_frames: Optional[int],
    time_source: str,
    fps: Optional[float],
    dark_frame: Optional[np.ndarray],
) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    width = reader.metadata.width
    height = reader.metadata.height
    for frame_index, frame in reader.iter_frames(
        start=start_frame,
        end=end_frame,
        max_frames=max_frames,
    ):
        processed = frame.astype(float, copy=False)
        if dark_frame is not None:
            processed = processed - dark_frame
        sources = detect_sources(processed, config.detection)
        pair = select_spot_pair(
            sources,
            expected_separation_px=config.spots.expected_separation_px,
            separation_tolerance_px=config.spots.separation_tolerance_px,
        )
        if pair is None:
            continue
        time_sec = _frame_time_seconds(reader, frame_index, time_source, fps)
        for spot_id, source in enumerate(pair, start=1):
            point = build_roi_safety_point(
                frame_index=frame_index,
                time_sec=time_sec,
                spot_id=spot_id,
                x=source.x,
                y=source.y,
                width=width,
                height=height,
            )
            if point is not None:
                points.append(point)
    return points


def analyze_frame_sequence(
    *,
    frames: Iterable[FrameInput],
    config: AnalysisConfig,
    output_dir: Optional[Path] = None,
    input_path: Optional[Path] = None,
    ser_metadata: Optional[SERMetadata] = None,
    dark_frame: Optional[np.ndarray] = None,
    saturation_level: Optional[float] = None,
    time_source: str = "frame_index",
    fps: Optional[float] = None,
    show_progress: bool = False,
    total_frames: Optional[int] = None,
    roi_safety_rows: Optional[List[Dict[str, Any]]] = None,
    roi_safety_summary: Optional[Dict[str, Any]] = None,
    roi_original_size_px: Optional[int] = None,
    roi_original_fallback_size_px: Optional[int] = None,
    roi_auto_shrunk: bool = False,
    input_path_original: Optional[Path] = None,
    input_path_resolved_ser: Optional[Path] = None,
    input_was_directory: bool = False,
    ser_selection_mode: Optional[str] = None,
    companion_files_copied: Optional[List[str]] = None,
    output_root: Optional[Path] = None,
    auto_output_name: Optional[bool] = None,
    input_manifest_path: Optional[Path] = None,
) -> AnalysisResult:
    validate_config(config)
    warnings_out = config_warnings(config)
    if dark_frame is not None:
        warnings_out.append(
            "ダーク補正を適用しました。差し引き後の負値は float のまま保持します。"
        )
    if time_source == "frame_index":
        warnings_out.append(
            "物理的な時刻情報がないため、frame index で block と plot を作成します。"
        )
    saturation_level = _effective_saturation_level(config, saturation_level)

    frame_results: List[FrameResult] = []
    previous_centers: Optional[Tuple[float, float, float, float]] = None
    example_success_roi: Optional[np.ndarray] = None
    example_failed_roi: Optional[np.ndarray] = None

    for frame_index, frame, time_sec in _with_progress(
        frames, total=total_frames, enabled=show_progress
    ):
        processed = frame.astype(float, copy=False)
        if dark_frame is not None:
            processed = processed - dark_frame
        result, previous_centers, success_roi, failed_roi = _process_frame(
            frame_index=frame_index,
            time_sec=time_sec,
            frame=processed,
            config=config,
            saturation_level=saturation_level,
            previous_centers=previous_centers,
        )
        if example_success_roi is None and success_roi is not None:
            example_success_roi = success_roi
        if example_failed_roi is None and failed_roi is not None:
            example_failed_roi = failed_roi
        frame_results.append(result)

    # Filter order is observable: relative-motion references use the frames left by FWHM filtering.
    fwhm_outlier_rows = apply_fwhm_outlier_filter(frame_results, config)
    relative_motion_outlier_rows = apply_relative_motion_filter(frame_results, config)
    spot_assignment_rows = build_spot_assignment_rows(frame_results)

    pixel_scale = effective_pixel_scale_arcsec_per_px(config)
    valid_positions = [idx for idx, frame in enumerate(frame_results) if frame.frame_valid]
    valid_dx = np.asarray([frame_results[idx].dx_px for idx in valid_positions], dtype=float)
    valid_dy = np.asarray([frame_results[idx].dy_px for idx in valid_positions], dtype=float)
    valid_block_masks = _make_valid_block_masks(frame_results, valid_positions, config, time_source)
    orientation_scan_rows = build_orientation_scan_rows(
        dx=valid_dx,
        dy=valid_dy,
        config=config,
        pixel_scale_arcsec_per_px=pixel_scale,
    )
    orientation = estimate_orientation(
        mode=config.orientation.mode,
        dx=valid_dx,
        dy=valid_dy,
        block_masks=valid_block_masks,
        orientation_config=config.orientation,
        instrument_config=config.instrument,
        pixel_scale_arcsec_per_px=pixel_scale,
        warnings_out=warnings_out,
    )
    if orientation.angle_deg is not None and valid_positions:
        d_l, d_t = rotate_motion(valid_dx, valid_dy, orientation.angle_deg)
        for pos, longitudinal, transverse in zip(valid_positions, d_l, d_t):
            frame_results[pos].dL_px = float(longitudinal)
            frame_results[pos].dT_px = float(transverse)

    block_results = build_block_results(
        frame_results=frame_results,
        config=config,
        pixel_scale_arcsec_per_px=pixel_scale,
        time_source=time_source,
    )
    rejection_summary_rows = build_rejection_summary_rows(frame_results)
    frame_distribution_rows = build_frame_distribution_rows(frame_results)
    reject_counts = reject_reason_counts(frame_results)

    if config.roi.auto_safety_check:
        if roi_safety_rows is None:
            roi_points = build_roi_safety_points_from_frame_results(frame_results)
            roi_safety_rows = finalize_roi_safety_rows(
                roi_points,
                roi_size_px=config.roi.size_px,
                safety_margin_px=config.roi.safety_margin_px,
            )
        if roi_safety_summary is None:
            roi_safety_summary, roi_warnings = summarize_roi_safety(
                roi_safety_rows,
                roi_size_px=config.roi.size_px,
                roi_fallback_size_px=config.roi.fallback_size_px,
                safety_margin_px=config.roi.safety_margin_px,
                max_allowed_size_px=config.roi.max_allowed_size_px,
                enabled=True,
                roi_auto_shrunk=roi_auto_shrunk,
                roi_original_size_px=roi_original_size_px,
                roi_original_fallback_size_px=roi_original_fallback_size_px,
                safety_reference_percentile=config.roi.safety_reference_percentile,
                unsafe_fraction_threshold=config.roi.unsafe_fraction_threshold,
                warning_fraction_threshold=config.roi.warning_fraction_threshold,
                total_frame_count=len(frame_results),
                roi_out_of_bounds_count=int(reject_counts.get("roi_out_of_bounds", 0)),
            )
            warnings_out.extend(roi_warnings)
        else:
            roi_safety_summary, roi_warnings = summarize_roi_safety(
                roi_safety_rows,
                roi_size_px=config.roi.size_px,
                roi_fallback_size_px=config.roi.fallback_size_px,
                safety_margin_px=config.roi.safety_margin_px,
                max_allowed_size_px=config.roi.max_allowed_size_px,
                enabled=True,
                roi_auto_shrunk=bool(roi_safety_summary.get("roi_auto_shrunk")),
                roi_original_size_px=roi_safety_summary.get("roi_original_size_px"),
                roi_original_fallback_size_px=roi_safety_summary.get(
                    "roi_original_fallback_size_px"
                ),
                safety_reference_percentile=config.roi.safety_reference_percentile,
                unsafe_fraction_threshold=config.roi.unsafe_fraction_threshold,
                warning_fraction_threshold=config.roi.warning_fraction_threshold,
                total_frame_count=len(frame_results),
                roi_out_of_bounds_count=int(reject_counts.get("roi_out_of_bounds", 0)),
            )
            warnings_out.extend(roi_warnings)
    else:
        roi_safety_rows = []
        roi_safety_summary, _ = summarize_roi_safety(
            [],
            roi_size_px=config.roi.size_px,
            roi_fallback_size_px=config.roi.fallback_size_px,
            safety_margin_px=config.roi.safety_margin_px,
            max_allowed_size_px=config.roi.max_allowed_size_px,
            enabled=False,
            roi_original_size_px=roi_original_size_px,
            roi_original_fallback_size_px=roi_original_fallback_size_px,
            safety_reference_percentile=config.roi.safety_reference_percentile,
            unsafe_fraction_threshold=config.roi.unsafe_fraction_threshold,
            warning_fraction_threshold=config.roi.warning_fraction_threshold,
            total_frame_count=len(frame_results),
            roi_out_of_bounds_count=int(reject_counts.get("roi_out_of_bounds", 0)),
        )

    summary = build_summary(
        frame_results=frame_results,
        block_results=block_results,
        config=config,
        pixel_scale_arcsec_per_px=pixel_scale,
        orientation=orientation,
        warnings_out=warnings_out,
        input_path=input_path,
        ser_metadata=ser_metadata,
        time_source=time_source,
        fps=fps,
        saturation_level=saturation_level,
        reject_reason_counts=reject_counts,
        frame_distribution_rows=frame_distribution_rows,
        orientation_scan_rows=orientation_scan_rows,
        roi_safety_summary=roi_safety_summary,
        input_path_original=input_path_original,
        input_path_resolved_ser=input_path_resolved_ser,
        input_was_directory=input_was_directory,
        ser_selection_mode=ser_selection_mode,
        companion_files_copied=companion_files_copied,
        output_root=output_root,
        output_dir=output_dir,
        auto_output_name=auto_output_name,
        input_manifest_path=input_manifest_path,
    )

    analysis_result = AnalysisResult(
        frame_results=frame_results,
        block_results=block_results,
        summary=summary,
        warnings=warnings_out,
        orientation=orientation,
        orientation_scan_rows=orientation_scan_rows,
        rejection_summary_rows=rejection_summary_rows,
        frame_distribution_rows=frame_distribution_rows,
        spot_assignment_rows=spot_assignment_rows,
        relative_motion_outlier_rows=relative_motion_outlier_rows,
        fwhm_outlier_rows=fwhm_outlier_rows,
        roi_safety_rows=roi_safety_rows,
        example_success_roi=example_success_roi,
        example_failed_roi=example_failed_roi,
    )
    validate_result_consistency(analysis_result)
    if output_dir is not None:
        write_outputs(analysis_result, config, output_dir)
    return analysis_result


def build_block_results(
    *,
    frame_results: Sequence[FrameResult],
    config: AnalysisConfig,
    pixel_scale_arcsec_per_px: float,
    time_source: str,
) -> List[BlockResult]:
    block_groups = _group_frame_positions(frame_results, config, time_source)
    blocks: List[BlockResult] = []
    for block_index, positions in enumerate(block_groups):
        frames = [frame_results[pos] for pos in positions]
        valid = [frame for frame in frames if frame.frame_valid and frame.dL_px is not None]
        n_total = len(frames)
        n_valid = len(valid)
        fit_attempts = sum(frame.fit_attempts for frame in frames)
        spot_successes = sum(int(frame.fit1.success) + int(frame.fit2.success) for frame in frames)
        block = BlockResult(
            block_index=block_index,
            time_start_sec=_min_optional([frame.time_sec for frame in frames]),
            time_end_sec=_max_optional([frame.time_sec for frame in frames]),
            frame_start=min(frame.frame_index for frame in frames),
            frame_end=max(frame.frame_index for frame in frames),
            n_total=n_total,
            n_valid=n_valid,
            frame_fit_success_rate=ratio(n_valid, n_total),
            spot_fit_success_rate=ratio(spot_successes, fit_attempts),
        )
        enough_frames = n_valid >= config.statistics.min_valid_frames_per_block
        enough_fraction = ratio(n_valid, n_total) >= config.statistics.min_valid_fraction_per_block
        if enough_frames and enough_fraction and n_valid >= 2:
            d_l = np.asarray([frame.dL_px for frame in valid], dtype=float)
            d_t = np.asarray([frame.dT_px for frame in valid], dtype=float)
            var_l = float(np.var(d_l, ddof=1))
            var_t = float(np.var(d_t, ddof=1))
            try:
                values = compute_dimm_block(
                    var_L_px2=var_l,
                    var_T_px2=var_t,
                    pixel_scale_arcsec_per_px=pixel_scale_arcsec_per_px,
                    aperture_diameter_m=config.instrument.aperture_diameter_m,
                    baseline_m=config.instrument.baseline_m,
                    wavelength_m=config.instrument.wavelength_m,
                    zenith_deg=config.instrument.zenith_deg,
                    zenith_correction=config.instrument.zenith_correction,
                )
            except Exception:
                values = {}
            if values:
                block.valid = True
                block.var_L_px2 = var_l
                block.var_T_px2 = var_t
                for key, value in values.items():
                    setattr(block, key, float(value))
        blocks.append(block)
    return blocks


def write_outputs(result: AnalysisResult, config: AnalysisConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.output.save_csv:
        write_per_frame_csv(output_dir / "per_frame_fits.csv", result.frame_results)
        write_block_csv(output_dir / "block_results.csv", result.block_results)
        write_orientation_scan_csv(
            output_dir / "orientation_scan.csv", result.orientation_scan_rows
        )
        write_orientation_scan_csv(
            output_dir / "orientation_diagnostics.csv", result.orientation_scan_rows
        )
        write_rejection_summary_csv(
            output_dir / "rejection_summary.csv", result.rejection_summary_rows
        )
        write_frame_distribution_csv(
            output_dir / "frame_distribution_summary.csv",
            result.frame_distribution_rows,
        )
        write_spot_assignment_debug_csv(
            output_dir / "spot_assignment_debug.csv", result.spot_assignment_rows
        )
        write_relative_motion_outliers_csv(
            output_dir / "relative_motion_outliers.csv",
            result.relative_motion_outlier_rows,
        )
        write_fwhm_outliers_csv(output_dir / "fwhm_outliers.csv", result.fwhm_outlier_rows)
        if result.summary.get("roi_safety_status") != "not_checked":
            write_roi_safety_report_csv(
                output_dir / "roi_safety_report.csv",
                result.roi_safety_rows,
            )
    if config.output.save_json:
        write_summary_json(output_dir / "summary.json", result.summary)
    if config.output.save_plots:
        save_plots(result, output_dir)


def validate_result_consistency(result: AnalysisResult) -> None:
    """Validate summary values against the tabular data sources before writing files."""

    summary = result.summary
    total_frames = len(result.frame_results)
    valid_frames = sum(1 for frame in result.frame_results if frame.frame_valid)
    valid_blocks = _valid_blocks_with_finite_seeing(result.block_results)
    if summary.get("total_frames") != total_frames:
        raise PipelineError("summary.total_frames が per_frame_fits.csv 相当の行数と一致しません。")
    if summary.get("valid_frames") != valid_frames:
        raise PipelineError("summary.valid_frames が frame_valid=True の行数と一致しません。")
    if summary.get("number_of_valid_blocks") != len(valid_blocks):
        raise PipelineError(
            "summary.number_of_valid_blocks が finite seeing block 数と一致しません。"
        )
    if not valid_blocks:
        for key in (
            "median_seeing_zenith_corrected_arcsec",
            "seeing_zenith_arcsec",
            "mean_seeing_zenith_corrected_arcsec",
            "std_seeing_zenith_corrected_arcsec",
            "r0_zenith_m",
            "median_r0_zenith_m",
        ):
            if summary.get(key) is not None:
                raise PipelineError(f"valid block が0なのに summary.{key} が null ではありません。")
        return

    expected = {
        "median_seeing_zenith_corrected_arcsec": _nanmedian(
            [block.seeing_mean_zenith_arcsec for block in valid_blocks]
        ),
        "seeing_zenith_arcsec": _nanmedian(
            [block.seeing_mean_zenith_arcsec for block in valid_blocks]
        ),
        "median_r0_zenith_m": _nanmedian([block.r0_mean_zenith_m for block in valid_blocks]),
        "r0_zenith_m": _nanmedian([block.r0_mean_zenith_m for block in valid_blocks]),
    }
    for key, value in expected.items():
        if not _optional_close(summary.get(key), value):
            raise PipelineError(f"summary.{key} が valid block から再計算した値と一致しません。")


def _saturation_summary(
    *,
    frame_results: Sequence[FrameResult],
    saturation_level: Optional[float],
    saturation_margin: float,
    saturation_core_min_pixels: int,
) -> Dict[str, Any]:
    threshold = saturation_level - saturation_margin if saturation_level is not None else None
    peaks1 = np.asarray(
        [_finite_or_nan(frame.fit1.peak) for frame in frame_results],
        dtype=float,
    )
    peaks2 = np.asarray(
        [_finite_or_nan(frame.fit2.peak) for frame in frame_results],
        dtype=float,
    )
    peak_raw_max1 = np.asarray(
        [_finite_or_nan(frame.fit1.peak_raw_max) for frame in frame_results],
        dtype=float,
    )
    peak_raw_max2 = np.asarray(
        [_finite_or_nan(frame.fit2.peak_raw_max) for frame in frame_results],
        dtype=float,
    )
    peak_core_max1 = np.asarray(
        [_finite_or_nan(frame.fit1.peak_core_max) for frame in frame_results],
        dtype=float,
    )
    peak_core_max2 = np.asarray(
        [_finite_or_nan(frame.fit2.peak_core_max) for frame in frame_results],
        dtype=float,
    )
    saturated_core_count = 0
    saturated_roi_pixel_outlier_count = 0
    for frame in frame_results:
        core1 = frame.fit1.saturated_core_pixel_count or 0
        core2 = frame.fit2.saturated_core_pixel_count or 0
        if core1 >= saturation_core_min_pixels or core2 >= saturation_core_min_pixels:
            saturated_core_count += 1
        elif frame.fit1.hot_pixel_or_roi_outlier or frame.fit2.hot_pixel_or_roi_outlier:
            saturated_roi_pixel_outlier_count += 1
    return {
        "saturation_level": saturation_level,
        "saturation_margin": saturation_margin,
        "saturation_threshold": threshold,
        "saturated_frame_count": saturated_core_count,
        "saturated_frame_fraction": ratio(saturated_core_count, len(frame_results)),
        "saturated_core_frame_count": saturated_core_count,
        "saturated_core_frame_fraction": ratio(saturated_core_count, len(frame_results)),
        "saturated_roi_pixel_outlier_count": saturated_roi_pixel_outlier_count,
        "peak_raw_max1": _array_stat(peak_raw_max1, "max"),
        "peak_raw_max2": _array_stat(peak_raw_max2, "max"),
        "peak_core_max1": _array_stat(peak_core_max1, "max"),
        "peak_core_max2": _array_stat(peak_core_max2, "max"),
        "peak1_median": _array_stat(peaks1, "median"),
        "peak2_median": _array_stat(peaks2, "median"),
        "peak1_p95": _array_stat(peaks1, "p95"),
        "peak2_p95": _array_stat(peaks2, "p95"),
        "peak1_max": _array_stat(peaks1, "max"),
        "peak2_max": _array_stat(peaks2, "max"),
    }


def _valid_rejected_medians(frame_results: Sequence[FrameResult]) -> Dict[str, Any]:
    groups = {
        "valid": [frame for frame in frame_results if frame.frame_valid],
        "rejected": [frame for frame in frame_results if not frame.frame_valid],
    }
    output: Dict[str, Any] = {}
    metrics = {
        "peak1": "peak1",
        "peak2": "peak2",
        "fwhm1": "fwhm_mean1",
        "fwhm2": "fwhm_mean2",
        "separation": "separation_px",
    }
    for group_name, frames in groups.items():
        for output_name, metric in metrics.items():
            values = np.asarray(
                [
                    value
                    for value in (_frame_metric_value(frame, metric) for frame in frames)
                    if value is not None and np.isfinite(value)
                ],
                dtype=float,
            )
            output[f"{group_name}_{output_name}_median"] = (
                float(np.median(values)) if values.size else None
            )
    return output


def _orientation_mismatch_score(rows: Sequence[Dict[str, Any]]) -> Optional[float]:
    values = [
        row.get("mismatch_abs_log_ratio")
        for row in rows
        if row.get("mismatch_abs_log_ratio") is not None
    ]
    if not values:
        return None
    return float(min(values))


def _result_reliability(
    *,
    valid_block_count: int,
    frame_fit_success_rate: float,
    saturated_frame_fraction: float,
    orientation_reliable: bool,
    relative_motion_outlier_count: int,
    roi_safety_status: str,
    total_frames: int,
) -> str:
    relative_bad = relative_motion_outlier_count > max(5, int(0.01 * total_frames))
    if roi_safety_status == "unsafe":
        return "bad"
    if (
        valid_block_count < 3
        or frame_fit_success_rate < 0.7
        or saturated_frame_fraction > 0.05
        or relative_bad
    ):
        return "bad"
    if (
        valid_block_count >= 5
        and frame_fit_success_rate >= 0.8
        and saturated_frame_fraction < 0.01
        and orientation_reliable
        and roi_safety_status != "warning"
    ):
        return "good"
    return "caution"


def _array_stat(values: np.ndarray, stat: str) -> Optional[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    if stat == "median":
        return float(np.median(finite))
    if stat == "p95":
        return float(np.percentile(finite, 95))
    if stat == "max":
        return float(np.max(finite))
    raise ValueError(f"unknown stat: {stat}")


def _valid_blocks_with_finite_seeing(blocks: Sequence[BlockResult]) -> List[BlockResult]:
    return [
        block
        for block in blocks
        if block.valid and _is_finite_number(block.seeing_mean_zenith_arcsec)
    ]


def _is_finite_number(value: Optional[float]) -> bool:
    return value is not None and bool(np.isfinite(value))


def _optional_close(left: Optional[float], right: Optional[float], *, atol: float = 1e-12) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return bool(np.isfinite(left) and np.isfinite(right) and np.isclose(left, right, atol=atol))


def _representative_block_values(valid_blocks: Sequence[BlockResult]) -> Dict[str, Optional[float]]:
    return {
        "seeing_observed_arcsec": _nanmedian(
            [block.seeing_mean_observed_arcsec for block in valid_blocks]
        ),
        "seeing_zenith_arcsec": _nanmedian(
            [block.seeing_mean_zenith_arcsec for block in valid_blocks]
        ),
        "r0_observed_m": _nanmedian([block.r0_mean_observed_m for block in valid_blocks]),
        "r0_zenith_m": _nanmedian([block.r0_mean_zenith_m for block in valid_blocks]),
        "var_L_px2": _nanmedian([block.var_L_px2 for block in valid_blocks]),
        "var_T_px2": _nanmedian([block.var_T_px2 for block in valid_blocks]),
        "var_L_rad2": _nanmedian([block.var_L_rad2 for block in valid_blocks]),
        "var_T_rad2": _nanmedian([block.var_T_rad2 for block in valid_blocks]),
        "seeing_L_observed_arcsec": _nanmedian(
            [block.seeing_L_observed_arcsec for block in valid_blocks]
        ),
        "seeing_T_observed_arcsec": _nanmedian(
            [block.seeing_T_observed_arcsec for block in valid_blocks]
        ),
        "seeing_L_zenith_arcsec": _nanmedian(
            [block.seeing_L_zenith_arcsec for block in valid_blocks]
        ),
        "seeing_T_zenith_arcsec": _nanmedian(
            [block.seeing_T_zenith_arcsec for block in valid_blocks]
        ),
    }


def _sanity_check_from_r0(
    *,
    r0_zenith_m: Optional[float],
    seeing_zenith_arcsec: Optional[float],
    wavelength_m: float,
) -> Dict[str, Any]:
    if r0_zenith_m is None:
        return {
            "formula": "seeing_arcsec = 0.98 * wavelength_m / r0_m * 206265",
            "r0_zenith_m": None,
            "wavelength_m": wavelength_m,
            "seeing_from_r0_arcsec": None,
            "reported_seeing_zenith_arcsec": seeing_zenith_arcsec,
            "absolute_difference_arcsec": None,
        }
    seeing_from_r0 = seeing_from_r0_arcsec(r0_zenith_m, wavelength_m)
    return {
        "formula": "seeing_arcsec = 0.98 * wavelength_m / r0_m * 206265",
        "r0_zenith_m": r0_zenith_m,
        "wavelength_m": wavelength_m,
        "seeing_from_r0_arcsec": seeing_from_r0,
        "reported_seeing_zenith_arcsec": seeing_zenith_arcsec,
        "absolute_difference_arcsec": abs(seeing_from_r0 - seeing_zenith_arcsec)
        if seeing_zenith_arcsec is not None
        else None,
    }


def build_summary(
    *,
    frame_results: Sequence[FrameResult],
    block_results: Sequence[BlockResult],
    config: AnalysisConfig,
    pixel_scale_arcsec_per_px: float,
    orientation: OrientationResult,
    warnings_out: Sequence[str],
    input_path: Optional[Path],
    ser_metadata: Optional[SERMetadata],
    time_source: str,
    fps: Optional[float],
    saturation_level: Optional[float],
    reject_reason_counts: Dict[str, int],
    frame_distribution_rows: Sequence[Dict[str, Any]],
    orientation_scan_rows: Sequence[Dict[str, Any]],
    roi_safety_summary: Dict[str, Any],
    input_path_original: Optional[Path] = None,
    input_path_resolved_ser: Optional[Path] = None,
    input_was_directory: bool = False,
    ser_selection_mode: Optional[str] = None,
    companion_files_copied: Optional[List[str]] = None,
    output_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    auto_output_name: Optional[bool] = None,
    input_manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    total_frames = len(frame_results)
    valid_frames = sum(1 for frame in frame_results if frame.frame_valid)
    fit_attempts = sum(frame.fit_attempts for frame in frame_results)
    spot_successes = sum(
        int(frame.fit1.success) + int(frame.fit2.success) for frame in frame_results
    )
    valid_blocks = _valid_blocks_with_finite_seeing(block_results)
    seeing_values = np.asarray(
        [
            block.seeing_mean_zenith_arcsec
            for block in valid_blocks
            if _is_finite_number(block.seeing_mean_zenith_arcsec)
        ],
        dtype=float,
    )
    r0_values = np.asarray(
        [
            block.r0_mean_zenith_m
            for block in valid_blocks
            if _is_finite_number(block.r0_mean_zenith_m)
        ],
        dtype=float,
    )
    representative = _representative_block_values(valid_blocks)
    sanity_check = _sanity_check_from_r0(
        r0_zenith_m=representative.get("r0_zenith_m"),
        seeing_zenith_arcsec=representative.get("seeing_zenith_arcsec"),
        wavelength_m=config.instrument.wavelength_m,
    )
    zenith_factor = (
        zenith_correction_factor(config.instrument.zenith_deg)
        if config.instrument.zenith_deg is not None
        else None
    )
    saturation_stats = _saturation_summary(
        frame_results=frame_results,
        saturation_level=saturation_level,
        saturation_margin=config.quality.saturation_margin,
        saturation_core_min_pixels=config.quality.saturation_core_min_pixels,
    )
    relative_motion_count = int(reject_reason_counts.get("bad_relative_motion", 0))
    orientation_valid_block_count = len(valid_blocks)
    orientation_mismatch_score = _orientation_mismatch_score(orientation_scan_rows)
    orientation_reliable = (
        orientation.angle_deg is not None
        and orientation_valid_block_count >= 3
        and not orientation.fallback_used
    )
    warnings = list(dict.fromkeys(warnings_out))
    if orientation_valid_block_count < 3:
        warnings.append(
            "valid block が 3 未満です。orientation と seeing は参考値として扱ってください。"
        )
    elif orientation_valid_block_count < 5:
        warnings.append(
            "valid block が 5 未満です。auto_consistency orientation の安定性を確認してください。"
        )
    quality_flags = {
        "too_few_valid_blocks": orientation_valid_block_count < 3,
        "low_fit_success_rate": ratio(valid_frames, total_frames) < 0.75,
        "high_saturation_fraction": saturation_stats["saturated_frame_fraction"] > 0.05,
        "orientation_unreliable": not orientation_reliable,
        "relative_motion_outliers_detected": relative_motion_count > 0,
        "roi_too_close_to_edge": roi_safety_summary.get("roi_safety_status") == "unsafe",
        "roi_safety_warning": roi_safety_summary.get("roi_safety_status") == "warning",
        "roi_auto_shrunk": bool(roi_safety_summary.get("roi_auto_shrunk")),
    }
    result_reliability = _result_reliability(
        valid_block_count=orientation_valid_block_count,
        frame_fit_success_rate=ratio(valid_frames, total_frames),
        saturated_frame_fraction=saturation_stats["saturated_frame_fraction"],
        orientation_reliable=orientation_reliable,
        relative_motion_outlier_count=relative_motion_count,
        roi_safety_status=str(roi_safety_summary.get("roi_safety_status")),
        total_frames=total_frames,
    )
    valid_rejected_stats = _valid_rejected_medians(frame_results)
    resolved_ser = input_path_resolved_ser if input_path_resolved_ser is not None else input_path
    original_input = input_path_original if input_path_original is not None else input_path
    output_root_value = output_root if output_root is not None else output_dir
    return {
        "input_file": str(input_path) if input_path is not None else None,
        "input_path_original": str(original_input) if original_input is not None else None,
        "input_path_resolved_ser": str(resolved_ser) if resolved_ser is not None else None,
        "input_was_directory": bool(input_was_directory),
        "ser_selection_mode": ser_selection_mode,
        "companion_files_copied": list(companion_files_copied or []),
        "input_ser_path": str(resolved_ser) if resolved_ser is not None else None,
        "input_ser_filename": resolved_ser.name if resolved_ser is not None else None,
        "input_ser_stem": resolved_ser.stem if resolved_ser is not None else None,
        "output_root": str(output_root_value) if output_root_value is not None else None,
        "output_dir": str(output_dir) if output_dir is not None else None,
        "auto_output_name": auto_output_name,
        "input_manifest_path": str(input_manifest_path)
        if input_manifest_path is not None
        else None,
        "ser_metadata": ser_metadata.to_dict() if ser_metadata is not None else None,
        "config_used": model_to_dict(config),
        "effective_pixel_scale_arcsec_per_px": pixel_scale_arcsec_per_px,
        "binning": config.camera.binning,
        "roi_size_px": config.roi.size_px,
        "roi_fallback_size_px": config.roi.fallback_size_px,
        **roi_safety_summary,
        "zenith_deg": config.instrument.zenith_deg,
        "zenith_correction_factor": zenith_factor,
        "time_source": time_source,
        "fps": fps,
        "estimated_fps": _estimated_fps(frame_results, ser_metadata, fps),
        "total_frames": total_frames,
        "valid_frames": valid_frames,
        "frame_fit_success_rate": ratio(valid_frames, total_frames),
        "spot_fit_success_rate": ratio(spot_successes, fit_attempts),
        "orientation_mode": orientation.mode,
        "orientation_angle_deg": orientation.angle_deg,
        "estimated_mask_angle_deg": orientation.angle_deg,
        "orientation_confidence": orientation.confidence,
        "orientation_reliable": orientation_reliable,
        "orientation_valid_block_count": orientation_valid_block_count,
        "orientation_mismatch_score": orientation_mismatch_score,
        "number_of_valid_blocks": len(valid_blocks),
        "median_seeing_observed_arcsec": _nanmedian(
            [block.seeing_mean_observed_arcsec for block in valid_blocks]
        ),
        "median_seeing_zenith_corrected_arcsec": _nanmedian(seeing_values),
        "seeing_observed_arcsec": representative.get("seeing_observed_arcsec"),
        "seeing_zenith_arcsec": representative.get("seeing_zenith_arcsec"),
        "mean_seeing_zenith_corrected_arcsec": _nanmean(seeing_values),
        "std_seeing_zenith_corrected_arcsec": _nanstd(seeing_values),
        "r0_observed_m": representative.get("r0_observed_m"),
        "r0_zenith_m": representative.get("r0_zenith_m"),
        "median_r0_zenith_m": _nanmedian(r0_values),
        "var_L_px2": representative.get("var_L_px2"),
        "var_T_px2": representative.get("var_T_px2"),
        "var_L_rad2": representative.get("var_L_rad2"),
        "var_T_rad2": representative.get("var_T_rad2"),
        "seeing_L_observed_arcsec": representative.get("seeing_L_observed_arcsec"),
        "seeing_T_observed_arcsec": representative.get("seeing_T_observed_arcsec"),
        "seeing_L_zenith_arcsec": representative.get("seeing_L_zenith_arcsec"),
        "seeing_T_zenith_arcsec": representative.get("seeing_T_zenith_arcsec"),
        **saturation_stats,
        **valid_rejected_stats,
        "reject_reason_counts": reject_reason_counts,
        "frame_distribution_summary": list(frame_distribution_rows),
        "quality_flags": quality_flags,
        "result_reliability": result_reliability,
        "sanity_check": sanity_check,
        "warnings": list(dict.fromkeys(warnings)),
    }


def _effective_saturation_level(
    config: AnalysisConfig,
    saturation_level: Optional[float],
) -> Optional[float]:
    if config.quality.saturation_level is not None:
        return float(config.quality.saturation_level)
    return saturation_level


def _resolve_time_source(reader: SERReader, config: AnalysisConfig) -> str:
    if config.timing.use_ser_timestamps_if_available and reader.read_timestamps() is not None:
        return "ser_timestamps"
    if config.timing.fps is not None:
        return "fps"
    return "frame_index"


def _frame_time_seconds(
    reader: SERReader,
    frame_index: int,
    time_source: str,
    fps: Optional[float],
) -> Optional[float]:
    if time_source == "ser_timestamps":
        return reader.frame_time_seconds(frame_index)
    if time_source == "fps" and fps is not None:
        return frame_index / fps
    return None


def _selected_frame_count(
    frame_count: int,
    start_frame: int,
    end_frame: Optional[int],
    max_frames: Optional[int],
) -> int:
    stop = frame_count if end_frame is None else min(end_frame, frame_count)
    count = max(0, stop - max(0, start_frame))
    if max_frames is not None:
        count = min(count, max_frames)
    return count


def _with_progress(
    frames: Iterable[FrameInput],
    *,
    total: Optional[int],
    enabled: bool,
) -> Iterator[FrameInput]:
    if not enabled:
        yield from frames
        return
    try:
        from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
    except Exception:
        yield from frames
        return
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task_id = progress.add_task("frame を解析中", total=total)
        for item in frames:
            yield item
            progress.advance(task_id)


def _make_valid_block_masks(
    frame_results: Sequence[FrameResult],
    valid_positions: Sequence[int],
    config: AnalysisConfig,
    time_source: str,
) -> List[np.ndarray]:
    if not valid_positions:
        return []
    groups = _group_frame_positions(frame_results, config, time_source)
    masks: List[np.ndarray] = []
    valid_position_to_array_index = {position: idx for idx, position in enumerate(valid_positions)}
    for group in groups:
        mask = np.zeros(len(valid_positions), dtype=bool)
        for position in group:
            if position in valid_position_to_array_index:
                mask[valid_position_to_array_index[position]] = True
        if np.any(mask):
            masks.append(mask)
    return masks


def _group_frame_positions(
    frame_results: Sequence[FrameResult],
    config: AnalysisConfig,
    time_source: str,
) -> List[List[int]]:
    if not frame_results:
        return []
    use_time = (
        config.statistics.block_mode == "time"
        and time_source != "frame_index"
        and any(frame.time_sec is not None for frame in frame_results)
    )
    groups: Dict[int, List[int]] = defaultdict(list)
    if use_time:
        times = [frame.time_sec for frame in frame_results if frame.time_sec is not None]
        t0 = min(times) if times else 0.0
        duration = config.statistics.block_duration_sec
        for pos, frame in enumerate(frame_results):
            time_sec = frame.time_sec if frame.time_sec is not None else t0
            block_id = int((time_sec - t0) // duration)
            groups[block_id].append(pos)
    else:
        first_frame = frame_results[0].frame_index
        size = config.statistics.block_size_frames
        for pos, frame in enumerate(frame_results):
            block_id = int((frame.frame_index - first_frame) // size)
            groups[block_id].append(pos)
    return [groups[key] for key in sorted(groups)]


def _min_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    return min(present) if present else None


def _max_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _estimated_fps(
    frame_results: Sequence[FrameResult],
    ser_metadata: Optional[SERMetadata],
    fps: Optional[float],
) -> Optional[float]:
    if ser_metadata is not None and ser_metadata.estimated_fps is not None:
        return ser_metadata.estimated_fps
    if fps is not None:
        return fps
    timed = [frame for frame in frame_results if frame.time_sec is not None]
    if len(timed) < 2:
        return None
    elapsed = timed[-1].time_sec - timed[0].time_sec  # type: ignore[operator]
    if elapsed is None or elapsed <= 0:
        return None
    return (len(timed) - 1) / elapsed


def _nanmedian(values) -> Optional[float]:  # type: ignore[no-untyped-def]
    array = np.asarray([value for value in values if value is not None], dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return float(np.nanmedian(array))


def _nanmean(values) -> Optional[float]:  # type: ignore[no-untyped-def]
    array = np.asarray([value for value in values if value is not None], dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return float(np.nanmean(array))


def _nanstd(values) -> Optional[float]:  # type: ignore[no-untyped-def]
    array = np.asarray([value for value in values if value is not None], dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return float(np.nanstd(array))
