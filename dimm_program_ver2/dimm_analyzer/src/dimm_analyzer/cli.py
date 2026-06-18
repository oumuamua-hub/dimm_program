"""Typer command-line interface for dimm-analyzer."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console

from .config import apply_cli_overrides, config_warnings, load_config, validate_config
from .exceptions import DimmAnalyzerError
from .file_picker import (
    provided_paths_are_complete,
    resolve_provided_paths,
    select_analysis_paths,
)
from .input_resolver import (
    ResolvedSERInputs,
    companion_source_root,
    copy_companion_files,
    resolve_ser_inputs,
)
from .pipeline import analyze_ser
from .report import (
    write_batch_manifest_json,
    write_batch_summary_csv,
    write_comparison_summary_csv,
    write_input_manifest_json,
)

app = typer.Typer(add_completion=False, help="SharpCap mono SER の DIMM 観測を解析します。")
console = Console()


@dataclass(frozen=True)
class OutputTarget:
    ser_path: Path
    output_root: Path
    output_dir: Path
    input_manifest_path: Path
    auto_output_name: bool


@app.command()
def main(
    input_path: Optional[Path] = typer.Option(
        None,
        "--input",
        exists=False,
        readable=True,
        help="解析対象の science SER、または SharpCap capture folder。",
    ),
    recursive: bool = typer.Option(
        True,
        "--recursive/--no-recursive",
        help="入力がディレクトリの場合にサブフォルダも検索します。",
    ),
    ser_select: str = typer.Option(
        "error",
        "--ser-select",
        help="ディレクトリ内に複数SERがある場合の選択: error, newest, largest, all。",
    ),
    config_path: Optional[Path] = typer.Option(
        None, "--config", exists=False, readable=True, help="YAML 設定ファイル。"
    ),
    output_dir: Optional[Path] = typer.Option(None, "--output", help="出力ディレクトリ。"),
    auto_output_name: bool = typer.Option(
        True,
        "--auto-output-name/--no-auto-output-name",
        help="SERファイル名のサブフォルダを出力先に自動作成します。",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="既存の出力フォルダへの書き込みを許可します。",
    ),
    append_run_id: bool = typer.Option(
        False,
        "--append-run-id",
        help="出力フォルダが既にある場合に _002 などを付けて衝突回避します。",
    ),
    dark_path: Optional[Path] = typer.Option(None, "--dark", exists=False, readable=True),
    fps: Optional[float] = typer.Option(None, "--fps", min=0.0, help="fallback 用 FPS。"),
    zenith_deg: Optional[float] = typer.Option(None, "--zenith-deg", min=0.0, max=89.999),
    orientation_mode: Optional[str] = typer.Option(
        None,
        "--orientation-mode",
        help="orientation mode を上書き: auto_consistency, manual, auto_pair。",
    ),
    orientation_angle_deg: Optional[float] = typer.Option(
        None,
        "--orientation-angle-deg",
        "--mask-angle-deg",
        min=0.0,
        max=180.0,
        help="manual orientation 用の mask angle。",
    ),
    roi_size: Optional[int] = typer.Option(
        None,
        "--roi-size",
        min=3,
        help="解析 ROI size。奇数を指定してください。",
    ),
    roi_safety_margin: Optional[float] = typer.Option(
        None,
        "--roi-safety-margin",
        min=0.0,
        help="ROI safety 判定で half ROI に足す余裕ピクセル。",
    ),
    disable_roi_safety_check: bool = typer.Option(
        False,
        "--disable-roi-safety-check",
        help="ROI safety 診断CSV/PNGとsummary判定を無効化します。",
    ),
    auto_shrink_roi_if_unsafe: bool = typer.Option(
        False,
        "--auto-shrink-roi-if-unsafe",
        help="解析前診断で unsafe の場合だけROIを安全側に自動縮小します。",
    ),
    disable_large_jump_rejection: bool = typer.Option(
        False,
        "--disable-large-jump-rejection",
        help="large_jump rejection を無効化します。",
    ),
    disable_saturation_rejection: bool = typer.Option(
        False,
        "--disable-saturation-rejection",
        help="saturated rejection を無効化します。診断値は出力されます。",
    ),
    disable_relative_motion_rejection: bool = typer.Option(
        False,
        "--disable-relative-motion-rejection",
        help="bad_relative_motion rejection を無効化します。",
    ),
    max_relative_motion_deviation_px: Optional[float] = typer.Option(
        None,
        "--max-relative-motion-deviation-px",
        min=0.0,
        help="dx/dy の running median からの許容ずれ量。",
    ),
    saturation_level: Optional[float] = typer.Option(
        None,
        "--saturation-level",
        min=1.0,
        help="飽和ADU値を上書きします。mono16なら通常65535。",
    ),
    saturation_margin: Optional[float] = typer.Option(
        None,
        "--saturation-margin",
        min=0.0,
        help="saturation_level から差し引く余裕値。",
    ),
    comparison_suite: bool = typer.Option(
        False,
        "--comparison-suite",
        help="orientation/ROI/large_jump の比較解析を一括実行します。",
    ),
    picker_mode: str = typer.Option(
        "auto",
        "--picker-mode",
        help="ファイル選択方式: auto, dialog, terminal。",
    ),
    max_frames: Optional[int] = typer.Option(None, "--max-frames", min=1),
    start_frame: int = typer.Option(0, "--start-frame", min=0),
    end_frame: Optional[int] = typer.Option(None, "--end-frame", min=1),
    preview: bool = typer.Option(False, "--preview", help="先頭最大 500 frame だけ解析。"),
    verbose: bool = typer.Option(False, "--verbose", help="詳細な summary を表示。"),
) -> None:
    """DIMM 解析パイプラインを実行します。"""

    try:
        early_config = load_config(config_path) if config_path is not None else None
        if (
            early_config is not None
            and provided_paths_are_complete(
                input_path=input_path,
                config_path=config_path,
                output_dir=output_dir,
                zenith_deg=zenith_deg,
                config=early_config,
            )
        ):
            selected = resolve_provided_paths(
                input_path=input_path,
                config_path=config_path,
                output_dir=output_dir,
                dark_path=dark_path,
                zenith_deg=zenith_deg,
            )
            config = early_config
        else:
            selected = select_analysis_paths(
                input_path=input_path,
                config_path=config_path,
                output_dir=output_dir,
                dark_path=dark_path,
                zenith_deg=zenith_deg,
                config=early_config,
                picker_mode=picker_mode,
            )
            config = early_config if early_config is not None else load_config(selected.config_path)
        apply_cli_overrides(
            config,
            fps=fps,
            dark_path=selected.dark_path,
            zenith_deg=selected.zenith_deg,
            roi_size_px=roi_size,
            roi_safety_margin_px=roi_safety_margin,
            disable_roi_safety_check=disable_roi_safety_check,
            auto_shrink_roi_if_unsafe=auto_shrink_roi_if_unsafe,
            disable_large_jump_rejection=disable_large_jump_rejection,
            disable_saturation_rejection=disable_saturation_rejection,
            disable_relative_motion_rejection=disable_relative_motion_rejection,
            max_relative_motion_deviation_px=max_relative_motion_deviation_px,
            saturation_level=saturation_level,
            saturation_margin=saturation_margin,
            orientation_mode=orientation_mode,
            orientation_angle_deg=orientation_angle_deg,
        )
        validate_config(config)
        if overwrite and append_run_id:
            raise DimmAnalyzerError("--overwrite と --append-run-id は同時指定できません。")
        for warning in config_warnings(config):
            console.print(f"[yellow]警告:[/yellow] {warning}")

        resolved_inputs = resolve_ser_inputs(
            selected.input_path,
            recursive=recursive,
            ser_select=ser_select,
        )

        if comparison_suite:
            if len(resolved_inputs.ser_paths) != 1:
                raise DimmAnalyzerError(
                    "--comparison-suite は単一SER入力で実行してください。"
                    "フォルダ入力では --ser-select newest または largest で1つに絞れます。"
                )
            target = _resolve_output_target(
                ser_path=resolved_inputs.ser_paths[0],
                output_root=selected.output_dir,
                auto_output_name=auto_output_name,
                overwrite=overwrite,
                append_run_id=append_run_id,
            )
            _print_output_resolution(target)
            rows = _run_comparison_suite(
                target=target,
                original_input_path=resolved_inputs.original_input_path,
                input_was_directory=resolved_inputs.input_was_directory,
                ser_selection_mode=resolved_inputs.ser_selection_mode,
                config=config,
                config_path=selected.config_path,
                start_frame=start_frame,
                end_frame=end_frame,
                max_frames=max_frames,
                preview=preview,
                recursive=recursive,
            )
            write_comparison_summary_csv(target.output_dir / "comparison_summary.csv", rows)
            console.print(
                f"[green]比較解析が完了しました。[/green] "
                f"比較表: {target.output_dir / 'comparison_summary.csv'}"
            )
            return

        if len(resolved_inputs.ser_paths) > 1:
            batch_output_root = _resolve_batch_output_root(
                selected.output_dir,
                overwrite=overwrite,
                append_run_id=append_run_id,
            )
            batch_rows = _run_batch_analysis(
                resolved_inputs=resolved_inputs,
                config=config,
                config_path=selected.config_path,
                output_root=batch_output_root,
                recursive=recursive,
                overwrite=overwrite,
                start_frame=start_frame,
                end_frame=end_frame,
                max_frames=max_frames,
                preview=preview,
            )
            failures = [row for row in batch_rows if row["status"] != "success"]
            console.print(f"解析SER数: {len(batch_rows)} / 失敗: {len(failures)}")
            console.print(f"batch summary: {batch_output_root / 'batch_summary.csv'}")
            if len(failures) == len(batch_rows):
                raise DimmAnalyzerError("batch解析の全SERが失敗しました。")
            if failures:
                console.print("[yellow]一部のSER解析が失敗しました。batch_summary.csvを確認してください。[/yellow]")
            else:
                console.print(
                    f"[green]batch解析が完了しました。[/green] 出力先: {batch_output_root}"
                )
            return

        target = _resolve_output_target(
            ser_path=resolved_inputs.ser_paths[0],
            output_root=selected.output_dir,
            auto_output_name=auto_output_name,
            overwrite=overwrite,
            append_run_id=append_run_id,
        )
        _print_output_resolution(target)
        companion_files = _copy_companions_for_output(
            resolved_inputs=resolved_inputs,
            ser_path=target.ser_path,
            output_dir=target.output_dir,
            recursive=recursive,
            output_root=target.output_root,
        )
        analysis_started_at = _utc_now_iso()
        result = analyze_ser(
            input_path=target.ser_path,
            config=config,
            output_dir=target.output_dir,
            start_frame=start_frame,
            end_frame=end_frame,
            max_frames=max_frames,
            preview=preview,
            show_progress=True,
            input_path_original=resolved_inputs.original_input_path,
            input_was_directory=resolved_inputs.input_was_directory,
            ser_selection_mode=resolved_inputs.ser_selection_mode,
            companion_files_copied=companion_files,
            output_root=target.output_root,
            auto_output_name=target.auto_output_name,
            input_manifest_path=target.input_manifest_path,
        )
        _write_input_manifest(
            target=target,
            config_path=selected.config_path,
            zenith_deg=config.instrument.zenith_deg,
            fps=config.timing.fps,
            analysis_started_at=analysis_started_at,
            ser_metadata=result.summary.get("ser_metadata"),
            companion_files=companion_files,
        )
    except DimmAnalyzerError as exc:
        console.print(f"[red]エラー:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        console.print(f"[red]予期しないエラー:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    summary = result.summary
    console.print(f"[green]完了しました。[/green] 出力先: {summary.get('output_dir')}")
    console.print(
        "有効 frame: "
        f"{summary['valid_frames']}/{summary['total_frames']} "
        f"({summary['frame_fit_success_rate']:.1%})"
    )
    console.print(f"Orientation angle: {summary['estimated_mask_angle_deg']}")
    console.print(
        "天頂補正後 median seeing: "
        f"{summary['median_seeing_zenith_corrected_arcsec']} arcsec"
    )
    _print_roi_safety(summary)
    _print_reliability_warnings(summary)
    if verbose:
        for warning in summary["warnings"]:
            console.print(f"[yellow]警告:[/yellow] {warning}")


def _resolve_output_target(
    *,
    ser_path: Path,
    output_root: Path,
    auto_output_name: bool,
    overwrite: bool,
    append_run_id: bool,
) -> OutputTarget:
    root = Path(output_root).expanduser()
    base_output_dir = root / _safe_output_stem(ser_path.stem) if auto_output_name else root
    resolved_output_dir = _resolve_collision(
        base_output_dir,
        overwrite=overwrite,
        append_run_id=append_run_id,
    )
    return OutputTarget(
        ser_path=ser_path,
        output_root=root,
        output_dir=resolved_output_dir,
        input_manifest_path=resolved_output_dir / "input_manifest.json",
        auto_output_name=auto_output_name,
    )


def _resolve_batch_output_root(
    output_root: Path,
    *,
    overwrite: bool,
    append_run_id: bool,
) -> Path:
    root = Path(output_root).expanduser()
    return _resolve_collision(root, overwrite=overwrite, append_run_id=append_run_id)


def _resolve_collision(path: Path, *, overwrite: bool, append_run_id: bool) -> Path:
    if not path.exists():
        return path
    if not path.is_dir():
        raise DimmAnalyzerError(f"出力先に同名ファイルが既に存在します: {path}")
    if overwrite:
        return path
    if append_run_id:
        return _next_available_output_path(path)
    raise DimmAnalyzerError(
        "出力フォルダが既に存在します。"
        f"--overwrite または --append-run-id を指定してください: {path}"
    )


def _next_available_output_path(path: Path) -> Path:
    for index in range(2, 10000):
        candidate = path.with_name(f"{path.name}_{index:03d}")
        if not candidate.exists():
            return candidate
    raise DimmAnalyzerError(f"出力フォルダ名の自動採番に失敗しました: {path}")


def _print_output_resolution(target: OutputTarget) -> None:
    console.print(f"Input SER: {target.ser_path}")
    console.print(f"Output root: {target.output_root}")
    console.print(f"Resolved output directory: {target.output_dir}")


def _write_input_manifest(
    *,
    target: OutputTarget,
    config_path: Path,
    zenith_deg: Optional[float],
    fps: Optional[float],
    analysis_started_at: str,
    ser_metadata: Optional[Dict[str, Any]],
    companion_files: List[str],
) -> None:
    manifest = {
        "input_ser_path": str(target.ser_path),
        "input_ser_filename": target.ser_path.name,
        "input_ser_stem": target.ser_path.stem,
        "output_root": str(target.output_root),
        "output_dir": str(target.output_dir),
        "config_path": str(config_path),
        "zenith_deg": zenith_deg,
        "fps": fps,
        "analysis_started_at": analysis_started_at,
        "ser_metadata": ser_metadata,
        "copied_companion_files": list(companion_files),
        "auto_output_name": target.auto_output_name,
    }
    target.input_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_input_manifest_json(target.input_manifest_path, manifest)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_batch_analysis(
    *,
    resolved_inputs: ResolvedSERInputs,
    config,
    config_path: Path,
    output_root: Path,
    recursive: bool,
    overwrite: bool,
    start_frame: int,
    end_frame: Optional[int],
    max_frames: Optional[int],
    preview: bool,
) -> List[Dict[str, Any]]:  # type: ignore[no-untyped-def]
    output_root.mkdir(parents=True, exist_ok=True)
    batch_started_at = _utc_now_iso()
    output_dirs = _ser_output_dirs(output_root, resolved_inputs.ser_paths)
    rows: List[Dict[str, Any]] = []
    manifest_entries: List[Dict[str, Any]] = []
    for ser_path, ser_output in zip(resolved_inputs.ser_paths, output_dirs):
        target = OutputTarget(
            ser_path=ser_path,
            output_root=output_root,
            output_dir=_resolve_collision(
                ser_output,
                overwrite=overwrite,
                append_run_id=False,
            ),
            input_manifest_path=ser_output / "input_manifest.json",
            auto_output_name=True,
        )
        console.print(f"[cyan]SER解析:[/cyan] {ser_path}")
        _print_output_resolution(target)
        companion_files = _copy_companions_for_output(
            resolved_inputs=resolved_inputs,
            ser_path=ser_path,
            output_dir=target.output_dir,
            recursive=recursive,
            output_root=output_root,
        )
        analysis_started_at = _utc_now_iso()
        try:
            result = analyze_ser(
                input_path=ser_path,
                config=deepcopy(config),
                output_dir=target.output_dir,
                start_frame=start_frame,
                end_frame=end_frame,
                max_frames=max_frames,
                preview=preview,
                show_progress=True,
                input_path_original=resolved_inputs.original_input_path,
                input_was_directory=resolved_inputs.input_was_directory,
                ser_selection_mode=resolved_inputs.ser_selection_mode,
                companion_files_copied=companion_files,
                output_root=target.output_root,
                auto_output_name=target.auto_output_name,
                input_manifest_path=target.input_manifest_path,
            )
        except Exception as exc:
            console.print(f"[red]SER解析失敗:[/red] {ser_path}: {exc}")
            row = _batch_error_row(ser_path, target.output_dir, exc)
            rows.append(row)
            manifest_entries.append(_batch_manifest_entry(row))
            continue
        _write_input_manifest(
            target=target,
            config_path=config_path,
            zenith_deg=config.instrument.zenith_deg,
            fps=config.timing.fps,
            analysis_started_at=analysis_started_at,
            ser_metadata=result.summary.get("ser_metadata"),
            companion_files=companion_files,
        )
        row = _batch_success_row(ser_path, target.output_dir, result.summary)
        rows.append(row)
        manifest_entries.append(_batch_manifest_entry(row))

    batch_summary_path = output_root / "batch_summary.csv"
    write_batch_summary_csv(batch_summary_path, rows)
    write_batch_manifest_json(
        output_root / "batch_manifest.json",
        {
            "input_path_original": str(resolved_inputs.original_input_path),
            "input_was_directory": resolved_inputs.input_was_directory,
            "ser_select_mode": resolved_inputs.ser_selection_mode,
            "number_of_ser_files": len(resolved_inputs.ser_paths),
            "ser_files": manifest_entries,
            "output_root": str(output_root),
            "batch_summary_csv": str(batch_summary_path),
            "entries": manifest_entries,
            "analysis_started_at": batch_started_at,
            "analysis_finished_at": _utc_now_iso(),
            "recursive": recursive,
        },
    )
    return rows


def _copy_companions_for_output(
    *,
    resolved_inputs: ResolvedSERInputs,
    ser_path: Path,
    output_dir: Path,
    recursive: bool,
    output_root: Path,
) -> List[str]:
    source_root = companion_source_root(resolved_inputs.original_input_path, ser_path)
    return copy_companion_files(
        source_root=source_root,
        output_dir=output_dir,
        recursive=recursive,
        excluded_roots=[output_root],
    )


def _ser_output_dirs(output_dir: Path, ser_paths: List[Path]) -> List[Path]:
    counts: Dict[str, int] = {}
    outputs: List[Path] = []
    for ser_path in ser_paths:
        stem = _safe_output_stem(ser_path.stem)
        count = counts.get(stem, 0) + 1
        counts[stem] = count
        name = stem if count == 1 else f"{stem}_{count:03d}"
        outputs.append(output_dir / name)
    return outputs


def _safe_output_stem(stem: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stem)
    return safe or "ser"


def _batch_success_row(
    ser_path: Path,
    output_dir: Path,
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "ser_file": ser_path.name,
        "ser_path": str(ser_path),
        "output_dir": str(output_dir),
        "summary_json": str(output_dir / "summary.json"),
        "block_results_csv": str(output_dir / "block_results.csv"),
        "per_frame_fits_csv": str(output_dir / "per_frame_fits.csv"),
        "status": "success",
        "total_frames": summary.get("total_frames"),
        "valid_frames": summary.get("valid_frames"),
        "frame_fit_success_rate": summary.get("frame_fit_success_rate"),
        "number_of_valid_blocks": summary.get("number_of_valid_blocks"),
        "median_seeing_zenith_corrected_arcsec": summary.get(
            "median_seeing_zenith_corrected_arcsec"
        ),
        "median_r0_zenith_m": summary.get("median_r0_zenith_m"),
        "result_reliability": summary.get("result_reliability"),
        "saturated_frame_fraction": summary.get("saturated_frame_fraction"),
        "roi_safety_status": summary.get("roi_safety_status"),
        "orientation_angle_deg": summary.get("orientation_angle_deg"),
        "error": "",
    }


def _batch_error_row(
    ser_path: Path,
    output_dir: Path,
    exc: Exception,
) -> Dict[str, Any]:
    return {
        "ser_file": ser_path.name,
        "ser_path": str(ser_path),
        "output_dir": str(output_dir),
        "summary_json": str(output_dir / "summary.json"),
        "block_results_csv": str(output_dir / "block_results.csv"),
        "per_frame_fits_csv": str(output_dir / "per_frame_fits.csv"),
        "status": "failed",
        "total_frames": None,
        "valid_frames": None,
        "frame_fit_success_rate": None,
        "number_of_valid_blocks": None,
        "median_seeing_zenith_corrected_arcsec": None,
        "median_r0_zenith_m": None,
        "result_reliability": None,
        "saturated_frame_fraction": None,
        "roi_safety_status": None,
        "orientation_angle_deg": None,
        "error": str(exc),
    }


def _batch_manifest_entry(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ser_file": row["ser_file"],
        "ser_path": row["ser_path"],
        "output_dir": row["output_dir"],
        "summary_json": row["summary_json"],
        "status": row["status"],
        "error": row.get("error") or None,
    }


def _run_comparison_suite(
    *,
    target: OutputTarget,
    original_input_path: Path,
    input_was_directory: bool,
    ser_selection_mode: str,
    config,
    config_path: Path,
    start_frame: int,
    end_frame: Optional[int],
    max_frames: Optional[int],
    preview: bool,
    recursive: bool,
):  # type: ignore[no-untyped-def]
    variants = [
        ("A_orientation_auto_consistency", "auto_consistency", None, 25, True),
        ("B_orientation_manual_0", "manual", 0.0, 25, True),
        ("C_orientation_manual_45", "manual", 45.0, 25, True),
        ("D_orientation_manual_90", "manual", 90.0, 25, True),
        ("E_roi_21", "auto_consistency", None, 21, True),
        ("F_roi_31", "auto_consistency", None, 31, True),
        ("G_large_jump_off", "auto_consistency", None, 25, False),
    ]
    rows = []
    for variant, mode, angle, roi_size, reject_large_jump in variants:
        variant_config = deepcopy(config)
        variant_config.orientation.mode = mode
        variant_config.orientation.mask_angle_deg = angle
        variant_config.roi.size_px = roi_size
        if variant_config.roi.fallback_size_px < roi_size:
            variant_config.roi.fallback_size_px = roi_size
        variant_config.quality.reject_large_jump = reject_large_jump
        variant_output = target.output_dir / variant
        variant_target = OutputTarget(
            ser_path=target.ser_path,
            output_root=target.output_root,
            output_dir=variant_output,
            input_manifest_path=variant_output / "input_manifest.json",
            auto_output_name=target.auto_output_name,
        )
        companion_files = copy_companion_files(
            source_root=companion_source_root(original_input_path, target.ser_path),
            output_dir=variant_output,
            recursive=recursive,
            excluded_roots=[target.output_dir],
        )
        analysis_started_at = _utc_now_iso()
        console.print(f"[cyan]比較解析:[/cyan] {variant}")
        result = analyze_ser(
            input_path=target.ser_path,
            config=variant_config,
            output_dir=variant_output,
            start_frame=start_frame,
            end_frame=end_frame,
            max_frames=max_frames,
            preview=preview,
            show_progress=True,
            input_path_original=original_input_path,
            input_was_directory=input_was_directory,
            ser_selection_mode=ser_selection_mode,
            companion_files_copied=companion_files,
            output_root=target.output_root,
            auto_output_name=target.auto_output_name,
            input_manifest_path=variant_target.input_manifest_path,
        )
        _write_input_manifest(
            target=variant_target,
            config_path=config_path,
            zenith_deg=variant_config.instrument.zenith_deg,
            fps=variant_config.timing.fps,
            analysis_started_at=analysis_started_at,
            ser_metadata=result.summary.get("ser_metadata"),
            companion_files=companion_files,
        )
        rows.append(_comparison_row(variant, variant_config, result.summary))
    return rows


def _comparison_row(variant: str, config, summary):  # type: ignore[no-untyped-def]
    return {
        "variant": variant,
        "orientation_mode": summary.get("orientation_mode"),
        "orientation_angle_deg": summary.get("orientation_angle_deg"),
        "roi_size_px": config.roi.size_px,
        "large_jump_rejection": config.quality.reject_large_jump,
        "seeing_zenith_arcsec": summary.get("seeing_zenith_arcsec"),
        "r0_zenith_m": summary.get("r0_zenith_m"),
        "valid_frames": summary.get("valid_frames"),
        "frame_fit_success_rate": summary.get("frame_fit_success_rate"),
        "reject_reason_counts": json.dumps(
            summary.get("reject_reason_counts", {}),
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def _print_roi_safety(summary):  # type: ignore[no-untyped-def]
    status = summary.get("roi_safety_status")
    if status == "not_checked":
        console.print("[cyan]ROI safety check:[/cyan] disabled")
        return
    color = "green" if status == "safe" else "red" if status == "unsafe" else "yellow"
    console.print(
        f"[{color}]ROI safety: {str(status).upper()}[/{color}] "
        f"(ROI={summary.get('roi_size_px')} px, "
        f"half={summary.get('roi_half_size_px')} px, "
        f"margin={summary.get('roi_safety_margin_px')} px, "
        f"min edge={summary.get('min_edge_margin_px')} px)"
    )
    recommended = summary.get("recommended_max_roi_size_px")
    if recommended is not None:
        console.print(f"Recommended safe ROI size: {recommended} px")
    if summary.get("roi_auto_shrunk"):
        console.print("[yellow]ROI was auto-shrunk before analysis.[/yellow]")
        console.print(f"Requested ROI size: {summary.get('roi_original_size_px')} px")
        console.print(f"Using ROI size: {summary.get('roi_size_px')} px")
    if status == "unsafe":
        console.print("[red]WARNING: ROI is too close to the image edge.[/red]")
        console.print("- 撮影ROIを広げる、または2つの星像をより中央に寄せてください。")
        console.print("- 再撮影できない場合は、より小さい解析ROIを比較してください。")
    elif status == "warning":
        console.print(
            "[yellow]ROI edge margin is small; diagnostic plots should be checked.[/yellow]"
        )


def _print_reliability_warnings(summary):  # type: ignore[no-untyped-def]
    reliability = summary.get("result_reliability")
    if reliability == "good":
        console.print("[green]Result reliability: GOOD[/green]")
        return
    color = "red" if reliability == "bad" else "yellow"
    console.print(f"[{color}]WARNING: Result reliability is {str(reliability).upper()}.[/{color}]")
    flags = summary.get("quality_flags", {})
    if flags.get("low_fit_success_rate"):
        console.print(
            f"- frame fit success rate is {summary.get('frame_fit_success_rate', 0):.1%}"
        )
    if flags.get("high_saturation_fraction"):
        console.print(
            "- saturated frame fraction is high "
            f"({summary.get('saturated_frame_fraction', 0):.1%})"
        )
    if flags.get("too_few_valid_blocks"):
        console.print(f"- valid block count is only {summary.get('number_of_valid_blocks')}")
    if flags.get("orientation_unreliable"):
        console.print("- orientation is unreliable")
    if flags.get("relative_motion_outliers_detected"):
        count = summary.get("reject_reason_counts", {}).get("bad_relative_motion", 0)
        console.print(f"- relative motion outliers were detected ({count} frames)")
    if reliability in {"bad", "caution"}:
        console.print("Seeing value should be treated as provisional.")


if __name__ == "__main__":
    app()
