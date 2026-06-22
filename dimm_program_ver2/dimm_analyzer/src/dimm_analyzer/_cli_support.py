"""Pure output-path and row-building helpers for the Typer CLI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .exceptions import DimmAnalyzerError


@dataclass(frozen=True)
class OutputTarget:
    ser_path: Path
    output_root: Path
    output_dir: Path
    input_manifest_path: Path
    auto_output_name: bool


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
    # Collision precedence is part of the CLI contract: overwrite wins before run-id allocation.
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
