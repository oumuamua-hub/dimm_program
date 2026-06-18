"""Matplotlib plot generation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List

import numpy as np

from .models import AnalysisResult, FrameResult


def save_plots(result: AnalysisResult, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    _configure_japanese_font(matplotlib)
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_block_metric(
        output_dir / "seeing_timeseries.png",
        result.block_results,
        "seeing_mean_zenith_arcsec",
        "シーイング (arcsec)",
    )
    _plot_block_metric(
        output_dir / "r0_timeseries.png",
        result.block_results,
        "r0_mean_zenith_m",
        "天頂補正後 r0 (m)",
    )
    _plot_block_metric(
        output_dir / "fit_success_rate_timeseries.png",
        result.block_results,
        "frame_fit_success_rate",
        "frame fit 成功率",
    )
    _plot_dx_dy(output_dir / "dx_dy_timeseries.png", result.frame_results)
    _plot_centroid_scatter(output_dir / "centroid_scatter.png", result.frame_results)
    _plot_fwhm(output_dir / "fwhm_timeseries.png", result.frame_results)
    _plot_rejections(output_dir / "rejection_summary.png", result.frame_results)
    _plot_peak_timeseries(output_dir / "peak_timeseries.png", result)
    _plot_peak_histogram(output_dir / "peak_histogram.png", result)
    _plot_separation_timeseries(output_dir / "separation_timeseries.png", result.frame_results)
    _plot_relative_motion_outliers(output_dir / "relative_motion_outliers.png", result)
    _plot_orientation_diagnostics(output_dir / "orientation_diagnostics.png", result)
    _plot_roi_safety_timeseries(output_dir / "roi_safety_timeseries.png", result)
    _plot_valid_rejected_histogram(
        output_dir / "valid_vs_rejected_peak_histogram.png",
        result.frame_results,
        ["peak1", "peak2"],
        "peak ADU",
    )
    _plot_valid_rejected_histogram(
        output_dir / "valid_vs_rejected_fwhm_histogram.png",
        result.frame_results,
        ["fwhm_mean1", "fwhm_mean2"],
        "FWHM (px)",
    )
    _plot_valid_rejected_histogram(
        output_dir / "valid_vs_rejected_separation_histogram.png",
        result.frame_results,
        ["separation_px"],
        "separation (px)",
    )
    if result.example_success_roi is not None:
        _plot_roi(output_dir / "example_fit_success.png", result.example_success_roi, "fit 成功例")
    if result.example_failed_roi is not None:
        _plot_roi(
            output_dir / "example_fit_failed.png",
            result.example_failed_roi,
            "fit 失敗例",
        )
    plt.close("all")


def _plot_block_metric(path: Path, blocks, metric: str, ylabel: str) -> None:  # type: ignore[no-untyped-def]
    import matplotlib.pyplot as plt

    x = [
        block.time_start_sec if block.time_start_sec is not None else block.block_index
        for block in blocks
    ]
    y = [getattr(block, metric) for block in blocks]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y, marker="o", linewidth=1.5)
    ax.set_xlabel("時刻 (s) または block index")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_dx_dy(path: Path, frames: List[FrameResult]) -> None:
    import matplotlib.pyplot as plt

    x = _frame_axis(frames)
    dx = [frame.dx_px for frame in frames]
    dy = [frame.dy_px for frame in frames]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, dx, label="dx", linewidth=1)
    ax.plot(x, dy, label="dy", linewidth=1)
    outliers = [frame for frame in frames if frame.reject_reason == "bad_relative_motion"]
    if outliers:
        x_bad = _frame_axis(outliers)
        ax.scatter(
            x_bad,
            [frame.dx_px for frame in outliers],
            s=16,
            label="bad dx",
            color="tab:red",
        )
        ax.scatter(
            x_bad,
            [frame.dy_px for frame in outliers],
            s=16,
            label="bad dy",
            color="tab:pink",
        )
    ax.set_xlabel("時刻 (s) または frame index")
    ax.set_ylabel("移動量 (px)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_centroid_scatter(path: Path, frames: List[FrameResult]) -> None:
    import matplotlib.pyplot as plt

    valid = [frame for frame in frames if frame.frame_valid]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter([frame.x1 for frame in valid], [frame.y1 for frame in valid], s=8, label="星像 1")
    ax.scatter([frame.x2 for frame in valid], [frame.y2 for frame in valid], s=8, label="星像 2")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_fwhm(path: Path, frames: List[FrameResult]) -> None:
    import matplotlib.pyplot as plt

    x = _frame_axis(frames)
    fwhm1 = [frame.fit1.fwhm_mean for frame in frames]
    fwhm2 = [frame.fit2.fwhm_mean for frame in frames]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, fwhm1, label="星像 1", linewidth=1)
    ax.plot(x, fwhm2, label="星像 2", linewidth=1)
    bad = [
        frame
        for frame in frames
        if frame.reject_reason in {"bad_fwhm", "fwhm_outlier"}
    ]
    if bad:
        x_bad = _frame_axis(bad)
        ax.scatter(
            x_bad,
            [frame.fit1.fwhm_mean for frame in bad],
            s=16,
            label="異常 FWHM 1",
            color="tab:red",
        )
        ax.scatter(
            x_bad,
            [frame.fit2.fwhm_mean for frame in bad],
            s=16,
            label="異常 FWHM 2",
            color="tab:pink",
        )
    ax.set_xlabel("時刻 (s) または frame index")
    ax.set_ylabel("FWHM (px)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_rejections(path: Path, frames: List[FrameResult]) -> None:
    import matplotlib.pyplot as plt

    counts = Counter(frame.reject_reason for frame in frames if not frame.frame_valid)
    labels = list(counts.keys()) or ["none"]
    values = [counts[label] for label in labels] or [0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values)
    ax.set_ylabel("frame 数")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_roi(path: Path, roi: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    image = ax.imshow(roi, origin="lower", cmap="viridis")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_peak_timeseries(path: Path, result: AnalysisResult) -> None:
    import matplotlib.pyplot as plt

    frames = result.frame_results
    x = _frame_axis(frames)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, [frame.fit1.peak for frame in frames], label="peak1", linewidth=1)
    ax.plot(x, [frame.fit2.peak for frame in frames], label="peak2", linewidth=1)
    threshold = result.summary.get("saturation_threshold")
    if threshold is not None:
        ax.axhline(float(threshold), color="tab:red", linestyle="--", label="saturation threshold")
    ax.set_xlabel("時刻 (s) または frame index")
    ax.set_ylabel("peak ADU")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_peak_histogram(path: Path, result: AnalysisResult) -> None:
    import matplotlib.pyplot as plt

    peak1 = _finite_values([frame.fit1.peak for frame in result.frame_results])
    peak2 = _finite_values([frame.fit2.peak for frame in result.frame_results])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if peak1.size:
        ax.hist(peak1, bins=40, alpha=0.6, label="peak1")
    if peak2.size:
        ax.hist(peak2, bins=40, alpha=0.6, label="peak2")
    threshold = result.summary.get("saturation_threshold")
    if threshold is not None:
        ax.axvline(float(threshold), color="tab:red", linestyle="--", label="saturation threshold")
    ax.set_xlabel("peak ADU")
    ax.set_ylabel("frame 数")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_separation_timeseries(path: Path, frames: List[FrameResult]) -> None:
    import matplotlib.pyplot as plt

    x = _frame_axis(frames)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, [frame.separation_px for frame in frames], linewidth=1)
    ax.set_xlabel("時刻 (s) または frame index")
    ax.set_ylabel("separation (px)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_relative_motion_outliers(path: Path, result: AnalysisResult) -> None:
    import matplotlib.pyplot as plt

    frames = result.frame_results
    x = _frame_axis(frames)
    dx_delta = [frame.relative_delta_dx_px for frame in frames]
    dy_delta = [frame.relative_delta_dy_px for frame in frames]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, dx_delta, label="|dx - median dx|", linewidth=1)
    ax.plot(x, dy_delta, label="|dy - median dy|", linewidth=1)
    threshold = result.summary.get("config_used", {}).get("quality", {}).get(
        "max_relative_motion_deviation_px"
    )
    if threshold is not None:
        ax.axhline(float(threshold), color="tab:red", linestyle="--", label="threshold")
    ax.set_xlabel("時刻 (s) または frame index")
    ax.set_ylabel("relative motion deviation (px)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_orientation_diagnostics(path: Path, result: AnalysisResult) -> None:
    import matplotlib.pyplot as plt

    rows = result.orientation_scan_rows
    x = [row.get("angle_deg") for row in rows]
    seeing = [row.get("seeing_mean_arcsec") for row in rows]
    mismatch = [row.get("mismatch_abs_log_ratio") for row in rows]
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(x, seeing, label="mean seeing", color="tab:blue", linewidth=1)
    ax1.set_xlabel("angle (deg)")
    ax1.set_ylabel("mean seeing (arcsec)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(x, mismatch, label="mismatch", color="tab:orange", linewidth=1)
    ax2.set_ylabel("mismatch abs log ratio", color="tab:orange")
    angle = result.summary.get("orientation_angle_deg")
    if angle is not None:
        ax1.axvline(float(angle), color="tab:red", linestyle="--", label="selected")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_roi_safety_timeseries(path: Path, result: AnalysisResult) -> None:
    import matplotlib.pyplot as plt

    rows = result.roi_safety_rows
    if result.summary.get("roi_safety_status") == "not_checked" or not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for spot_id in sorted({row.get("spot_id") for row in rows}):
        spot_rows = [row for row in rows if row.get("spot_id") == spot_id]
        x = [
            float(row["time_sec"] if row.get("time_sec") is not None else row["frame_index"])
            for row in spot_rows
        ]
        y = [row.get("min_margin_px") for row in spot_rows]
        ax.plot(x, y, linewidth=1, label=f"spot {spot_id}")
    half_roi_px = result.summary.get("roi_half_size_px")
    required_margin_px = result.summary.get("roi_required_margin_px")
    if half_roi_px is not None:
        ax.axhline(float(half_roi_px), color="tab:red", linestyle="--", label="half ROI")
    if required_margin_px is not None:
        ax.axhline(
            float(required_margin_px),
            color="tab:orange",
            linestyle="--",
            label="required margin",
        )
    below_half = [row for row in rows if row.get("below_half_roi")]
    if below_half:
        ax.scatter(
            _row_axis(below_half),
            [row.get("min_margin_px") for row in below_half],
            s=24,
            color="tab:red",
            label="below half ROI",
            zorder=4,
        )
    suspected = [row for row in rows if row.get("suspected_outlier")]
    if suspected:
        ax.scatter(
            _row_axis(suspected),
            [row.get("min_margin_px") for row in suspected],
            s=34,
            facecolors="none",
            edgecolors="black",
            label="suspected outlier",
            zorder=5,
        )
    p05 = result.summary.get("edge_margin_reliable_p05_px")
    recommended = result.summary.get("recommended_max_roi_size_px")
    if p05 is not None:
        ax.axhline(float(p05), color="tab:green", linestyle=":", label="reliable p05")
    if p05 is not None or recommended is not None:
        ax.text(
            0.01,
            0.98,
            f"p05={p05:.2f}px, recommended ROI={recommended}px"
            if p05 is not None
            else f"recommended ROI={recommended}px",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    ax.set_xlabel("時刻 (s) または frame index")
    ax.set_ylabel("画像端までの最小距離 (px)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _row_axis(rows) -> List[float]:  # type: ignore[no-untyped-def]
    return [
        float(row["time_sec"] if row.get("time_sec") is not None else row["frame_index"])
        for row in rows
    ]


def _plot_valid_rejected_histogram(
    path: Path,
    frames: List[FrameResult],
    metrics: List[str],
    xlabel: str,
) -> None:
    import matplotlib.pyplot as plt

    valid_values = _metric_values([frame for frame in frames if frame.frame_valid], metrics)
    rejected_values = _metric_values([frame for frame in frames if not frame.frame_valid], metrics)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if valid_values.size:
        ax.hist(valid_values, bins=40, alpha=0.6, label="valid")
    if rejected_values.size:
        ax.hist(rejected_values, bins=40, alpha=0.6, label="rejected")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _frame_axis(frames: List[FrameResult]) -> List[float]:
    axis: List[float] = []
    for frame in frames:
        axis.append(float(frame.time_sec if frame.time_sec is not None else frame.frame_index))
    return axis


def _metric_values(frames: List[FrameResult], metrics: List[str]) -> np.ndarray:
    values = []
    for frame in frames:
        for metric in metrics:
            if metric == "separation_px":
                value = frame.separation_px
            elif metric.endswith("1"):
                value = getattr(frame.fit1, metric[:-1], None)
            elif metric.endswith("2"):
                value = getattr(frame.fit2, metric[:-1], None)
            else:
                value = None
            if value is not None and np.isfinite(value):
                values.append(float(value))
    return np.asarray(values, dtype=float)


def _finite_values(values) -> np.ndarray:  # type: ignore[no-untyped-def]
    return np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(value)],
        dtype=float,
    )


def _configure_japanese_font(matplotlib) -> None:  # type: ignore[no-untyped-def]
    from matplotlib import font_manager

    candidates = []
    for pattern in (
        "/System/Library/Fonts/*角*.ttc",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/*NotoSansCJK*.otf",
        "/Library/Fonts/*NotoSansCJK*.ttc",
    ):
        candidates.extend(Path("/").glob(pattern.lstrip("/")))

    for font_path in candidates:
        try:
            font_manager.fontManager.addfont(str(font_path))
            family = font_manager.FontProperties(fname=str(font_path)).get_name()
        except Exception:
            continue
        matplotlib.rcParams["font.family"] = [family, "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        return

    matplotlib.rcParams["font.family"] = [
        "Hiragino Sans",
        "Yu Gothic",
        "Meiryo",
        "Noto Sans CJK JP",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
