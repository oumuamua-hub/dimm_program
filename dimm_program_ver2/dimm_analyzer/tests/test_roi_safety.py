import pytest

from dimm_analyzer.roi_safety import (
    build_roi_safety_point,
    finalize_roi_safety_rows,
    recommended_max_roi_size,
    roi_half_size,
    roi_safety_status,
    shrink_roi_sizes_for_safety,
    summarize_roi_safety,
)


def _rows_for_min_margin(min_margin):
    point = build_roi_safety_point(
        frame_index=0,
        time_sec=0.0,
        spot_id=1,
        x=min_margin,
        y=50.0,
        width=100,
        height=100,
    )
    assert point is not None
    return finalize_roi_safety_rows([point], roi_size_px=25, safety_margin_px=5.0)


def _rows_for_margins(margins):
    points = []
    for idx, margin in enumerate(margins):
        point = build_roi_safety_point(
            frame_index=idx,
            time_sec=float(idx),
            spot_id=1,
            x=margin,
            y=50.0,
            width=100,
            height=100,
        )
        assert point is not None
        point["included_in_reliable_population"] = True
        points.append(point)
    return finalize_roi_safety_rows(points, roi_size_px=25, safety_margin_px=5.0)


def _summary_for_margins(margins):
    summary, _ = summarize_roi_safety(
        _rows_for_margins(margins),
        roi_size_px=25,
        roi_fallback_size_px=31,
        safety_margin_px=5.0,
        max_allowed_size_px=41,
        enabled=True,
        total_frame_count=len(margins),
    )
    return summary


def test_roi_half_size():
    assert roi_half_size(25) == pytest.approx(12.0)


def test_roi_safety_status_safe_warning_unsafe():
    assert roi_safety_status(18.0, roi_size_px=25, safety_margin_px=5.0) == "safe"
    assert roi_safety_status(14.0, roi_size_px=25, safety_margin_px=5.0) == "warning"
    assert roi_safety_status(11.0, roi_size_px=25, safety_margin_px=5.0) == "unsafe"


def test_recommended_max_roi_size_is_odd_and_bounded():
    assert recommended_max_roi_size(20.4, 5.0, 41) == 31
    assert recommended_max_roi_size(100.0, 5.0, 41) == 41
    assert recommended_max_roi_size(2.0, 5.0, 41) == 1


def test_single_extreme_outlier_does_not_force_recommended_roi_to_one():
    summary = _summary_for_margins([0.5] + [30.0] * 99)

    assert summary["edge_margin_all_min_px"] == 0.5
    assert summary["recommended_max_roi_size_px_from_min"] == 1
    assert summary["recommended_max_roi_size_px"] > 1
    assert summary["edge_margin_absolute_outlier_detected"] is True


def test_many_frames_near_edge_is_unsafe():
    summary = _summary_for_margins([10.0] * 20 + [30.0] * 80)

    assert summary["roi_safety_status"] == "unsafe"
    assert summary["edge_margin_reliable_p05_px"] < summary["roi_half_size_px"]


def test_few_frames_below_required_margin_is_warning():
    summary = _summary_for_margins([14.0] * 3 + [30.0] * 97)

    assert summary["roi_safety_status"] == "warning"
    assert 0 < summary["edge_margin_below_required_fraction"] <= 0.05


def test_p05_meets_required_margin_is_safe():
    summary = _summary_for_margins([18.0] * 100)

    assert summary["roi_safety_status"] == "safe"
    assert summary["edge_margin_reliable_p05_px"] >= summary["roi_required_margin_px"]


def test_shrink_roi_sizes_for_safety_caps_fallback():
    size_px, fallback_size_px = shrink_roi_sizes_for_safety(
        requested_size_px=31,
        requested_fallback_size_px=31,
        recommended_size_px=25,
        min_allowed_size_px=21,
        max_allowed_size_px=41,
    )
    assert size_px == 25
    assert fallback_size_px == 25
