import csv
import json
from types import SimpleNamespace

from typer.testing import CliRunner

from dimm_analyzer import cli as cli_module
from dimm_analyzer.file_picker import SelectedAnalysisInputs

runner = CliRunner()


def _write_config(path, *, zenith="12.0"):
    path.write_text(f"instrument:\n  zenith_deg: {zenith}\n", encoding="utf-8")
    return path


def _fake_result():
    return SimpleNamespace(
        summary={
            "valid_frames": 1,
            "total_frames": 1,
            "frame_fit_success_rate": 1.0,
            "number_of_valid_blocks": 1,
            "estimated_mask_angle_deg": 0.0,
            "orientation_angle_deg": 0.0,
            "median_seeing_zenith_corrected_arcsec": 1.0,
            "median_r0_zenith_m": 0.1,
            "seeing_zenith_arcsec": 1.0,
            "r0_zenith_m": 0.1,
            "saturated_frame_fraction": 0.0,
            "reject_reason_counts": {},
            "quality_flags": {},
            "roi_safety_status": "safe",
            "roi_size_px": 25,
            "roi_half_size_px": 12.0,
            "roi_safety_margin_px": 5.0,
            "min_edge_margin_px": 25.0,
            "recommended_max_roi_size_px": 31,
            "roi_auto_shrunk": False,
            "result_reliability": "good",
            "warnings": [],
        }
    )


def test_cli_full_paths_does_not_call_selector(monkeypatch, tmp_path):
    science = tmp_path / "science.ser"
    science.write_bytes(b"")
    config = _write_config(tmp_path / "config.yaml")
    output = tmp_path / "out"
    captured = {}

    def fake_analyze_ser(**kwargs):
        captured.update(kwargs)
        return _fake_result()

    monkeypatch.setattr(cli_module, "analyze_ser", fake_analyze_ser)
    monkeypatch.setattr(
        cli_module,
        "select_analysis_paths",
        lambda **_: (_ for _ in ()).throw(AssertionError("selector should not be called")),
    )

    result = runner.invoke(
        cli_module.app,
        [
            "--input",
            str(science),
            "--config",
            str(config),
            "--output",
            str(output),
            "--zenith-deg",
            "12.0",
            "--roi-size",
            "25",
            "--roi-safety-margin",
            "3.5",
            "--disable-large-jump-rejection",
            "--orientation-mode",
            "manual",
            "--orientation-angle-deg",
            "90",
        ],
    )

    assert result.exit_code == 0
    assert captured["input_path"] == science
    assert captured["output_dir"] == output / "science"
    assert captured["output_root"] == output
    assert captured["auto_output_name"] is True
    assert captured["input_manifest_path"] == output / "science" / "input_manifest.json"
    assert captured["config"].roi.size_px == 25
    assert captured["config"].roi.safety_margin_px == 3.5
    assert not captured["config"].quality.reject_large_jump
    assert captured["config"].orientation.mode == "manual"
    assert captured["config"].orientation.mask_angle_deg == 90.0
    assert (output / "science" / "input_manifest.json").exists()
    with (output / "science" / "input_manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["input_ser_path"] == str(science)
    assert manifest["input_ser_filename"] == "science.ser"
    assert manifest["input_ser_stem"] == "science"
    assert manifest["output_root"] == str(output)
    assert manifest["output_dir"] == str(output / "science")
    assert manifest["config_path"] == str(config)
    assert manifest["zenith_deg"] == 12.0
    assert manifest["auto_output_name"] is True


def test_cli_no_auto_output_name_uses_output_directly(monkeypatch, tmp_path):
    science = tmp_path / "science.ser"
    science.write_bytes(b"")
    config = _write_config(tmp_path / "config.yaml")
    output = tmp_path / "out"
    captured = {}

    def fake_analyze_ser(**kwargs):
        captured.update(kwargs)
        return _fake_result()

    monkeypatch.setattr(cli_module, "analyze_ser", fake_analyze_ser)

    result = runner.invoke(
        cli_module.app,
        [
            "--input",
            str(science),
            "--config",
            str(config),
            "--output",
            str(output),
            "--zenith-deg",
            "12.0",
            "--no-auto-output-name",
        ],
    )

    assert result.exit_code == 0
    assert captured["output_dir"] == output
    assert captured["auto_output_name"] is False


def test_cli_existing_auto_output_dir_errors_by_default(monkeypatch, tmp_path):
    science = tmp_path / "science.ser"
    science.write_bytes(b"")
    config = _write_config(tmp_path / "config.yaml")
    output = tmp_path / "results"
    (output / "science").mkdir(parents=True)

    monkeypatch.setattr(
        cli_module,
        "analyze_ser",
        lambda **_: (_ for _ in ()).throw(AssertionError("analysis should not run")),
    )

    result = runner.invoke(
        cli_module.app,
        [
            "--input",
            str(science),
            "--config",
            str(config),
            "--output",
            str(output),
            "--zenith-deg",
            "12.0",
        ],
    )

    assert result.exit_code == 1
    assert "出力フォルダが既に存在します" in result.output


def test_cli_append_run_id_avoids_existing_output_dir(monkeypatch, tmp_path):
    science = tmp_path / "science.ser"
    science.write_bytes(b"")
    config = _write_config(tmp_path / "config.yaml")
    output = tmp_path / "results"
    (output / "science").mkdir(parents=True)
    (output / "science_002").mkdir()
    captured = {}

    def fake_analyze_ser(**kwargs):
        captured.update(kwargs)
        return _fake_result()

    monkeypatch.setattr(cli_module, "analyze_ser", fake_analyze_ser)

    result = runner.invoke(
        cli_module.app,
        [
            "--input",
            str(science),
            "--config",
            str(config),
            "--output",
            str(output),
            "--zenith-deg",
            "12.0",
            "--append-run-id",
        ],
    )

    assert result.exit_code == 0
    assert captured["output_dir"] == output / "science_003"
    assert (output / "science_003" / "input_manifest.json").exists()


def test_cli_overwrite_allows_existing_output_dir(monkeypatch, tmp_path):
    science = tmp_path / "science.ser"
    science.write_bytes(b"")
    config = _write_config(tmp_path / "config.yaml")
    output = tmp_path / "results"
    (output / "science").mkdir(parents=True)
    captured = {}

    def fake_analyze_ser(**kwargs):
        captured.update(kwargs)
        return _fake_result()

    monkeypatch.setattr(cli_module, "analyze_ser", fake_analyze_ser)

    result = runner.invoke(
        cli_module.app,
        [
            "--input",
            str(science),
            "--config",
            str(config),
            "--output",
            str(output),
            "--zenith-deg",
            "12.0",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0
    assert captured["output_dir"] == output / "science"


def test_cli_roi_size_larger_than_fallback_raises_fallback(monkeypatch, tmp_path):
    science = tmp_path / "science.ser"
    science.write_bytes(b"")
    config = _write_config(tmp_path / "config.yaml")
    output = tmp_path / "out"
    captured = {}

    def fake_analyze_ser(**kwargs):
        captured.update(kwargs)
        return _fake_result()

    monkeypatch.setattr(cli_module, "analyze_ser", fake_analyze_ser)

    result = runner.invoke(
        cli_module.app,
        [
            "--input",
            str(science),
            "--config",
            str(config),
            "--output",
            str(output),
            "--zenith-deg",
            "12.0",
            "--roi-size",
            "41",
        ],
    )

    assert result.exit_code == 0
    assert captured["config"].roi.size_px == 41
    assert captured["config"].roi.fallback_size_px == 41


def test_cli_missing_input_uses_selector(monkeypatch, tmp_path):
    science = tmp_path / "selected.ser"
    science.write_bytes(b"")
    config = _write_config(tmp_path / "config.yaml")
    output = tmp_path / "out"
    captured = {}

    def fake_selector(**kwargs):
        assert kwargs["input_path"] is None
        return SelectedAnalysisInputs(
            input_path=science,
            config_path=config,
            output_dir=output,
            dark_path=None,
            zenith_deg=12.0,
        )

    def fake_analyze_ser(**kwargs):
        captured.update(kwargs)
        return _fake_result()

    monkeypatch.setattr(cli_module, "select_analysis_paths", fake_selector)
    monkeypatch.setattr(cli_module, "analyze_ser", fake_analyze_ser)

    result = runner.invoke(
        cli_module.app,
        [
            "--config",
            str(config),
            "--output",
            str(output),
            "--zenith-deg",
            "12.0",
            "--picker-mode",
            "terminal",
        ],
    )

    assert result.exit_code == 0
    assert captured["input_path"] == science
    assert captured["output_dir"] == output / "selected"


def test_cli_folder_with_one_ser_auto_detects(monkeypatch, tmp_path):
    capture = tmp_path / "capture"
    science = capture / "science.ser"
    science.parent.mkdir()
    science.write_bytes(b"")
    config = _write_config(tmp_path / "config.yaml")
    output = tmp_path / "out"
    captured = {}

    def fake_analyze_ser(**kwargs):
        captured.update(kwargs)
        return _fake_result()

    monkeypatch.setattr(cli_module, "analyze_ser", fake_analyze_ser)

    result = runner.invoke(
        cli_module.app,
        [
            "--input",
            str(capture),
            "--config",
            str(config),
            "--output",
            str(output),
            "--zenith-deg",
            "12.0",
        ],
    )

    assert result.exit_code == 0
    assert captured["input_path"] == science
    assert captured["output_dir"] == output / "science"
    assert captured["input_path_original"] == capture
    assert captured["input_was_directory"] is True
    assert captured["ser_selection_mode"] == "error"


def test_cli_ser_select_all_writes_batch_outputs(monkeypatch, tmp_path):
    capture = tmp_path / "capture"
    first = capture / "00_31_02.ser"
    second = capture / "00_33_10.ser"
    first.parent.mkdir()
    first.write_bytes(b"")
    second.write_bytes(b"")
    (capture / "CameraSettings.txt").write_text("camera", encoding="utf-8")
    config = _write_config(tmp_path / "config.yaml")
    output = tmp_path / "batch"
    calls = []

    def fake_analyze_ser(**kwargs):
        calls.append(kwargs)
        kwargs["output_dir"].mkdir(parents=True, exist_ok=True)
        return _fake_result()

    monkeypatch.setattr(cli_module, "analyze_ser", fake_analyze_ser)

    result = runner.invoke(
        cli_module.app,
        [
            "--input",
            str(capture),
            "--config",
            str(config),
            "--output",
            str(output),
            "--zenith-deg",
            "12.0",
            "--ser-select",
            "all",
        ],
    )

    assert result.exit_code == 0
    assert [call["input_path"] for call in calls] == [first, second]
    assert [call["output_dir"] for call in calls] == [
        output / "00_31_02",
        output / "00_33_10",
    ]
    assert all(call["companion_files_copied"] == ["CameraSettings.txt"] for call in calls)
    assert (output / "00_31_02" / "input_metadata" / "CameraSettings.txt").exists()
    assert (output / "00_33_10" / "input_metadata" / "CameraSettings.txt").exists()
    assert (output / "batch_summary.csv").exists()
    assert (output / "batch_manifest.json").exists()
    with (output / "batch_summary.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0].keys()) == [
        "ser_file",
        "ser_path",
        "output_dir",
        "summary_json",
        "block_results_csv",
        "per_frame_fits_csv",
        "total_frames",
        "valid_frames",
        "frame_fit_success_rate",
        "number_of_valid_blocks",
        "median_seeing_zenith_corrected_arcsec",
        "median_r0_zenith_m",
        "result_reliability",
        "saturated_frame_fraction",
        "roi_safety_status",
        "orientation_angle_deg",
    ]
    assert rows[0]["ser_file"] == "00_31_02.ser"
    assert rows[0]["ser_path"] == str(first)
    assert rows[0]["output_dir"] == str(output / "00_31_02")
    assert rows[0]["summary_json"] == str(output / "00_31_02" / "summary.json")
    assert rows[0]["block_results_csv"] == str(output / "00_31_02" / "block_results.csv")
    assert rows[0]["per_frame_fits_csv"] == str(output / "00_31_02" / "per_frame_fits.csv")
    with (output / "batch_manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["input_path_original"] == str(capture)
    assert manifest["input_was_directory"] is True
    assert manifest["ser_select_mode"] == "all"
    assert manifest["number_of_ser_files"] == 2
    assert manifest["output_root"] == str(output)
    assert manifest["batch_summary_csv"] == str(output / "batch_summary.csv")
    assert len(manifest["ser_files"]) == 2


def test_cli_disable_roi_safety_check(monkeypatch, tmp_path):
    science = tmp_path / "science.ser"
    science.write_bytes(b"")
    config = _write_config(tmp_path / "config.yaml")
    output = tmp_path / "out"
    captured = {}

    def fake_analyze_ser(**kwargs):
        captured.update(kwargs)
        return _fake_result()

    monkeypatch.setattr(cli_module, "analyze_ser", fake_analyze_ser)

    result = runner.invoke(
        cli_module.app,
        [
            "--input",
            str(science),
            "--config",
            str(config),
            "--output",
            str(output),
            "--zenith-deg",
            "12.0",
            "--disable-roi-safety-check",
        ],
    )

    assert result.exit_code == 0
    assert captured["config"].roi.auto_safety_check is False
    assert captured["config"].roi.auto_shrink_if_unsafe is False


def test_cli_auto_shrink_roi_if_unsafe(monkeypatch, tmp_path):
    science = tmp_path / "science.ser"
    science.write_bytes(b"")
    config = _write_config(tmp_path / "config.yaml")
    output = tmp_path / "out"
    captured = {}

    def fake_analyze_ser(**kwargs):
        captured.update(kwargs)
        return _fake_result()

    monkeypatch.setattr(cli_module, "analyze_ser", fake_analyze_ser)

    result = runner.invoke(
        cli_module.app,
        [
            "--input",
            str(science),
            "--config",
            str(config),
            "--output",
            str(output),
            "--zenith-deg",
            "12.0",
            "--auto-shrink-roi-if-unsafe",
        ],
    )

    assert result.exit_code == 0
    assert captured["config"].roi.auto_safety_check is True
    assert captured["config"].roi.auto_shrink_if_unsafe is True


def test_cli_roi_safety_disable_and_auto_shrink_conflict(tmp_path):
    science = tmp_path / "science.ser"
    science.write_bytes(b"")
    config = _write_config(tmp_path / "config.yaml")
    output = tmp_path / "out"

    result = runner.invoke(
        cli_module.app,
        [
            "--input",
            str(science),
            "--config",
            str(config),
            "--output",
            str(output),
            "--zenith-deg",
            "12.0",
            "--disable-roi-safety-check",
            "--auto-shrink-roi-if-unsafe",
        ],
    )

    assert result.exit_code == 1
    assert "同時指定できません" in result.output


def test_cli_comparison_suite_writes_summary(monkeypatch, tmp_path):
    science = tmp_path / "science.ser"
    science.write_bytes(b"")
    config = _write_config(tmp_path / "config.yaml")
    output = tmp_path / "out"
    calls = []

    def fake_analyze_ser(**kwargs):
        calls.append(kwargs)
        kwargs["output_dir"].mkdir(parents=True, exist_ok=True)
        return _fake_result()

    monkeypatch.setattr(cli_module, "analyze_ser", fake_analyze_ser)

    result = runner.invoke(
        cli_module.app,
        [
            "--input",
            str(science),
            "--config",
            str(config),
            "--output",
            str(output),
            "--zenith-deg",
            "12.0",
            "--comparison-suite",
        ],
    )

    assert result.exit_code == 0
    assert len(calls) == 7
    assert [call["config"].roi.size_px for call in calls] == [25, 25, 25, 25, 21, 31, 25]
    assert (output / "science" / "comparison_summary.csv").exists()
