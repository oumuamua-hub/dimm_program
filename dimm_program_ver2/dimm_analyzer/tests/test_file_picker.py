from pathlib import Path

from dimm_analyzer.config import AnalysisConfig
from dimm_analyzer.file_picker import select_analysis_paths


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def test_all_paths_specified_does_not_call_picker(monkeypatch, tmp_path):
    science = _touch(tmp_path / "science.ser")
    config_path = _touch(tmp_path / "config.yaml")
    output = tmp_path / "results"
    config = AnalysisConfig()
    config.instrument.zenith_deg = 12.0

    monkeypatch.setattr(
        "dimm_analyzer.file_picker._select_file_dialog",
        lambda **_: (_ for _ in ()).throw(AssertionError("dialog should not be called")),
    )
    monkeypatch.setattr(
        "dimm_analyzer.file_picker._select_file_terminal",
        lambda **_: (_ for _ in ()).throw(AssertionError("terminal should not be called")),
    )
    monkeypatch.setattr(
        "dimm_analyzer.file_picker._select_optional_dark",
        lambda *_: (_ for _ in ()).throw(AssertionError("dark picker should not be called")),
    )

    selected = select_analysis_paths(
        input_path=science,
        config_path=config_path,
        output_dir=output,
        dark_path=None,
        zenith_deg=12.0,
        config=config,
        picker_mode="auto",
    )

    assert selected.input_path == science
    assert selected.config_path == config_path
    assert selected.output_dir == output
    assert selected.dark_path is None
    assert selected.zenith_deg == 12.0


def test_input_directory_is_allowed(tmp_path):
    capture = tmp_path / "capture"
    capture.mkdir()
    config_path = _touch(tmp_path / "config.yaml")
    output = tmp_path / "results"
    config = AnalysisConfig()
    config.instrument.zenith_deg = 12.0

    selected = select_analysis_paths(
        input_path=capture,
        config_path=config_path,
        output_dir=output,
        dark_path=None,
        zenith_deg=12.0,
        config=config,
        picker_mode="terminal",
    )

    assert selected.input_path == capture


def test_missing_input_uses_selector(monkeypatch, tmp_path):
    science = _touch(tmp_path / "selected.ser")
    config_path = _touch(tmp_path / "config.yaml")
    output = tmp_path / "results"
    config = AnalysisConfig()
    config.instrument.zenith_deg = 12.0

    monkeypatch.setattr(
        "dimm_analyzer.file_picker._select_file_or_directory_terminal",
        lambda **kwargs: science if kwargs["prompt"] == "解析対象の science SER" else config_path,
    )
    monkeypatch.setattr("dimm_analyzer.file_picker._select_optional_dark", lambda *_: None)

    selected = select_analysis_paths(
        input_path=None,
        config_path=config_path,
        output_dir=output,
        dark_path=None,
        zenith_deg=None,
        config=config,
        picker_mode="terminal",
    )

    assert selected.input_path == science
    assert selected.dark_path is None
    assert selected.zenith_deg is None


def test_dark_selection_can_be_skipped(monkeypatch, tmp_path):
    science = _touch(tmp_path / "science.ser")
    selected_science = _touch(tmp_path / "selected.ser")
    config_path = _touch(tmp_path / "config.yaml")
    output = tmp_path / "results"
    config = AnalysisConfig()
    config.instrument.zenith_deg = 12.0

    monkeypatch.setattr(
        "dimm_analyzer.file_picker._select_file_or_directory_terminal",
        lambda **_: selected_science,
    )
    monkeypatch.setattr("dimm_analyzer.file_picker._select_optional_dark", lambda *_: None)

    selected = select_analysis_paths(
        input_path=None,
        config_path=config_path,
        output_dir=output,
        dark_path=None,
        zenith_deg=12.0,
        config=config,
        picker_mode="terminal",
    )

    assert selected.input_path == selected_science
    assert selected.input_path != science
    assert selected.dark_path is None


def test_missing_zenith_prompts_for_override(monkeypatch, tmp_path):
    science = _touch(tmp_path / "science.ser")
    config_path = _touch(tmp_path / "config.yaml")
    output = tmp_path / "results"
    config = AnalysisConfig()

    monkeypatch.setattr("dimm_analyzer.file_picker._select_optional_dark", lambda *_: None)
    monkeypatch.setattr("dimm_analyzer.file_picker._prompt_zenith_deg", lambda: 23.5)

    selected = select_analysis_paths(
        input_path=science,
        config_path=config_path,
        output_dir=output,
        dark_path=None,
        zenith_deg=None,
        config=config,
        picker_mode="terminal",
    )

    assert selected.zenith_deg == 23.5
