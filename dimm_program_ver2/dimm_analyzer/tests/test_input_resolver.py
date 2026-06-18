import os

import pytest

from dimm_analyzer.exceptions import ConfigError
from dimm_analyzer.input_resolver import copy_companion_files, resolve_ser_inputs


def _touch(path, content=b""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_input_ser_file_is_used_directly(tmp_path):
    science = _touch(tmp_path / "science.ser")

    resolved = resolve_ser_inputs(science)

    assert resolved.ser_paths == [science]
    assert resolved.input_was_directory is False
    assert resolved.ser_selection_mode == "error"


def test_folder_with_one_ser_is_auto_detected(tmp_path):
    science = _touch(tmp_path / "capture" / "science.ser")

    resolved = resolve_ser_inputs(tmp_path / "capture")

    assert resolved.ser_paths == [science]
    assert resolved.input_was_directory is True


def test_folder_with_zero_ser_raises(tmp_path):
    capture = tmp_path / "capture"
    capture.mkdir()

    with pytest.raises(ConfigError, match="SER ファイルが見つかりません"):
        resolve_ser_inputs(capture)


def test_folder_multiple_ser_error_mode_raises(tmp_path):
    capture = tmp_path / "capture"
    _touch(capture / "a.ser")
    _touch(capture / "b.ser")

    with pytest.raises(ConfigError, match="複数の SER"):
        resolve_ser_inputs(capture, ser_select="error")


def test_folder_newest_selects_newest_ser(tmp_path):
    capture = tmp_path / "capture"
    old = _touch(capture / "old.ser")
    new = _touch(capture / "new.ser")
    os.utime(old, (1000, 1000))
    os.utime(new, (2000, 2000))

    resolved = resolve_ser_inputs(capture, ser_select="newest")

    assert resolved.ser_paths == [new]


def test_folder_largest_selects_largest_ser(tmp_path):
    capture = tmp_path / "capture"
    small = _touch(capture / "small.ser", b"1")
    large = _touch(capture / "large.ser", b"12345")

    resolved = resolve_ser_inputs(capture, ser_select="largest")

    assert resolved.ser_paths == [large]
    assert small not in resolved.ser_paths


def test_folder_all_returns_all_targets(tmp_path):
    capture = tmp_path / "capture"
    first = _touch(capture / "a.ser")
    second = _touch(capture / "nested" / "b.ser")

    resolved = resolve_ser_inputs(capture, ser_select="all")

    assert resolved.ser_paths == [first, second]


def test_no_recursive_ignores_nested_ser(tmp_path):
    capture = tmp_path / "capture"
    first = _touch(capture / "a.ser")
    _touch(capture / "nested" / "b.ser")

    resolved = resolve_ser_inputs(capture, recursive=False, ser_select="all")

    assert resolved.ser_paths == [first]


def test_companion_files_are_copied_without_ser(tmp_path):
    capture = tmp_path / "capture"
    settings = _touch(capture / "CaptureSettings.txt", b"settings")
    camera = _touch(capture / "nested" / "Camera.ini", b"camera")
    _touch(capture / "science.ser", b"ser")
    _touch(capture / "ignore.fits", b"fits")

    copied = copy_companion_files(source_root=capture, output_dir=tmp_path / "out")

    assert copied == [
        settings.relative_to(capture).as_posix(),
        camera.relative_to(capture).as_posix(),
    ]
    assert (tmp_path / "out" / "input_metadata" / "CaptureSettings.txt").exists()
    assert (tmp_path / "out" / "input_metadata" / "nested" / "Camera.ini").exists()
    assert not (tmp_path / "out" / "input_metadata" / "science.ser").exists()
