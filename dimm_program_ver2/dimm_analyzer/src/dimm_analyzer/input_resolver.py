"""SER file/directory input resolution and companion metadata copying."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .exceptions import ConfigError

SERSelectMode = str

_SER_SELECT_MODES = {"error", "newest", "largest", "all"}
_COMPANION_SUFFIXES = {".txt", ".ini", ".json", ".yaml", ".csv"}
_COMPANION_NAME_MARKERS = ("settings", "capture", "camera")


@dataclass(frozen=True)
class ResolvedSERInputs:
    original_input_path: Path
    input_was_directory: bool
    ser_selection_mode: SERSelectMode
    ser_paths: List[Path]


def resolve_ser_inputs(
    input_path: Path,
    *,
    recursive: bool = True,
    ser_select: SERSelectMode = "error",
) -> ResolvedSERInputs:
    """Resolve a CLI input path into one or more concrete SER files."""

    mode = _normalize_ser_select(ser_select)
    original = Path(input_path).expanduser()
    if original.is_file():
        if original.suffix.lower() != ".ser":
            raise ConfigError(f"入力ファイルは .ser である必要があります: {original}")
        return ResolvedSERInputs(
            original_input_path=original,
            input_was_directory=False,
            ser_selection_mode=mode,
            ser_paths=[original],
        )
    if not original.is_dir():
        raise ConfigError(f"入力パスが見つかりません: {original}")

    ser_files = find_ser_files(original, recursive=recursive)
    if not ser_files:
        raise ConfigError(f"入力ディレクトリに SER ファイルが見つかりません: {original}")
    if len(ser_files) == 1:
        selected = ser_files
    elif mode == "error":
        listing = "\n".join(f"- {path}" for path in ser_files)
        raise ConfigError(
            "入力ディレクトリに複数の SER ファイルがあります。"
            "--ser-select newest/largest/all を指定してください。\n"
            f"{listing}"
        )
    elif mode == "newest":
        selected = [_newest_ser_file(ser_files)]
    elif mode == "largest":
        selected = [_largest_ser_file(ser_files)]
    else:
        selected = ser_files

    return ResolvedSERInputs(
        original_input_path=original,
        input_was_directory=True,
        ser_selection_mode=mode,
        ser_paths=selected,
    )


def find_ser_files(directory: Path, *, recursive: bool = True) -> List[Path]:
    iterator: Iterable[Path]
    if recursive:
        iterator = directory.rglob("*")
    else:
        iterator = directory.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() == ".ser"
    )


def companion_source_root(original_input_path: Path, resolved_ser_path: Path) -> Path:
    if original_input_path.is_dir():
        return original_input_path
    return resolved_ser_path.parent


def find_companion_files(
    source_root: Path,
    *,
    recursive: bool = True,
    excluded_roots: Optional[Sequence[Path]] = None,
) -> List[Path]:
    """Find likely SharpCap sidecar files under a capture directory."""

    excluded = [root.expanduser().resolve() for root in (excluded_roots or []) if root.exists()]
    iterator: Iterable[Path]
    if recursive:
        iterator = source_root.rglob("*")
    else:
        iterator = source_root.iterdir()
    companions: List[Path] = []
    for path in iterator:
        if not path.is_file():
            continue
        if path.suffix.lower() == ".ser":
            continue
        resolved = path.resolve()
        if any(_is_relative_to(resolved, root) for root in excluded):
            continue
        if _is_companion_file(path):
            companions.append(path)
    return sorted(companions)


def copy_companion_files(
    *,
    source_root: Path,
    output_dir: Path,
    recursive: bool = True,
    excluded_roots: Optional[Sequence[Path]] = None,
) -> List[str]:
    """Copy SharpCap companion files to output/input_metadata and return relative names."""

    companions = find_companion_files(
        source_root,
        recursive=recursive,
        excluded_roots=[output_dir, *(excluded_roots or [])],
    )
    if not companions:
        return []

    metadata_dir = output_dir / "input_metadata"
    copied: List[str] = []
    for source in companions:
        relative = _relative_to_or_name(source, source_root)
        destination = metadata_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(relative.as_posix())
    return copied


def _normalize_ser_select(mode: SERSelectMode) -> SERSelectMode:
    value = mode.lower()
    if value not in _SER_SELECT_MODES:
        raise ConfigError("--ser-select は error, newest, largest, all のいずれかにしてください。")
    return value


def _newest_ser_file(paths: Sequence[Path]) -> Path:
    return max(paths, key=lambda path: (path.stat().st_mtime, path.name))


def _largest_ser_file(paths: Sequence[Path]) -> Path:
    return max(paths, key=lambda path: (path.stat().st_size, path.name))


def _is_companion_file(path: Path) -> bool:
    name = path.name.lower()
    return (
        path.suffix.lower() in _COMPANION_SUFFIXES
        or any(marker in name for marker in _COMPANION_NAME_MARKERS)
    )


def _relative_to_or_name(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return Path(path.name)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
