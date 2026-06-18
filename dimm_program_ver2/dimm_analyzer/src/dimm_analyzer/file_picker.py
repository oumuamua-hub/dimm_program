"""CLI 補助用のファイル/ディレクトリ選択処理。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer

from .config import AnalysisConfig, load_config
from .exceptions import ConfigError

PickerMode = str


@dataclass
class SelectedAnalysisInputs:
    input_path: Path
    config_path: Path
    output_dir: Path
    dark_path: Optional[Path]
    zenith_deg: Optional[float]


def provided_paths_are_complete(
    *,
    input_path: Optional[Path],
    config_path: Optional[Path],
    output_dir: Optional[Path],
    zenith_deg: Optional[float],
    config: AnalysisConfig,
) -> bool:
    return (
        input_path is not None
        and config_path is not None
        and output_dir is not None
        and (zenith_deg is not None or not _config_zenith_missing(config))
    )


def resolve_provided_paths(
    *,
    input_path: Path,
    config_path: Path,
    output_dir: Path,
    dark_path: Optional[Path],
    zenith_deg: Optional[float],
) -> SelectedAnalysisInputs:
    return SelectedAnalysisInputs(
        input_path=_require_existing_file_or_directory(
            input_path,
            "science SER または capture folder",
        ),
        config_path=_require_existing_file(config_path, "YAML 設定ファイル"),
        output_dir=Path(output_dir).expanduser(),
        dark_path=_require_existing_file(dark_path, "dark SER") if dark_path else None,
        zenith_deg=zenith_deg,
    )


def select_analysis_paths(
    *,
    input_path: Optional[Path],
    config_path: Optional[Path],
    output_dir: Optional[Path],
    dark_path: Optional[Path],
    zenith_deg: Optional[float],
    config: Optional[AnalysisConfig],
    picker_mode: PickerMode,
) -> SelectedAnalysisInputs:
    """未指定の入力を dialog または terminal prompt で解決する。"""

    mode = _normalize_picker_mode(picker_mode)
    required_missing = input_path is None or config_path is None or output_dir is None
    if input_path is None:
        input_path = _select_file_or_directory(
            title="解析対象の science SER を選択",
            directory_title="SharpCap capture folder を選択",
            filetypes=(("SER files", "*.ser"), ("All files", "*.*")),
            mode=mode,
            terminal_prompt="解析対象の science SER",
            required=True,
        )
    if config_path is None:
        config_path = _select_file(
            title="YAML 設定ファイルを選択",
            filetypes=(("YAML files", "*.yaml *.yml"), ("All files", "*.*")),
            mode=mode,
            terminal_prompt="YAML 設定ファイル",
            required=True,
        )
    if output_dir is None:
        output_dir = _select_directory(
            title="出力ディレクトリを選択",
            mode=mode,
            terminal_prompt="出力ディレクトリ",
            required=True,
        )
    if config is None:
        config = load_config(_require_existing_file(config_path, "YAML 設定ファイル"))
    needs_interaction = required_missing or (
        zenith_deg is None and _config_zenith_missing(config)
    )
    if needs_interaction and dark_path is None and _should_ask_dark(config):
        dark_path = _select_optional_dark(mode)
    if zenith_deg is None and _config_zenith_missing(config):
        zenith_deg = _prompt_zenith_deg()

    return SelectedAnalysisInputs(
        input_path=_require_existing_file_or_directory(
            input_path,
            "science SER または capture folder",
        ),
        config_path=_require_existing_file(config_path, "YAML 設定ファイル"),
        output_dir=Path(output_dir).expanduser(),
        dark_path=_require_existing_file(dark_path, "dark SER") if dark_path else None,
        zenith_deg=zenith_deg,
    )


def _normalize_picker_mode(mode: PickerMode) -> PickerMode:
    value = mode.lower()
    if value not in {"auto", "dialog", "terminal"}:
        raise ConfigError("--picker-mode は auto, dialog, terminal のいずれかにしてください。")
    return value


def _select_file(
    *,
    title: str,
    filetypes: tuple[tuple[str, str], ...],
    mode: PickerMode,
    terminal_prompt: str,
    required: bool,
) -> Optional[Path]:
    if mode in {"auto", "dialog"}:
        selected = _select_file_dialog(title=title, filetypes=filetypes)
        if selected is not None:
            return selected
        if mode == "dialog":
            if required:
                raise ConfigError(f"{terminal_prompt} が選択されませんでした。")
            return None
    return _select_file_terminal(prompt=terminal_prompt, required=required)


def _select_file_or_directory(
    *,
    title: str,
    directory_title: str,
    filetypes: tuple[tuple[str, str], ...],
    mode: PickerMode,
    terminal_prompt: str,
    required: bool,
) -> Optional[Path]:
    if mode in {"auto", "dialog"}:
        selected = _select_file_dialog(title=title, filetypes=filetypes)
        if selected is None:
            selected = _select_directory_dialog(title=directory_title)
        if selected is not None:
            return selected
        if mode == "dialog":
            if required:
                raise ConfigError(f"{terminal_prompt} が選択されませんでした。")
            return None
    return _select_file_or_directory_terminal(prompt=terminal_prompt, required=required)


def _select_directory(
    *,
    title: str,
    mode: PickerMode,
    terminal_prompt: str,
    required: bool,
) -> Optional[Path]:
    if mode in {"auto", "dialog"}:
        selected = _select_directory_dialog(title=title)
        if selected is not None:
            return selected
        if mode == "dialog":
            if required:
                raise ConfigError(f"{terminal_prompt} が選択されませんでした。")
            return None
    return _select_directory_terminal(prompt=terminal_prompt, required=required)


def _select_optional_dark(mode: PickerMode) -> Optional[Path]:
    if mode in {"auto", "dialog"}:
        selected = _select_file_dialog(
            title="dark SER を選択。使わない場合はキャンセル",
            filetypes=(("SER files", "*.ser"), ("All files", "*.*")),
        )
        if selected is not None:
            return selected
        if mode == "dialog":
            return None
    if typer.confirm("dark SER を選択しますか？", default=False):
        return _select_file_terminal(prompt="dark SER", required=True)
    return None


def _select_file_dialog(
    *,
    title: str,
    filetypes: tuple[tuple[str, str], ...],
) -> Optional[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.update()
        selected = filedialog.askopenfilename(title=title, filetypes=filetypes)
        root.destroy()
    except Exception:
        return None
    if not selected:
        return None
    return Path(selected)


def _select_directory_dialog(*, title: str) -> Optional[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.update()
        selected = filedialog.askdirectory(title=title, mustexist=False)
        root.destroy()
    except Exception:
        return None
    if not selected:
        return None
    return Path(selected)


def _select_file_terminal(*, prompt: str, required: bool) -> Optional[Path]:
    while True:
        value = _prompt_path(f"{prompt} のパス", required=required)
        if not value and not required:
            return None
        path = Path(value).expanduser()
        if path.is_file():
            return path
        typer.echo(f"ファイルが見つかりません: {path}")


def _select_file_or_directory_terminal(*, prompt: str, required: bool) -> Optional[Path]:
    while True:
        value = _prompt_path(f"{prompt} またはフォルダのパス", required=required)
        if not value and not required:
            return None
        path = Path(value).expanduser()
        if path.is_file() or path.is_dir():
            return path
        typer.echo(f"ファイルまたはディレクトリが見つかりません: {path}")


def _select_directory_terminal(*, prompt: str, required: bool) -> Optional[Path]:
    while True:
        value = _prompt_path(f"{prompt} のパス", required=required)
        if not value and not required:
            return None
        path = Path(value).expanduser()
        if path.exists() and not path.is_dir():
            typer.echo(f"ディレクトリではありません: {path}")
            continue
        return path


def _prompt_path(prompt: str, *, required: bool) -> str:
    if required:
        return str(typer.prompt(prompt))
    return str(typer.prompt(prompt, default=""))


def _prompt_zenith_deg() -> float:
    while True:
        value = typer.prompt("zenith_deg を入力してください", type=float)
        if 0 <= value < 90:
            return float(value)
        typer.echo("zenith_deg は 0 以上 90 未満で入力してください。")


def _should_ask_dark(config: Optional[AnalysisConfig]) -> bool:
    if config is None:
        return True
    return config.calibration.dark_path is None and not config.calibration.subtract_dark


def _config_zenith_missing(config: Optional[AnalysisConfig]) -> bool:
    if config is None:
        return True
    return config.instrument.zenith_deg is None


def _require_existing_file_or_directory(path: Path, label: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise ConfigError(f"{label} が見つかりません: {resolved}")
    if not (resolved.is_file() or resolved.is_dir()):
        raise ConfigError(f"{label} はファイルまたはディレクトリである必要があります: {resolved}")
    return resolved


def _require_existing_file(path: Optional[Path], label: str) -> Path:
    if path is None:
        raise ConfigError(f"{label} が指定されていません。")
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise ConfigError(f"{label} が見つかりません: {resolved}")
    return resolved
