"""SharpCap mono SER ファイル用の最小 SER reader。"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np

from .exceptions import SERFormatError
from .models import SERMetadata

HEADER_SIZE = 178
HEADER_STRUCT = struct.Struct("<14s7I40s40s40sQQ")
MONO_COLOR_IDS = {0: "mono"}
COLOR_ID_NAMES = {
    0: "mono",
    8: "bayer_rggb",
    9: "bayer_grbg",
    10: "bayer_gbrg",
    11: "bayer_bggr",
    16: "bayer_cyym",
    17: "bayer_ycym",
    18: "bayer_yymc",
    19: "bayer_mcyy",
    100: "rgb",
    101: "bgr",
}


def _clean_text(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("utf-8", errors="ignore").strip()


class SERReader:
    """Read mono8/mono16 SER frames without loading the full video."""

    def __init__(self, path: Path, *, reject_color: bool = True) -> None:
        self.path = Path(path)
        self.reject_color = reject_color
        self._handle = None
        self.metadata = self._read_metadata()
        self._timestamps: Optional[List[int]] = None

    def __enter__(self) -> "SERReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close()

    @property
    def bytes_per_pixel(self) -> int:
        if self.metadata.pixel_depth <= 8:
            return 1
        if self.metadata.pixel_depth <= 16:
            return 2
        raise SERFormatError(f"未対応の SER pixel depth です: {self.metadata.pixel_depth}")

    @property
    def frame_bytes(self) -> int:
        return self.metadata.width * self.metadata.height * self.bytes_per_pixel

    @property
    def frames_offset(self) -> int:
        return HEADER_SIZE

    def open(self) -> None:
        if self._handle is None or self._handle.closed:
            self._handle = self.path.open("rb")

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()

    def _read_metadata(self) -> SERMetadata:
        if not self.path.exists():
            raise SERFormatError(f"SER ファイルが存在しません: {self.path}")
        with self.path.open("rb") as handle:
            raw = handle.read(HEADER_SIZE)
        if len(raw) != HEADER_SIZE:
            raise SERFormatError(f"SER header が短すぎます: {self.path}")

        (
            file_id,
            _lu_id,
            color_id,
            little_endian,
            width,
            height,
            pixel_depth,
            frame_count,
            observer,
            instrument,
            telescope,
            date_time_local_raw,
            date_time_utc_raw,
        ) = HEADER_STRUCT.unpack(raw)

        if not file_id.startswith(b"LUCAM-RECORDER"):
            raise SERFormatError(
                f"未対応の SER signature です: {file_id!r}。LUCAM-RECORDER を期待しました。"
            )
        if width <= 0 or height <= 0 or frame_count < 0:
            raise SERFormatError("SER の画像サイズまたは frame count が不正です。")
        if pixel_depth not in (8, 16):
            raise SERFormatError(
                f"mono8/mono16 SER のみ対応しています。取得した bit depth: {pixel_depth}"
            )

        color_mode = COLOR_ID_NAMES.get(color_id, f"color_id_{color_id}")
        if self.reject_color and color_id not in MONO_COLOR_IDS:
            raise SERFormatError(
                f"MVP では Color/Bayer SER は未対応です。取得した color mode: {color_mode}"
            )

        # SER の LittleEndian field は SharpCap/APollo-M の実ファイルでは 0 が
        # little-endian を示す。bool(0) にすると逆解釈になるため明示的に扱う。
        is_little_endian = little_endian == 0

        metadata = SERMetadata(
            path=self.path,
            width=width,
            height=height,
            frame_count=frame_count,
            pixel_depth=pixel_depth,
            color_id=color_id,
            color_mode="mono16" if pixel_depth == 16 else "mono8",
            little_endian=is_little_endian,
            observer=_clean_text(observer),
            instrument=_clean_text(instrument),
            telescope=_clean_text(telescope),
            date_time_local_raw=date_time_local_raw,
            date_time_utc_raw=date_time_utc_raw,
        )
        self._populate_timestamp_metadata(metadata)
        return metadata

    def _populate_timestamp_metadata(self, metadata: SERMetadata) -> None:
        frame_bytes = metadata.width * metadata.height * (1 if metadata.pixel_depth <= 8 else 2)
        timestamp_offset = HEADER_SIZE + frame_bytes * metadata.frame_count
        size = os.path.getsize(metadata.path)
        available = size - timestamp_offset
        if metadata.frame_count > 0 and available >= metadata.frame_count * 8:
            metadata.timestamps_available = True
            metadata.timestamp_count = metadata.frame_count

    def _dtype(self) -> np.dtype:
        if self.metadata.pixel_depth <= 8:
            return np.dtype("uint8")
        endian = "<" if self.metadata.little_endian else ">"
        return np.dtype(f"{endian}u2")

    def read_timestamps(self) -> Optional[List[int]]:
        if not self.metadata.timestamps_available:
            return None
        if self._timestamps is not None:
            return self._timestamps
        offset = self.frames_offset + self.frame_bytes * self.metadata.frame_count
        with self.path.open("rb") as handle:
            handle.seek(offset)
            raw = handle.read(self.metadata.frame_count * 8)
        if len(raw) < self.metadata.frame_count * 8:
            return None
        values = list(struct.unpack(f"<{self.metadata.frame_count}Q", raw))
        if not _timestamps_are_valid(values):
            return None
        self._timestamps = values
        if len(values) > 1:
            elapsed = (values[-1] - values[0]) / 10_000_000.0
            if elapsed > 0:
                self.metadata.estimated_fps = (len(values) - 1) / elapsed
        return values

    def frame_time_seconds(self, frame_index: int) -> Optional[float]:
        timestamps = self.read_timestamps()
        if timestamps is None or frame_index >= len(timestamps):
            return None
        return (timestamps[frame_index] - timestamps[0]) / 10_000_000.0

    def iter_frames(
        self,
        *,
        start: int = 0,
        end: Optional[int] = None,
        max_frames: Optional[int] = None,
    ) -> Iterator[Tuple[int, np.ndarray]]:
        self.open()
        if self._handle is None:
            raise SERFormatError("SER reader が open されていません。")
        frame_count = self.metadata.frame_count
        stop = frame_count if end is None else min(end, frame_count)
        start = max(0, start)
        if start >= stop:
            return
        if max_frames is not None:
            stop = min(stop, start + max_frames)

        dtype = self._dtype()
        for frame_index in range(start, stop):
            offset = self.frames_offset + frame_index * self.frame_bytes
            self._handle.seek(offset)
            raw = self._handle.read(self.frame_bytes)
            if len(raw) != self.frame_bytes:
                raise SERFormatError(f"frame {frame_index} の読み込み中に予期しない EOF です。")
            frame = np.frombuffer(raw, dtype=dtype).reshape(
                (self.metadata.height, self.metadata.width)
            )
            yield frame_index, frame.copy()


def _timestamps_are_valid(values: List[int]) -> bool:
    if len(values) < 2:
        return False
    if values[0] == 0:
        return False
    diffs = np.diff(np.asarray(values, dtype=np.float64))
    if not np.all(np.isfinite(diffs)):
        return False
    return bool(np.all(diffs > 0))
