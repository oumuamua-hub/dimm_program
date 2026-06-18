import struct

import numpy as np

from dimm_analyzer.ser_reader import SERReader


def test_mono16_little_endian_flag_zero_reads_little_endian(tmp_path):
    path = tmp_path / "little.ser"
    width = 2
    height = 2
    frame_count = 1
    header = struct.pack(
        "<14s7I40s40s40sQQ",
        b"LUCAM-RECORDER",
        0,
        0,
        0,
        width,
        height,
        16,
        frame_count,
        b"Observer",
        b"Camera",
        b"Scope",
        0,
        0,
    )
    values = np.array([[1, 256], [4096, 32704]], dtype="<u2")
    path.write_bytes(header + values.tobytes())

    reader = SERReader(path)
    _, frame = next(reader.iter_frames())

    assert reader.metadata.little_endian is True
    assert frame.tolist() == [[1, 256], [4096, 32704]]
