#!/usr/bin/env python3
"""Download and validate the Oxford VGG Dinosaur tracks and cameras.

The source files remain outside the repository.  This script uses only the
Python standard library, including a deliberately small MATLAB-v5 reader for
the 1x36 cell array of 3x4 double camera matrices in ``dino_Ps.mat``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import struct
import sys
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Any


SOURCE_PAGE = "https://www.robots.ox.ac.uk/~vgg/data/mview/"
SOURCES = {
    "README.txt": "https://www.robots.ox.ac.uk/~vgg/data/dino/README.txt",
    "viff.xy": "https://www.robots.ox.ac.uk/~vgg/data/dino/viff.xy",
    "dino_Ps.mat": "https://www.robots.ox.ac.uk/~vgg/data/dino/dino_Ps.mat",
}
EXPECTED_SHA256 = {
    "README.txt": "0f88ffa9d193d7cbd6e784092f02c8c9b009d367b8e191670f88f82b2c752c8b",
    "viff.xy": "a23e0044853968dcfbc899bcf80cbdfd5c04f3664ccd7ccd9a7b5701f6d53a8b",
    "dino_Ps.mat": "61adf55edf43764ab50ce389fd3e95516046cd4ed833584b6f6a2e7ea268d281",
}

MI_UINT32 = 6
MI_DOUBLE = 9
MI_MATRIX = 14
MI_COMPRESSED = 15
MX_CELL = 1
MX_DOUBLE = 6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path, offline: bool) -> None:
    if path.exists() and sha256(path) == EXPECTED_SHA256[path.name]:
        return
    if offline:
        raise RuntimeError(f"missing or checksum-mismatched offline input: {path}")
    request = urllib.request.Request(
        url, headers={"User-Agent": "triangulation-real-data-probe/1.0"}
    )
    temporary = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(request, timeout=60) as response:
        temporary.write_bytes(response.read())
    temporary.replace(path)


def read_element(data: bytes, offset: int, endian: str) -> tuple[int, bytes, int]:
    if offset + 8 > len(data):
        raise ValueError(f"truncated MATLAB element tag at byte {offset}")
    data_type, small_size = struct.unpack_from(endian + "HH", data, offset)
    if small_size:
        if small_size > 4:
            raise ValueError(f"invalid small MATLAB element size {small_size}")
        return data_type, data[offset + 4 : offset + 4 + small_size], offset + 8

    data_type, size = struct.unpack_from(endian + "II", data, offset)
    payload_start = offset + 8
    payload_end = payload_start + size
    if payload_end > len(data):
        raise ValueError(f"truncated MATLAB element payload at byte {offset}")
    next_offset = payload_start + ((size + 7) // 8) * 8
    return data_type, data[payload_start:payload_end], next_offset


def parse_matrix_payload(payload: bytes, endian: str) -> dict[str, Any]:
    offset = 0
    fields: list[tuple[int, bytes]] = []
    while offset < len(payload):
        data_type, value, next_offset = read_element(payload, offset, endian)
        fields.append((data_type, value))
        if next_offset <= offset:
            raise ValueError("MATLAB element parser made no progress")
        offset = next_offset

    if len(fields) < 3:
        raise ValueError("MATLAB matrix is missing flags, dimensions, or name")
    if fields[0][0] != MI_UINT32 or len(fields[0][1]) < 8:
        raise ValueError("unsupported MATLAB array flags")
    array_flags = struct.unpack_from(endian + "II", fields[0][1])
    matrix_class = array_flags[0] & 0xFF

    dimensions_payload = fields[1][1]
    if len(dimensions_payload) % 4:
        raise ValueError("invalid MATLAB dimension array")
    dimensions = list(
        struct.unpack(endian + "i" * (len(dimensions_payload) // 4), dimensions_payload)
    )
    name = fields[2][1].decode("latin-1")

    if matrix_class == MX_CELL:
        cells = []
        for data_type, value in fields[3:]:
            if data_type != MI_MATRIX:
                raise ValueError(f"unsupported cell payload type {data_type}")
            cells.append(parse_matrix_payload(value, endian))
        return {"class": matrix_class, "dimensions": dimensions, "name": name, "cells": cells}

    if matrix_class == MX_DOUBLE:
        if len(fields) != 4 or fields[3][0] != MI_DOUBLE:
            raise ValueError("unsupported non-real MATLAB double matrix")
        numeric_payload = fields[3][1]
        if len(numeric_payload) % 8:
            raise ValueError("invalid MATLAB double payload length")
        values = list(
            struct.unpack(endian + "d" * (len(numeric_payload) // 8), numeric_payload)
        )
        return {
            "class": matrix_class,
            "dimensions": dimensions,
            "name": name,
            "values": values,
        }

    raise ValueError(f"unsupported MATLAB matrix class {matrix_class}")


def parse_matlab_cameras(path: Path) -> list[list[list[float]]]:
    data = path.read_bytes()
    if len(data) < 136 or not data.startswith(b"MATLAB 5.0 MAT-file"):
        raise ValueError("not a MATLAB-v5 file")
    endian_marker = data[126:128]
    if endian_marker == b"IM":
        endian = "<"
    elif endian_marker == b"MI":
        endian = ">"
    else:
        raise ValueError(f"unknown MATLAB endian marker {endian_marker!r}")

    data_type, payload, _ = read_element(data, 128, endian)
    if data_type == MI_COMPRESSED:
        expanded = zlib.decompress(payload)
        data_type, payload, _ = read_element(expanded, 0, endian)
    if data_type != MI_MATRIX:
        raise ValueError(f"expected top-level MATLAB matrix, found type {data_type}")
    matrix = parse_matrix_payload(payload, endian)
    if matrix["class"] != MX_CELL or matrix["dimensions"] != [1, 36] or matrix["name"] != "P":
        raise ValueError("expected camera variable P as a 1x36 cell array")

    cameras: list[list[list[float]]] = []
    for index, cell in enumerate(matrix["cells"], start=1):
        if cell["class"] != MX_DOUBLE or cell["dimensions"] != [3, 4]:
            raise ValueError(f"camera {index} is not a real 3x4 double matrix")
        values = cell["values"]
        if len(values) != 12:
            raise ValueError(f"camera {index} has {len(values)} values, expected 12")
        # MATLAB stores numeric matrices column-major.
        cameras.append([[values[row + 3 * column] for column in range(4)] for row in range(3)])
    if len(cameras) != 36:
        raise ValueError(f"found {len(cameras)} cameras, expected 36")
    return cameras


def validate_tracks(path: Path) -> tuple[int, int]:
    rows = 0
    visible_observations = 0
    with path.open("r", encoding="ascii") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split()
            if len(fields) != 72:
                raise ValueError(f"track row {line_number} has {len(fields)} fields, expected 72")
            values = [float(field) for field in fields]
            for view in range(36):
                x, y = values[2 * view : 2 * view + 2]
                x_missing = x == -1.0
                y_missing = y == -1.0
                if x_missing != y_missing:
                    raise ValueError(f"track row {line_number}, view {view + 1} has a split sentinel")
                if not x_missing:
                    visible_observations += 1
            rows += 1
    return rows, visible_observations


def write_cameras(path: Path, cameras: list[list[list[float]]]) -> None:
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("# view p00 p01 p02 p03 p10 p11 p12 p13 p20 p21 p22 p23\n")
        for view, camera in enumerate(cameras, start=1):
            values = [camera[row][column] for row in range(3) for column in range(4)]
            stream.write(str(view) + " " + " ".join(format(value, ".17g") for value in values) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("/private/tmp/oxford_vgg_dinosaur"),
        help="temporary download/output directory (default: %(default)s)",
    )
    parser.add_argument("--offline", action="store_true", help="never access the network")
    args = parser.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)

    file_records: dict[str, dict[str, Any]] = {}
    for name, url in SOURCES.items():
        path = args.dest / name
        download(url, path, args.offline)
        actual_hash = sha256(path)
        expected_hash = EXPECTED_SHA256[name]
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"checksum mismatch for {name}: expected {expected_hash}, got {actual_hash}"
            )
        file_records[name] = {
            "url": url,
            "bytes": path.stat().st_size,
            "sha256": actual_hash,
        }

    track_rows, visible_observations = validate_tracks(args.dest / "viff.xy")
    cameras = parse_matlab_cameras(args.dest / "dino_Ps.mat")
    camera_text = args.dest / "dino_cameras.tsv"
    write_cameras(camera_text, cameras)

    readme_text = (args.dest / "README.txt").read_text(encoding="ascii")
    declared_match = re.search(r"There are\s+(\d+)\s+tracks", readme_text)
    declared_tracks = int(declared_match.group(1)) if declared_match else None
    manifest = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_page": SOURCE_PAGE,
        "source_files": file_records,
        "tracks": {
            "rows_observed": track_rows,
            "columns_per_row": 72,
            "views": 36,
            "visible_observations": visible_observations,
            "readme_declared_tracks": declared_tracks,
            "readme_count_matches_file": declared_tracks == track_rows,
        },
        "cameras": {
            "variable": "P",
            "count": len(cameras),
            "shape_each": [3, 4],
            "text_export": camera_text.name,
            "text_export_sha256": sha256(camera_text),
        },
        "note": (
            "The official README declares 4838 tracks, but the checksummed viff.xy "
            "contains 4983 well-formed rows. The probe preserves and reports this discrepancy."
        ),
    }
    manifest_path = args.dest / "provenance.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"validated {track_rows} tracks, {visible_observations} visible observations, {len(cameras)} cameras")
    if declared_tracks != track_rows:
        print(
            f"warning: README declares {declared_tracks} tracks but viff.xy has {track_rows} rows",
            file=sys.stderr,
        )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, urllib.error.URLError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
