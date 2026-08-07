"""Convert Overture transportation GeoParquet to LabelTraj spatial NPZ tiles.

Run this script through the Overture/uv Python environment; the interactive
labeler itself only needs NumPy and Matplotlib to consume the result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import sys
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import shapely


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.offline_map_schema import (  # noqa: E402
    CLASS_CODES,
    DEFAULT_ROAD_CODE,
    OFFLINE_MAP_VERSION,
)


HEADER = struct.Struct("<QBI")
WEB_MERCATOR_RADIUS = 6378137.0


def _gcj_delta(lon, lat):
    x = lon - 105.0
    y = lat - 35.0
    pi = np.pi
    dlat = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * np.sqrt(np.abs(x))
    dlat += (20.0 * np.sin(6.0 * x * pi) + 20.0 * np.sin(2.0 * x * pi)) * 2.0 / 3.0
    dlat += (20.0 * np.sin(y * pi) + 40.0 * np.sin(y / 3.0 * pi)) * 2.0 / 3.0
    dlat += (160.0 * np.sin(y / 12.0 * pi) + 320.0 * np.sin(y * pi / 30.0)) * 2.0 / 3.0
    dlon = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * np.sqrt(np.abs(x))
    dlon += (20.0 * np.sin(6.0 * x * pi) + 20.0 * np.sin(2.0 * x * pi)) * 2.0 / 3.0
    dlon += (20.0 * np.sin(x * pi) + 40.0 * np.sin(x / 3.0 * pi)) * 2.0 / 3.0
    dlon += (150.0 * np.sin(x / 12.0 * pi) + 300.0 * np.sin(x / 30.0 * pi)) * 2.0 / 3.0
    a = 6378245.0
    ee = 0.00669342162296594323
    radlat = lat / 180.0 * pi
    magic = 1.0 - ee * np.sin(radlat) ** 2
    sqrtmagic = np.sqrt(magic)
    dlat = dlat * 180.0 / ((a * (1.0 - ee)) / (magic * sqrtmagic) * pi)
    dlon = dlon * 180.0 / (a / sqrtmagic * np.cos(radlat) * pi)
    return dlon, dlat


def wgs84_to_display(coords):
    lon = np.asarray(coords[:, 0], dtype=np.float64)
    lat = np.asarray(coords[:, 1], dtype=np.float64)
    dlon, dlat = _gcj_delta(lon, lat)
    lon = lon + dlon
    lat = np.clip(lat + dlat, -85.05112878, 85.05112878)
    x = WEB_MERCATOR_RADIUS * np.deg2rad(lon)
    y = WEB_MERCATOR_RADIUS * np.log(np.tan(np.pi / 4.0 + np.deg2rad(lat) / 2.0))
    return np.column_stack([x, y]).astype(np.float32)


def iter_line_parts(geometry):
    if geometry is None or geometry.is_empty:
        return
    if geometry.geom_type == "LineString":
        yield geometry
    elif geometry.geom_type in {"MultiLineString", "GeometryCollection"}:
        for part in geometry.geoms:
            yield from iter_line_parts(part)


def feature_hash(feature_id, part_index):
    digest = hashlib.blake2b(
        f"{feature_id}:{part_index}".encode("utf-8"), digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little", signed=False)


class TileWriters:
    def __init__(self, staging_dir, max_open=64):
        self.staging_dir = Path(staging_dir)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.max_open = max_open
        self.handles = OrderedDict()
        self.paths = set()

    def _handle(self, tile_x, tile_y):
        key = (tile_x, tile_y)
        handle = self.handles.pop(key, None)
        if handle is None:
            path = self.staging_dir / f"{tile_x}_{tile_y}.bin"
            handle = path.open("ab")
            self.paths.add(path)
        self.handles[key] = handle
        while len(self.handles) > self.max_open:
            _, old_handle = self.handles.popitem(last=False)
            old_handle.close()
        return handle

    def append(self, tile_x, tile_y, feature_id, class_code, coords):
        handle = self._handle(tile_x, tile_y)
        handle.write(HEADER.pack(feature_id, class_code, len(coords)))
        handle.write(np.asarray(coords, dtype="<f4").tobytes(order="C"))

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def convert_binary_tile(binary_path, output_path):
    coords_parts = []
    offsets = [0]
    codes = []
    ids = []
    with open(binary_path, "rb") as handle:
        while True:
            header = handle.read(HEADER.size)
            if not header:
                break
            if len(header) != HEADER.size:
                raise ValueError(f"truncated tile record header: {binary_path}")
            feature_id, class_code, point_count = HEADER.unpack(header)
            payload = handle.read(point_count * 2 * 4)
            if len(payload) != point_count * 2 * 4:
                raise ValueError(f"truncated tile coordinates: {binary_path}")
            coords = np.frombuffer(payload, dtype="<f4").reshape(point_count, 2).copy()
            coords_parts.append(coords)
            offsets.append(offsets[-1] + point_count)
            codes.append(class_code)
            ids.append(feature_id)
    all_coords = (
        np.concatenate(coords_parts, axis=0)
        if coords_parts else np.empty((0, 2), dtype=np.float32)
    )
    np.savez_compressed(
        output_path,
        coords=all_coords,
        offsets=np.asarray(offsets, dtype=np.int32),
        class_codes=np.asarray(codes, dtype=np.uint8),
        feature_ids=np.asarray(ids, dtype=np.uint64),
    )
    return len(ids), len(all_coords)


def build(input_path, output_dir, tile_size=50000.0, batch_size=10000):
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    tiles_dir = output_dir / "tiles"
    tiles_build_dir = output_dir / ".tiles_build"
    tiles_backup_dir = output_dir / ".tiles_backup"
    staging_dir = output_dir / ".staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    if tiles_build_dir.exists():
        shutil.rmtree(tiles_build_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    tiles_build_dir.mkdir(parents=True, exist_ok=True)

    writers = TileWriters(staging_dir)
    parquet_file = pq.ParquetFile(input_path)
    counts = Counter()
    source_rows = 0
    unique_parts = 0
    tile_records = 0
    columns = ["id", "subtype", "class", "geometry"]
    try:
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            data = batch.to_pydict()
            for row_index, feature_id in enumerate(data["id"]):
                source_rows += 1
                subtype = data["subtype"][row_index]
                road_class = data["class"][row_index]
                if subtype == "road":
                    class_code = CLASS_CODES.get(road_class, DEFAULT_ROAD_CODE)
                elif subtype == "rail":
                    class_code = CLASS_CODES["rail"]
                else:
                    continue
                geometry = shapely.from_wkb(data["geometry"][row_index])
                for part_index, part in enumerate(iter_line_parts(geometry)):
                    raw_coords = np.asarray(part.coords, dtype=np.float64)
                    if len(raw_coords) < 2 or not np.isfinite(raw_coords[:, :2]).all():
                        continue
                    coords = wgs84_to_display(raw_coords[:, :2])
                    part_id = feature_hash(feature_id, part_index)
                    tx_min = math.floor(float(coords[:, 0].min()) / tile_size)
                    tx_max = math.floor(float(coords[:, 0].max()) / tile_size)
                    ty_min = math.floor(float(coords[:, 1].min()) / tile_size)
                    ty_max = math.floor(float(coords[:, 1].max()) / tile_size)
                    unique_parts += 1
                    counts[str(road_class if subtype == "road" else "rail")] += 1
                    for tile_x in range(tx_min, tx_max + 1):
                        for tile_y in range(ty_min, ty_max + 1):
                            writers.append(tile_x, tile_y, part_id, class_code, coords)
                            tile_records += 1
            if source_rows and source_rows % 100000 < batch_size:
                print(
                    f"  processed {source_rows:,} rows, {unique_parts:,} line parts, "
                    f"{len(writers.paths):,} tiles",
                    flush=True,
                )
    finally:
        writers.close()

    total_records = 0
    total_points = 0
    binary_paths = sorted(staging_dir.glob("*.bin"))
    for index, binary_path in enumerate(binary_paths, start=1):
        output_path = tiles_build_dir / (binary_path.stem + ".npz")
        records, points = convert_binary_tile(binary_path, output_path)
        total_records += records
        total_points += points
        binary_path.unlink()
        if index % 25 == 0 or index == len(binary_paths):
            print(f"  packed {index:,}/{len(binary_paths):,} tiles", flush=True)
    staging_dir.rmdir()

    manifest = {
        "version": OFFLINE_MAP_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source": "Overture Maps transportation segment",
        "source_file": input_path.name,
        "attribution": "© Overture Maps Foundation, © OpenStreetMap contributors",
        "display_crs": "GCJ-02 coordinates projected as Web Mercator",
        "tile_size_m": float(tile_size),
        "tile_count": len(binary_paths),
        "source_rows": source_rows,
        "unique_line_parts": unique_parts,
        "tile_records": total_records,
        "coordinate_points": total_points,
        "class_counts": dict(sorted(counts.items())),
    }
    temporary = output_dir / "manifest.json.tmp"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    # Switch the complete tile set only after every tile and the new manifest
    # have been built. An interrupted refresh therefore leaves the old map usable.
    if tiles_backup_dir.exists():
        shutil.rmtree(tiles_backup_dir)
    if tiles_dir.exists():
        os.replace(tiles_dir, tiles_backup_dir)
    try:
        os.replace(tiles_build_dir, tiles_dir)
        os.replace(temporary, output_dir / "manifest.json")
    except Exception:
        if tiles_dir.exists():
            shutil.rmtree(tiles_dir)
        if tiles_backup_dir.exists():
            os.replace(tiles_backup_dir, tiles_dir)
        raise
    if tiles_backup_dir.exists():
        shutil.rmtree(tiles_backup_dir)
    # Windows consoles may use GBK, which cannot encode the copyright symbol.
    print(json.dumps(manifest, ensure_ascii=True, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "offline_basemap"))
    parser.add_argument("--tile-size", type=float, default=50000.0)
    parser.add_argument("--batch-size", type=int, default=10000)
    args = parser.parse_args()
    build(args.input, args.output_dir, tile_size=args.tile_size, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
