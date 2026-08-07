"""Download and build the full offline basemap for the LabelTraj hex extent."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "offline_basemap"
SOURCE_PATH = DATA_DIR / "overture_segments.parquet"
UV_CACHE_DIR = PROJECT_ROOT / "data" / ".uv_cache"


def hex_bounds(cache_path=PROJECT_ROOT / "data" / "hex_cache.npz"):
    with np.load(cache_path) as data:
        return (
            float(np.nanmin(data["lon"])),
            float(np.nanmin(data["lat"])),
            float(np.nanmax(data["lon"])),
            float(np.nanmax(data["lat"])),
        )


def run(command, env):
    print("Running:", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def parquet_is_complete(path):
    """A completed Parquet file has PAR1 magic at both ends."""
    path = Path(path)
    if not path.exists() or path.stat().st_size < 12:
        return False
    with path.open("rb") as handle:
        beginning = handle.read(4)
        handle.seek(-4, os.SEEK_END)
        ending = handle.read(4)
    return beginning == b"PAR1" and ending == b"PAR1"


def main():
    parser = argparse.ArgumentParser(description="Build the complete offline vector basemap")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--remove-source", action="store_true")
    parser.add_argument("--tile-size", type=float, default=50000.0)
    args = parser.parse_args()

    uvx = shutil.which("uvx")
    uv = shutil.which("uv")
    if not uvx or not uv:
        raise RuntimeError("uv/uvx is required; install uv before building the offline map")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str(UV_CACHE_DIR)
    west, south, east, north = hex_bounds()
    bbox = f"{west:.7f},{south:.7f},{east:.7f},{north:.7f}"

    if args.force_download or not parquet_is_complete(SOURCE_PATH):
        run([
            uvx, "--from", "overturemaps==1.0.1", "overturemaps", "download",
            f"--bbox={bbox}", "-f", "geoparquet", "-t", "segment",
            "-o", str(SOURCE_PATH), "--connect_timeout", "30",
            "--request_timeout", "180",
        ], env)
    else:
        print(f"Using existing source: {SOURCE_PATH}")

    run([
        uv, "run", "--python", "3.10", "--with", "overturemaps==1.0.1",
        "python", str(PROJECT_ROOT / "tools" / "build_offline_basemap.py"),
        "--input", str(SOURCE_PATH), "--output-dir", str(DATA_DIR),
        "--tile-size", str(args.tile_size),
    ], env)

    if args.remove_source:
        SOURCE_PATH.unlink(missing_ok=True)
        state_path = SOURCE_PATH.with_suffix(SOURCE_PATH.suffix + ".state")
        state_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
