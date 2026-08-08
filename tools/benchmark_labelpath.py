"""Repeatable real-data benchmark for LabelPath startup and OD switching."""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
import statistics
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("LABELTRAJ_BASEMAP_MODE", "offline")

import matplotlib.pyplot as plt

import LabelPath
from utils.osm_pois import load_osm_pois
from utils.tools import hex_mapdata_to_road_sets


def timed(label, function):
    started = time.perf_counter()
    value = function()
    elapsed = time.perf_counter() - started
    print(f"{label}: {elapsed:.4f}s", flush=True)
    return value, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--switches", type=int, default=8)
    parser.add_argument("--profile-switch", type=int)
    parser.add_argument("--snapshot", type=str)
    args = parser.parse_args()

    grid, _ = timed("hex_load", lambda: LabelPath.load_hex_mapdata(LabelPath.HEX_PKL_PATH))
    road_sets, _ = timed("road_sets", lambda: hex_mapdata_to_road_sets(grid))
    trajectories, _ = timed(
        "trajectory_load",
        lambda: LabelPath.load_traj_csv_hex(LabelPath.DEFAULT_CSV_PATH, sample_step=1),
    )
    pois, _ = timed(
        "poi_load",
        lambda: load_osm_pois(LabelPath.DEFAULT_POI_PATH, hex_grid=grid),
    )
    labeled, _ = timed(
        "label_load", lambda: LabelPath.load_labeled_modes(LabelPath.DEFAULT_OUTPUT_DIR),
    )
    multi, _ = timed("road_union", lambda: set().union(*road_sets.values()))

    def make_state(index):
        return LabelPath.LabelState(trajectories.iloc[index], multi, hex_grid=grid)

    renderer, _ = timed(
        "first_render",
        lambda: LabelPath.PathRenderer(
            make_state(0), grid, road_sets=road_sets, traj_df=trajectories,
            current_idx=0, output_dir=LabelPath.DEFAULT_OUTPUT_DIR,
            pois=pois, labeled_modes=labeled,
        ),
    )
    timings = []
    candidates = [1, 2, 3, 10, 100, 1_000, 10_000, 50_000, 100_000]
    for index in candidates[: max(1, args.switches)]:
        started = time.perf_counter()
        renderer.show_segment(make_state(index), index)
        elapsed = time.perf_counter() - started
        timings.append(elapsed)
        print(f"switch[{index}]: {elapsed:.4f}s", flush=True)

    print(
        f"switch_summary: median={statistics.median(timings):.4f}s "
        f"mean={statistics.mean(timings):.4f}s max={max(timings):.4f}s",
        flush=True,
    )
    if args.profile_switch is not None:
        index = min(max(0, args.profile_switch), len(trajectories) - 1)
        profiler = cProfile.Profile()
        profiler.enable()
        renderer.show_segment(make_state(index), index)
        profiler.disable()
        pstats.Stats(profiler).sort_stats("cumulative").print_stats(35)
    if args.snapshot:
        renderer.fig.savefig(args.snapshot, dpi=120)
        print(f"snapshot: {os.path.abspath(args.snapshot)}", flush=True)
    plt.close(renderer.fig)


if __name__ == "__main__":
    main()
