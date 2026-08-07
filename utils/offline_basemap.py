"""Runtime reader and Matplotlib renderer for the local vector basemap."""

from __future__ import annotations

import json
import math
import os
from collections import OrderedDict, defaultdict

import numpy as np
from matplotlib.collections import LineCollection

from utils.offline_map_schema import CLASS_STYLES, OFFLINE_MAP_VERSION


DEFAULT_OFFLINE_MAP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "offline_basemap",
)


class OfflineBasemap:
    """Lazy spatial-tile reader; only tiles touching the current view are read."""

    def __init__(self, root=DEFAULT_OFFLINE_MAP_DIR, max_cached_tiles=24):
        self.root = os.path.abspath(root)
        manifest_path = os.path.join(self.root, "manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        version = int(self.manifest.get("version", -1))
        if version != OFFLINE_MAP_VERSION:
            raise ValueError(
                f"offline basemap version {version} is unsupported; "
                f"expected {OFFLINE_MAP_VERSION}"
            )
        self.tile_size = float(self.manifest["tile_size_m"])
        self.tiles_dir = os.path.join(self.root, "tiles")
        self.max_cached_tiles = int(max_cached_tiles)
        self._cache = OrderedDict()

    def _tile_path(self, tile_x, tile_y):
        return os.path.join(self.tiles_dir, f"{tile_x}_{tile_y}.npz")

    def _load_tile(self, tile_x, tile_y):
        key = (int(tile_x), int(tile_y))
        cached = self._cache.pop(key, None)
        if cached is not None:
            self._cache[key] = cached
            return cached
        path = self._tile_path(*key)
        if not os.path.exists(path):
            return None
        with np.load(path) as data:
            tile = {
                "coords": data["coords"],
                "offsets": data["offsets"],
                "class_codes": data["class_codes"],
                "feature_ids": data["feature_ids"],
            }
        self._cache[key] = tile
        while len(self._cache) > self.max_cached_tiles:
            self._cache.popitem(last=False)
        return tile

    def iter_tiles_for_bounds(self, xmin, xmax, ymin, ymax):
        tx_min = math.floor(float(xmin) / self.tile_size)
        tx_max = math.floor(float(xmax) / self.tile_size)
        ty_min = math.floor(float(ymin) / self.tile_size)
        ty_max = math.floor(float(ymax) / self.tile_size)
        for tile_x in range(tx_min, tx_max + 1):
            for tile_y in range(ty_min, ty_max + 1):
                yield tile_x, tile_y

    def draw(self, ax, alpha=1.0):
        xmin, xmax, ymin, ymax = ax.axis()
        grouped = defaultdict(list)
        seen = set()
        loaded_tiles = 0
        for tile_x, tile_y in self.iter_tiles_for_bounds(xmin, xmax, ymin, ymax):
            tile = self._load_tile(tile_x, tile_y)
            if tile is None:
                continue
            loaded_tiles += 1
            offsets = tile["offsets"]
            for index, feature_id in enumerate(tile["feature_ids"]):
                feature_id = int(feature_id)
                if feature_id in seen:
                    continue
                seen.add(feature_id)
                start, end = int(offsets[index]), int(offsets[index + 1])
                coords = tile["coords"][start:end]
                if len(coords) >= 2:
                    grouped[int(tile["class_codes"][index])].append(coords)

        if loaded_tiles == 0:
            return False

        ax.set_facecolor("#f3f1ea")
        for code, style in sorted(
            CLASS_STYLES.items(), key=lambda item: item[1]["zorder"],
        ):
            segments = grouped.get(code)
            if not segments:
                continue
            collection = LineCollection(
                segments,
                colors=style["color"],
                linewidths=style["linewidth"],
                linestyles=style["linestyle"],
                alpha=alpha,
                zorder=style["zorder"],
                antialiased=True,
            )
            ax.add_collection(collection)

        ax.axis((xmin, xmax, ymin, ymax))
        ax.text(
            0.995,
            0.005,
            self.manifest.get(
                "attribution", "© Overture Maps Foundation, © OpenStreetMap contributors",
            ),
            transform=ax.transAxes,
            horizontalalignment="right",
            verticalalignment="bottom",
            fontsize=5.5,
            color="#555555",
            zorder=0.9,
            bbox=dict(
                boxstyle="round,pad=0.1", facecolor="white", alpha=0.72,
                edgecolor="none",
            ),
        )
        return True


def offline_basemap_available(root=DEFAULT_OFFLINE_MAP_DIR):
    return os.path.isfile(os.path.join(root, "manifest.json"))

