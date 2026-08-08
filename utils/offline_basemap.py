"""Runtime reader and Matplotlib renderer for the local vector basemap."""

from __future__ import annotations

import json
import math
import os
from collections import OrderedDict, defaultdict

import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.path import Path as MatplotlibPath

from utils.offline_map_schema import CLASS_STYLES, OFFLINE_MAP_VERSION


DEFAULT_OFFLINE_MAP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "offline_basemap",
)


class OfflineBasemap:
    """Lazy spatial-tile reader; only tiles touching the current view are read."""

    def __init__(self, root=DEFAULT_OFFLINE_MAP_DIR, max_cached_tiles=24,
                 max_cached_scenes=6):
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
        self.max_cached_scenes = int(max_cached_scenes)
        self._scene_cache = OrderedDict()

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

    def _scene_for_tiles(self, tile_keys):
        """Build one multi-subpath object per road class and cache the scene.

        A LineCollection creates a separate Matplotlib Path for every road
        feature. Dense tiles can contain tens of thousands of features, which
        dominated OD switching. One coded Path per class retains disconnected
        lines while cutting artist construction to roughly 15 objects.
        """
        scene_key = tuple(tile_keys)
        cached = self._scene_cache.pop(scene_key, None)
        if cached is not None:
            self._scene_cache[scene_key] = cached
            return cached

        grouped = defaultdict(list)
        seen = set()
        loaded_tiles = 0
        for tile_x, tile_y in scene_key:
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
                if end - start >= 2:
                    grouped[int(tile["class_codes"][index])].append(
                        tile["coords"][start:end],
                    )

        if loaded_tiles == 0:
            scene = None
        else:
            paths = {}
            for code, segments in grouped.items():
                lengths = np.fromiter((len(segment) for segment in segments), dtype=np.int64)
                vertices = np.concatenate(segments, axis=0)
                codes = np.full(len(vertices), MatplotlibPath.LINETO, dtype=np.uint8)
                starts = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(lengths[:-1])))
                codes[starts] = MatplotlibPath.MOVETO
                paths[code] = MatplotlibPath(vertices, codes)
            scene = paths

        self._scene_cache[scene_key] = scene
        while len(self._scene_cache) > self.max_cached_scenes:
            self._scene_cache.popitem(last=False)
        return scene

    def draw(self, ax, alpha=1.0):
        xmin, xmax, ymin, ymax = ax.axis()
        tile_keys = tuple(self.iter_tiles_for_bounds(xmin, xmax, ymin, ymax))
        paths = self._scene_for_tiles(tile_keys)
        if paths is None:
            return False

        ax.set_facecolor("#f3f1ea")
        for code, style in sorted(
            CLASS_STYLES.items(), key=lambda item: item[1]["zorder"],
        ):
            path = paths.get(code)
            if path is None:
                continue
            collection = PathCollection(
                [path],
                facecolors="none",
                edgecolors=style["color"],
                linewidths=style["linewidth"],
                linestyles=style["linestyle"],
                alpha=alpha,
                zorder=style["zorder"],
                antialiaseds=True,
                transform=ax.transData,
            )
            ax.add_collection(collection, autolim=False)

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
