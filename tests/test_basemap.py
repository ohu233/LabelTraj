import os
import json
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from utils import basemap
from utils.offline_basemap import OfflineBasemap


class BasemapTest(unittest.TestCase):
    def test_online_viewport_disables_tile_cache_and_preserves_extent(self):
        figure, ax = plt.subplots()
        ax.axis((100.0, 200.0, 300.0, 400.0))
        fake_image = np.zeros((16, 16, 4), dtype=np.uint8)
        fake_extent = (90.0, 210.0, 290.0, 410.0)

        fake_contextily = SimpleNamespace()
        fake_contextily.bounds2img = unittest.mock.Mock(
            return_value=(fake_image, fake_extent),
        )
        fake_contextily.add_attribution = unittest.mock.Mock()
        with patch.object(basemap, "_get_contextily", return_value=fake_contextily), patch.dict(
            os.environ, {"LABELTRAJ_BASEMAP_MODE": "online"}, clear=False,
        ):
            loaded = basemap.add_basemap(ax, alpha=0.8, zoom=11)

        self.assertTrue(loaded)
        self.assertEqual(tuple(ax.axis()), (100.0, 200.0, 300.0, 400.0))
        self.assertFalse(fake_contextily.bounds2img.call_args.kwargs["use_cache"])
        self.assertEqual(fake_contextily.bounds2img.call_args.kwargs["n_connections"], 1)
        self.assertEqual(fake_contextily.bounds2img.call_args.kwargs["zoom"], 11)
        plt.close(figure)

    def test_offline_tile_draws_without_contextily(self):
        temp_dir = Path(__file__).resolve().parent / "_tmp_offline_basemap"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        try:
            tiles_dir = os.path.join(str(temp_dir), "tiles")
            os.makedirs(tiles_dir)
            manifest = {
                "version": 1,
                "tile_size_m": 50000.0,
                "attribution": "test attribution",
            }
            with open(os.path.join(str(temp_dir), "manifest.json"), "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            np.savez_compressed(
                os.path.join(tiles_dir, "0_0.npz"),
                coords=np.asarray([[1000, 1000], [2000, 2000]], dtype=np.float32),
                offsets=np.asarray([0, 2], dtype=np.int32),
                class_codes=np.asarray([1], dtype=np.uint8),
                feature_ids=np.asarray([123], dtype=np.uint64),
            )
            figure, ax = plt.subplots()
            ax.axis((0, 10000, 0, 10000))
            loaded = OfflineBasemap(str(temp_dir)).draw(ax)
            self.assertTrue(loaded)
            self.assertEqual(len(ax.collections), 1)
            self.assertEqual(tuple(ax.axis()), (0.0, 10000.0, 0.0, 10000.0))
            plt.close(figure)
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
