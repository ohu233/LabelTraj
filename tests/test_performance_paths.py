import unittest

import numpy as np
import pandas as pd

import LabelPath
from utils import geo_utils


class PerformancePathTest(unittest.TestCase):
    def test_vectorized_segment_builder_preserves_interval_semantics(self):
        frame = pd.DataFrame({
            "uid": [1, 1, 1, 1, 2, 2, 2],
            "idx": [0, 1, 2, 3, 0, 1, 2],
            "stime": [10, 11, 12, 13, 20, 21, 22],
            "lat": [30.0] * 7,
            "lon": [120.0] * 7,
            "hex_x": [0, 1, 2, 3, 10, 11, 12],
            "hex_y": [0, -1, -2, -3, -10, -11, -12],
            "hex_z": [0, 0, 0, 0, 0, 0, 0],
            "time_value": [0, 2, 3, 4, 0, 5, 6],
            "dist_value": [0, 10, 15, 20, 0, 25, 30],
            "velocity": [0, 18, 18, 18, 0, 18, 18],
            "attribution": ["origin", "ALL", "ALL", "ALL", "origin", "ALL", "ALL"],
        })

        every_point = LabelPath._build_segments_from_point_df(frame, 1)
        self.assertEqual(len(every_point), 5)
        self.assertEqual(every_point["idx_o"].tolist(), [0, 1, 2, 0, 1])
        self.assertEqual(every_point["idx_d"].tolist(), [1, 2, 3, 1, 2])
        self.assertEqual(every_point["time_d"].tolist(), [2, 3, 4, 5, 6])

        sampled = LabelPath._build_segments_from_point_df(frame, 2)
        self.assertEqual(len(sampled), 2)
        self.assertEqual(sampled["idx_o"].tolist(), [0, 0])
        self.assertEqual(sampled["idx_d"].tolist(), [2, 2])
        self.assertEqual(sampled["time_d"].tolist(), [5, 11])
        self.assertEqual(sampled["dist_d"].tolist(), [25, 55])
        np.testing.assert_allclose(sampled["velocity_d"], [18.0, 18.0])

    def test_vectorized_hex_lookup_matches_scalar_lookup(self):
        geo_utils.get_hex_grid()
        valid = np.argwhere(~np.isnan(geo_utils._HEX_LON))[:8]
        xs = valid[:, 0] + geo_utils._HEX_X_OFFSET
        zs = valid[:, 1] + geo_utils._HEX_Z_OFFSET
        ys = -xs - zs
        batch_lon, batch_lat = geo_utils.hex_to_wgs84(xs, ys, zs)
        scalar = [geo_utils.hex_to_wgs84(int(x), int(y), int(z)) for x, y, z in zip(xs, ys, zs)]
        np.testing.assert_allclose(batch_lon, [item[0] for item in scalar])
        np.testing.assert_allclose(batch_lat, [item[1] for item in scalar])


if __name__ == "__main__":
    unittest.main()
