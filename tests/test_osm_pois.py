import unittest

from utils.osm_pois import (
    CATEGORY_SUBWAY,
    CATEGORY_TOLL,
    CATEGORY_TRAIN,
    build_overpass_query,
    classify_tags,
    element_lon_lat,
    iter_query_tiles,
    preferred_name,
)


class OSMPOITest(unittest.TestCase):
    def test_classifies_subway_train_and_toll_tags(self):
        self.assertEqual(
            classify_tags({"railway": "station", "station": "subway"}),
            [CATEGORY_SUBWAY],
        )
        self.assertEqual(
            classify_tags({"railway": "station", "train": "yes", "subway": "yes"}),
            [CATEGORY_SUBWAY, CATEGORY_TRAIN],
        )
        self.assertEqual(
            classify_tags({"railway": "halt"}),
            [CATEGORY_TRAIN],
        )
        self.assertEqual(classify_tags({"barrier": "toll_booth"}), [CATEGORY_TOLL])
        self.assertEqual(classify_tags({"highway": "toll_gantry"}), [CATEGORY_TOLL])

    def test_excludes_other_urban_rail_station_types(self):
        self.assertEqual(
            classify_tags({"railway": "station", "station": "light_rail"}),
            [],
        )
        self.assertEqual(
            classify_tags({"railway": "station", "station": "monorail"}),
            [],
        )
        self.assertEqual(
            classify_tags({
                "railway": "construction", "station": "subway",
                "public_transport": "station", "subway": "yes",
            }),
            [],
        )
        self.assertEqual(
            classify_tags({
                "railway": "station", "station": "subway", "subway": "yes",
                "construction": "yes",
            }),
            [],
        )
        self.assertEqual(
            classify_tags({
                "railway": "station", "station": "subway", "subway": "yes",
                "name": "测试站（在建）",
            }),
            [],
        )

    def test_reads_node_and_center_coordinates(self):
        self.assertEqual(element_lon_lat({"lon": 120, "lat": 31}), (120.0, 31.0))
        self.assertEqual(
            element_lon_lat({"center": {"lon": 121, "lat": 32}}),
            (121.0, 32.0),
        )
        self.assertIsNone(element_lon_lat({}))

    def test_prefers_chinese_name(self):
        self.assertEqual(preferred_name({"name": "Shanghai", "name:zh": "上海站"}), "上海站")
        self.assertEqual(preferred_name({}), "未命名")

    def test_tiles_cover_bbox_without_overshoot(self):
        tiles = list(iter_query_tiles((27, 114, 35, 123), tile_degrees=4.5))
        self.assertEqual(len(tiles), 4)
        self.assertEqual(tiles[0], (27.0, 114.0, 31.5, 118.5))
        self.assertEqual(tiles[-1], (31.5, 118.5, 35.0, 123.0))

    def test_query_contains_all_osm_tag_families(self):
        query = build_overpass_query((27, 114, 35, 123))
        self.assertIn('"railway"~"^(station|halt)$"', query)
        self.assertIn('"barrier"="toll_booth"', query)
        self.assertIn('"highway"="toll_gantry"', query)
        self.assertIn("out center tags", query)


if __name__ == "__main__":
    unittest.main()
