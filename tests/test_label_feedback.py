import unittest
import json
from unittest import mock

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import LabelPath


class LabelFeedbackTest(unittest.TestCase):
    def test_loads_saved_modes_by_uid_and_segment(self):
        labeled = pd.DataFrame([
            {"uid": 11, "idx_o": 3, "mode": "TG"},
            {"uid": 11, "idx_o": 4, "mode": "dt"},
            {"uid": 11, "idx_o": 5, "mode": "not-a-label"},
            {"uid": 11, "idx_o": 3, "mode": "GG"},
        ])
        with mock.patch.object(LabelPath.os.path, "exists", return_value=True), \
             mock.patch.object(LabelPath.pd, "read_csv", return_value=labeled):
            result = LabelPath.load_labeled_modes("unused")

        self.assertEqual(result, {(11, 3): "GG", (11, 4): "DT"})

    def test_missing_label_file_returns_empty_feedback(self):
        with mock.patch.object(LabelPath.os.path, "exists", return_value=False):
            self.assertEqual(LabelPath.load_labeled_modes("unused"), {})

    def test_loads_only_valid_saved_paths(self):
        labeled = pd.DataFrame([
            {"uid": 11, "segment_id": 1, "mode": "TG", "idx_o": 3,
             "idx_d": 6, "traj": "[[1, -1, 0], [2, -2, 0]]"},
            {"uid": 11, "segment_id": 2, "mode": "TS", "idx_o": 6,
             "idx_d": 8, "traj": ""},
            {"uid": 11, "segment_id": 3, "mode": "GG", "idx_o": 8,
             "idx_d": 9, "traj": "not-json"},
        ])
        with mock.patch.object(LabelPath.os.path, "exists", return_value=True), \
             mock.patch.object(LabelPath.pd, "read_csv", return_value=labeled):
            paths, modes = LabelPath.load_labeled_path_data("unused")

        self.assertEqual(
            paths, {(11, 1): [(1, -1, 0), (2, -2, 0)]},
        )
        self.assertEqual(modes, {(11, 1): "TG"})

    def test_loads_saved_path_window_metadata(self):
        labeled = pd.DataFrame([
            {"uid": 11, "segment_id": 1, "anchor_idx_o": 25,
             "steps": 50, "match": 1.0},
            {"uid": 11, "segment_id": 2, "anchor_idx_o": 53,
             "steps": 83, "match": 0.95},
        ])
        with mock.patch.object(LabelPath.os.path, "exists", return_value=True), \
             mock.patch.object(LabelPath.pd, "read_csv", return_value=labeled):
            result = LabelPath.load_labeled_path_metadata("unused")

        self.assertEqual(result[(11, 1)]["anchor_idx_o"], 25)
        self.assertEqual(result[(11, 2)]["steps"], 83)
        self.assertAlmostEqual(result[(11, 2)]["match"], 0.95)

    def test_user_selected_boundaries_and_road_modes_are_stored_per_segment(self):
        output_dir = LabelPath.os.path.join(
            LabelPath.os.getcwd(), "tests", "_runtime_path_output",
        )
        path_csv = LabelPath.os.path.join(output_dir, LabelPath.PATH_LABEL_FILENAME)
        try:
            _, first_key = LabelPath.write_truth_path_segment(
                output_dir, 7, "TG", 3,
                [(0, 0, 0), (1, -1, 0)], 1.0, 1,
            )
            _, second_key = LabelPath.write_truth_path_segment(
                output_dir, 7, "TG", 4,
                [(1, -1, 0), (2, -2, 0)], 1.0, 1,
            )
            _, third_key = LabelPath.write_truth_path_segment(
                output_dir, 7, "GG", 5,
                [(2, -2, 0), (3, -3, 0)], 1.0, 1,
            )

            saved = pd.read_csv(path_csv)
            self.assertEqual(first_key, (7, 1))
            self.assertEqual(second_key, (7, 2))
            self.assertEqual(third_key, (7, 3))
            self.assertEqual(saved["segment_id"].tolist(), [1, 2, 3])
            self.assertEqual(saved["mode"].tolist(), ["TG", "TG", "GG"])
        finally:
            for suffix in ("", ".tmp"):
                target = path_csv + suffix
                if LabelPath.os.path.exists(target):
                    LabelPath.os.remove(target)

    def test_saved_path_can_be_replaced_then_deleted_with_uid_renumbering(self):
        output_dir = LabelPath.os.path.join(
            LabelPath.os.getcwd(), "tests", "_runtime_path_output",
        )
        path_csv = LabelPath.os.path.join(output_dir, LabelPath.PATH_LABEL_FILENAME)
        try:
            _, first_key = LabelPath.write_truth_path_segment(
                output_dir, 7, "TG", 3,
                [(0, 0, 0), (1, -1, 0)], 1.0, 1,
            )
            _, second_key = LabelPath.write_truth_path_segment(
                output_dir, 7, "GG", 4,
                [(2, -2, 0), (3, -3, 0)], 1.0, 1,
            )
            LabelPath.write_truth_path_segment(
                output_dir, 8, "TS", 1,
                [(9, -9, 0), (10, -10, 0)], 1.0, 1,
            )

            _, replaced_key = LabelPath.write_truth_path_segment(
                output_dir, 7, "DT", 8,
                [(4, -4, 0), (5, -5, 0), (6, -6, 0)], 0.9, 2,
                segment_key=first_key,
            )
            saved = pd.read_csv(path_csv)
            saved = saved.loc[saved["uid"] == 7].sort_values("segment_id")
            self.assertEqual(replaced_key, first_key)
            self.assertEqual(saved["segment_id"].tolist(), [1, 2])
            self.assertEqual(saved["mode"].tolist(), ["DT", "GG"])
            self.assertEqual(int(saved.iloc[0]["anchor_idx_o"]), 8)
            self.assertEqual(
                json.loads(saved.iloc[0]["traj"]),
                [[4, -4, 0], [5, -5, 0], [6, -6, 0]],
            )

            _, deleted_count = LabelPath.delete_truth_path_segment(
                output_dir, first_key,
            )
            remaining = pd.read_csv(path_csv).sort_values(["uid", "segment_id"])
            self.assertEqual(deleted_count, 1)
            self.assertEqual(
                remaining[["uid", "segment_id"]].values.tolist(),
                [[7, 1], [8, 1]],
            )
            self.assertEqual(remaining["mode"].tolist(), ["GG", "TS"])
            self.assertEqual(second_key, (7, 2))
        finally:
            for suffix in ("", ".tmp"):
                target = path_csv + suffix
                if LabelPath.os.path.exists(target):
                    LabelPath.os.remove(target)

    def test_startup_opens_first_unlabeled_od_without_prompt(self):
        traj_df = pd.DataFrame([
            {"uid": 10, "idx_o": 0},
            {"uid": 10, "idx_o": 1},
            {"uid": 20, "idx_o": 0},
        ])
        labeled_modes = {(10, 0): "TG", (10, 1): "TS"}

        self.assertEqual(LabelPath.first_unlabeled_index(traj_df, labeled_modes), 2)
        self.assertEqual(
            LabelPath.first_unlabeled_index(
                traj_df, {**labeled_modes, (20, 0): "GG"},
            ),
            0,
        )

    def test_startup_and_navigation_skip_ignored_points(self):
        traj_df = pd.DataFrame([
            {"uid": 10, "idx_o": 0},
            {"uid": 10, "idx_o": 1},
            {"uid": 10, "idx_o": 2},
            {"uid": 10, "idx_o": 3},
        ])
        ignored = {(10, 1), (10, 2)}

        self.assertEqual(LabelPath.first_unlabeled_index(traj_df, {}, ignored), 0)
        self.assertEqual(LabelPath.next_nonignored_index(traj_df, ignored, 0, 1), 3)
        self.assertEqual(LabelPath.next_nonignored_index(traj_df, ignored, 3, -1), 0)
        all_resolved = {(10, 0): "TG", (10, 3): "GG"}
        self.assertEqual(
            LabelPath.first_unlabeled_index(traj_df, all_resolved, ignored), 0,
        )
        self.assertEqual(
            LabelPath.first_unlabeled_index(
                traj_df, {(10, 1): "TG", (10, 2): "TS", (10, 3): "GG"},
                {(10, 0)},
            ),
            1,
        )

    def test_navigation_skips_final_od_when_terminal_point_is_ignored(self):
        traj_df = pd.DataFrame([
            {"uid": 10, "idx_o": 0, "idx_d": 1},
            {"uid": 10, "idx_o": 1, "idx_d": 2},
            {"uid": 10, "idx_o": 2, "idx_d": 3},
        ])
        ignored = {(10, 3)}

        self.assertTrue(LabelPath.segment_is_reviewable(traj_df, 1, ignored))
        self.assertFalse(LabelPath.segment_is_reviewable(traj_df, 2, ignored))
        self.assertEqual(
            LabelPath.next_nonignored_index(traj_df, ignored, 1, 1),
            len(traj_df),
        )
        self.assertEqual(
            LabelPath.next_nonignored_index(traj_df, ignored, len(traj_df), -1),
            1,
        )

    def test_startup_and_navigation_skip_excluded_uid(self):
        traj_df = pd.DataFrame([
            {"uid": 10, "idx_o": 0},
            {"uid": 10, "idx_o": 1},
            {"uid": 20, "idx_o": 0},
            {"uid": 20, "idx_o": 1},
        ])

        self.assertEqual(
            LabelPath.first_unlabeled_index(traj_df, {}, set(), {10}), 2,
        )
        self.assertEqual(
            LabelPath.next_nonignored_index(traj_df, set(), 0, 1, {10}), 2,
        )

    def test_path_navigation_uses_regular_od_order_without_mode_requirement(self):
        traj_df = pd.DataFrame([
            {"uid": 10, "idx_o": 0, "idx_d": 1},
            {"uid": 10, "idx_o": 1, "idx_d": 2},
            {"uid": 10, "idx_o": 2, "idx_d": 3},
            {"uid": 20, "idx_o": 0, "idx_d": 1},
        ])
        self.assertEqual(LabelPath.next_nonignored_index(
            traj_df, {(10, 1)}, 0, 1,
        ), 2)
        self.assertEqual(LabelPath.next_nonignored_index(
            traj_df, set(), 2, 1,
        ), 3)

    def test_auto_route_finds_shortest_connected_mode_road_path(self):
        road_cells = {
            (0, 0, 0), (1, -1, 0), (2, -2, 0), (3, -3, 0),
            (1, 0, -1), (2, -1, -1),
        }

        self.assertEqual(
            LabelPath.find_road_route((0, 0, 0), (3, -3, 0), road_cells),
            [(0, 0, 0), (1, -1, 0), (2, -2, 0), (3, -3, 0)],
        )
        self.assertEqual(
            LabelPath.nearest_road_hex((0, 1, -1), road_cells),
            (0, 0, 0),
        )

    def test_dated_copy_uses_dataset_date_and_excludes_uid_without_main_mutation(self):
        output_dir = LabelPath.os.path.join(
            LabelPath.os.getcwd(), "tests", "_runtime_ignore_output",
        )
        active_path = LabelPath.os.path.join(output_dir, "traj_labeled.csv")
        export_path = LabelPath.os.path.join(
            output_dir, "labeled_data_20230917.csv",
        )
        excluded_path = LabelPath.os.path.join(
            output_dir, LabelPath.EXCLUDED_UID_FILENAME,
        )
        try:
            pd.DataFrame([
                {"uid": 202309170001, "idx_o": 0, "idx_d": 1, "mode": "TG"},
                {"uid": 202309170002, "idx_o": 0, "idx_d": 1, "mode": "GG"},
            ]).to_csv(active_path, index=False)
            self.assertEqual(
                LabelPath.derive_labeled_data_date(
                    r"data\dataset_multicity_20230917_unpacked.csv",
                ),
                "20230917",
            )

            renderer = LabelPath.PathRenderer.__new__(LabelPath.PathRenderer)
            renderer.output_dir = output_dir
            renderer.export_date = "20230917"
            renderer.excluded_uids = set()
            renderer.state = mock.Mock(uid=202309170002)
            renderer.traj_df = pd.DataFrame([
                {"uid": 202309170001, "idx_o": 0},
                {"uid": 202309170002, "idx_o": 0},
            ])
            renderer.ignored_points = set()
            renderer._uid_segment_positions = {
                202309170001: np.asarray([0]),
                202309170002: np.asarray([1]),
            }
            renderer._draw_uid_navigation_list = mock.Mock()
            renderer.fig = mock.Mock()
            renderer.segment_select_callback = mock.Mock()

            renderer._toggle_excluded_uid(202309170001)

            self.assertEqual(
                LabelPath.load_excluded_uids(output_dir), {202309170001},
            )
            self.assertEqual(pd.read_csv(active_path)["uid"].tolist(), [
                202309170001, 202309170002,
            ])
            self.assertEqual(
                pd.read_csv(export_path)["uid"].tolist(), [202309170002],
            )

            renderer._toggle_excluded_uid(202309170001)
            self.assertEqual(LabelPath.load_excluded_uids(output_dir), set())
            self.assertEqual(pd.read_csv(export_path)["uid"].tolist(), [
                202309170001, 202309170002,
            ])
        finally:
            for path in (active_path, export_path, excluded_path):
                if LabelPath.os.path.exists(path):
                    LabelPath.os.remove(path)

    def test_effective_segment_merges_intervals_across_ignored_point(self):
        traj_df = pd.DataFrame([
            {"uid": 7, "idx_o": 0, "idx_d": 1,
             "x_o": 0, "y_o": 0, "z_o": 0, "x_d": 1, "y_d": 0, "z_d": -1,
             "time_d": 10.0, "dist_d": 100.0, "velocity_d": 36.0},
            {"uid": 7, "idx_o": 1, "idx_d": 2,
             "x_o": 1, "y_o": 0, "z_o": -1, "x_d": 2, "y_d": 0, "z_d": -2,
             "time_d": 20.0, "dist_d": 400.0, "velocity_d": 72.0},
            {"uid": 7, "idx_o": 2, "idx_d": 3,
             "x_o": 2, "y_o": 0, "z_o": -2, "x_d": 3, "y_d": 0, "z_d": -3,
             "time_d": 10.0, "dist_d": 100.0, "velocity_d": 36.0},
        ])

        row = LabelPath.effective_segment_row(traj_df, 0, {(7, 1)})

        self.assertEqual((row["x_o"], row["x_d"], row["idx_d"]), (0, 2, 2))
        self.assertEqual(row["time_d"], 30.0)
        self.assertEqual(row["dist_d"], 500.0)
        self.assertAlmostEqual(row["velocity_d"], 60.0)

    def test_ignore_point_removes_two_adjacent_labels_without_archive(self):
        traj_df = pd.DataFrame([
            {"uid": 7, "idx_o": 0},
            {"uid": 7, "idx_o": 1},
            {"uid": 7, "idx_o": 2},
        ])
        output_dir = LabelPath.os.path.join(
            LabelPath.os.getcwd(), "tests", "_runtime_ignore_output",
        )
        artifact_names = (
            "traj_labeled.csv",
            LabelPath.PATH_LABEL_FILENAME,
            LabelPath.IGNORED_POINT_FILENAME,
        )
        for name in artifact_names:
            path = LabelPath.os.path.join(output_dir, name)
            if LabelPath.os.path.exists(path):
                LabelPath.os.remove(path)
        try:
            active_path = LabelPath.os.path.join(output_dir, "traj_labeled.csv")
            pd.DataFrame([
                {"uid": 7, "idx_o": 0, "mode": "TG"},
                {"uid": 7, "idx_o": 1, "mode": "TS"},
                {"uid": 7, "idx_o": 2, "mode": "GG"},
            ]).to_csv(active_path, index=False)

            renderer = LabelPath.PathRenderer.__new__(LabelPath.PathRenderer)
            renderer.traj_df = traj_df
            renderer.output_dir = output_dir
            renderer.ignored_points = set()
            renderer.labeled_modes = {(7, 0): "TG", (7, 1): "TS", (7, 2): "GG"}
            renderer.labeled_paths = {}
            renderer.export_date = None
            renderer.excluded_uids = set()
            renderer._uid_segment_positions = {7: np.asarray([0, 1, 2])}
            renderer._uid_resolved_counts = {7: 3}
            renderer.current_idx = 2
            renderer.segment_select_callback = mock.Mock()
            renderer.refresh_uid_segment_list = mock.Mock()

            renderer._toggle_ignored_point(1)

            self.assertEqual(renderer.ignored_points, {(7, 1)})
            self.assertEqual(renderer.labeled_modes, {(7, 2): "GG"})
            self.assertEqual(
                set(pd.read_csv(active_path)["idx_o"].tolist()), {2},
            )
            self.assertFalse(LabelPath.os.path.exists(
                LabelPath.os.path.join(
                    output_dir, "traj_labeled_invalidated.csv",
                ),
            ))
            self.assertEqual(LabelPath.load_ignored_points(output_dir), {(7, 1)})

            # A merged A->C label made while B is ignored also becomes stale
            # when B is restored, so it must leave the active CSV as well.
            active = pd.read_csv(active_path)
            active = pd.concat([
                active,
                pd.DataFrame([{"uid": 7, "idx_o": 0, "mode": "BR"}]),
            ], ignore_index=True)
            active.to_csv(active_path, index=False)
            renderer.labeled_modes[(7, 0)] = "BR"

            renderer._toggle_ignored_point(1)
            self.assertEqual(renderer.ignored_points, set())
            self.assertEqual(renderer.labeled_modes, {(7, 2): "GG"})
            self.assertEqual(LabelPath.load_ignored_points(output_dir), set())
            self.assertEqual(pd.read_csv(active_path)["idx_o"].tolist(), [2])
        finally:
            for name in artifact_names:
                path = LabelPath.os.path.join(output_dir, name)
                if LabelPath.os.path.exists(path):
                    LabelPath.os.remove(path)

    def test_ignore_terminal_point_removes_only_the_final_od_label(self):
        traj_df = pd.DataFrame([
            {"uid": 7, "idx_o": 0, "idx_d": 1},
            {"uid": 7, "idx_o": 1, "idx_d": 2},
            {"uid": 7, "idx_o": 2, "idx_d": 3},
        ])
        output_dir = LabelPath.os.path.join(
            LabelPath.os.getcwd(), "tests", "_runtime_terminal_ignore_output",
        )
        active_path = LabelPath.os.path.join(output_dir, "traj_labeled.csv")
        ignored_path = LabelPath.os.path.join(
            output_dir, LabelPath.IGNORED_POINT_FILENAME,
        )
        try:
            LabelPath.os.makedirs(output_dir, exist_ok=True)
            pd.DataFrame([
                {"uid": 7, "idx_o": 0, "mode": "TG"},
                {"uid": 7, "idx_o": 1, "mode": "TS"},
                {"uid": 7, "idx_o": 2, "mode": "GG"},
            ]).to_csv(active_path, index=False)

            renderer = LabelPath.PathRenderer.__new__(LabelPath.PathRenderer)
            renderer.traj_df = traj_df
            renderer.output_dir = output_dir
            renderer.ignored_points = set()
            renderer.labeled_modes = {
                (7, 0): "TG", (7, 1): "TS", (7, 2): "GG",
            }
            renderer.labeled_paths = {}
            renderer.export_date = None
            renderer.excluded_uids = set()
            renderer._uid_segment_positions = {7: np.asarray([0, 1, 2])}
            renderer._uid_resolved_counts = {7: 3}
            renderer.current_idx = 0
            renderer.segment_select_callback = mock.Mock()
            renderer.refresh_uid_segment_list = mock.Mock()

            renderer._toggle_ignored_point(
                2, point_key=(7, 3), destination=True,
            )

            self.assertEqual(renderer.ignored_points, {(7, 3)})
            self.assertEqual(
                renderer.labeled_modes,
                {(7, 0): "TG", (7, 1): "TS"},
            )
            self.assertEqual(renderer._uid_resolved_counts[7], 3)
            self.assertEqual(pd.read_csv(active_path)["idx_o"].tolist(), [0, 1])
            self.assertEqual(LabelPath.load_ignored_points(output_dir), {(7, 3)})

            renderer._toggle_ignored_point(
                2, point_key=(7, 3), destination=True,
            )
            self.assertEqual(renderer.ignored_points, set())
            self.assertEqual(renderer._uid_resolved_counts[7], 2)
            self.assertEqual(LabelPath.load_ignored_points(output_dir), set())
        finally:
            for path in (active_path, ignored_path):
                if LabelPath.os.path.exists(path):
                    LabelPath.os.remove(path)

    def test_normalize_storage_sorts_active_records(self):
        output_dir = LabelPath.os.path.join(
            LabelPath.os.getcwd(), "tests", "_runtime_ignore_output",
        )
        active_path = LabelPath.os.path.join(output_dir, "traj_labeled.csv")
        try:
            pd.DataFrame([
                {"uid": 7, "idx_o": 1, "idx_d": 2, "mode": "TS", "path": "old"},
                {"uid": 7, "idx_o": 0, "idx_d": 1, "mode": "TG"},
                {"uid": 6, "idx_o": 9, "idx_d": 10, "mode": "GG"},
            ]).to_csv(active_path, index=False)

            reordered = LabelPath.normalize_label_storage(output_dir)

            self.assertEqual(reordered, 1)
            active = pd.read_csv(active_path)
            self.assertNotIn("traj", active.columns)
            self.assertNotIn("path", active.columns)
            self.assertEqual(
                list(active[["uid", "idx_o", "idx_d"]].itertuples(
                    index=False, name=None,
                )),
                [(6, 9, 10), (7, 0, 1), (7, 1, 2)],
            )
        finally:
            if LabelPath.os.path.exists(active_path):
                LabelPath.os.remove(active_path)

    def test_saved_point_colors_match_road_overlay_colors(self):
        for mode, color in LabelPath.MODE_COLORS.items():
            self.assertEqual(LabelPath.LABELED_POINT_COLORS[mode], color)

    def test_context_points_use_label_colors_and_keep_full_window_visible(self):
        traj_df = pd.DataFrame([
            {"uid": 7, "idx_o": i, "x_o": i, "y_o": 0, "z_o": -i,
             "x_d": i + 1, "y_d": 0, "z_d": -(i + 1)}
            for i in range(5)
        ])
        state = mock.Mock(uid=7, start=(2, 0, -2), end=(3, 0, -3))
        renderer = LabelPath.PathRenderer.__new__(LabelPath.PathRenderer)
        renderer.fig, renderer.ax = plt.subplots()
        renderer.ax.set_xlim(-1, 5)
        renderer.ax.set_ylim(-1, 1)
        renderer.traj_df = traj_df
        renderer.current_idx = 2
        renderer.labeled_modes = {
            (7, 0): "TG", (7, 1): "TS", (7, 3): "GG", (7, 4): "GSD",
        }

        with mock.patch.object(LabelPath, "hex_to_mercator", side_effect=lambda x, y, z: (x, y)), \
             mock.patch.object(LabelPath, "mercator_wgs84_to_gcj02", side_effect=lambda x, y: (x, y)):
            renderer._draw_context_points(state)

        collections = renderer.ax.collections
        self.assertEqual(len(collections), 4)
        self.assertTrue(all(c.get_alpha() == LabelPath.CONTEXT_ALPHA for c in collections))
        rendered_colors = [
            color[:3]
            for collection in collections
            for color in collection.get_facecolors()
        ]
        self.assertTrue(any(np.allclose(color, LabelPath.MODE_COLORS["TS"])
                            for color in rendered_colors))
        self.assertTrue(any(np.allclose(color, LabelPath.MODE_COLORS["GG"])
                            for color in rendered_colors))
        hovered_sequences = {
            record[2]
            for _artist, records in renderer._point_scatter_meta
            for record in records
        }
        self.assertEqual(hovered_sequences, {1, 2, 5, 6})
        plt.close(renderer.fig)

    def test_point_sequence_uses_original_uid_list_number_across_merged_od(self):
        renderer = LabelPath.PathRenderer.__new__(LabelPath.PathRenderer)
        renderer.traj_df = pd.DataFrame([
            {"uid": 7, "idx_o": 10, "idx_d": 11},
            {"uid": 7, "idx_o": 11, "idx_d": 12},
            {"uid": 7, "idx_o": 12, "idx_d": 13},
        ])
        renderer._uid_segment_positions = {7: np.asarray([0, 1, 2])}
        renderer._uid_point_sequence_cache = {}

        self.assertEqual(renderer._point_sequence_number(7, 10, 0), 1)
        self.assertEqual(renderer._point_sequence_number(7, 12, 0, True), 3)
        self.assertEqual(renderer._point_sequence_number(7, 13, 2, True), 4)

    def test_main_view_hover_shows_trajectory_point_sequence(self):
        renderer = LabelPath.PathRenderer.__new__(LabelPath.PathRenderer)
        renderer.fig, renderer.ax = plt.subplots()
        renderer._point_hover = renderer.ax.annotate("", (0, 0))
        renderer._point_hover.set_visible(False)
        renderer._poi_hover = None
        renderer._poi_scatter_meta = []
        artist = mock.Mock()
        artist.contains.return_value = (True, {"ind": [0]})
        renderer._point_scatter_meta = [(artist, [(1.0, 2.0, 17)])]

        renderer._on_poi_hover(mock.Mock(inaxes=renderer.ax))

        self.assertTrue(renderer._point_hover.get_visible())
        self.assertEqual(renderer._point_hover.get_text(), "序号: 17")
        self.assertEqual(renderer._point_hover.xy, (1.0, 2.0))
        plt.close(renderer.fig)

    def test_road_hex_hover_lists_only_enabled_road_layers(self):
        renderer = LabelPath.PathRenderer.__new__(LabelPath.PathRenderer)
        renderer.fig, renderer.ax = plt.subplots()
        renderer._point_hover = None
        renderer._poi_hover = None
        renderer._point_scatter_meta = []
        renderer._poi_scatter_meta = []
        renderer._road_hover = renderer.ax.annotate("", (0, 0))
        renderer._road_hover.set_visible(False)
        hex_key = (1, -2, 1)
        renderer.raw_mapdata = {
            hex_key: {"code": (1 << 12) | (1 << 9) | (1 << 5)},  # TG + GG + L2
        }
        renderer.road_sets = {}
        renderer.road_visibility = {
            mode: mode not in {"TG", "L2"}
            for mode in LabelPath.DISPLAY_MODE_LIST
        }
        renderer._hovered_hex_from_event = mock.Mock(return_value=hex_key)
        event = mock.Mock(inaxes=renderer.ax, xdata=4.0, ydata=5.0)

        renderer._on_poi_hover(event)

        self.assertTrue(renderer._road_hover.get_visible())
        self.assertIn("[GG]高速", renderer._road_hover.get_text())
        self.assertNotIn("TG", renderer._road_hover.get_text())
        self.assertNotIn("L2", renderer._road_hover.get_text())

        renderer.road_visibility["L2"] = True
        renderer._on_poi_hover(event)
        self.assertIn("[L2]二级公路", renderer._road_hover.get_text())

        renderer.road_visibility["GG"] = False
        renderer.road_visibility["L2"] = False
        renderer._on_poi_hover(event)
        self.assertFalse(renderer._road_hover.get_visible())
        plt.close(renderer.fig)

    def test_hovered_display_coordinate_round_trips_to_hex(self):
        hex_key = (1786, -2429, 643)
        mx, my = LabelPath.hex_to_mercator(*hex_key)
        gx, gy = LabelPath.mercator_wgs84_to_gcj02(mx, my)
        renderer = LabelPath.PathRenderer.__new__(LabelPath.PathRenderer)

        result = renderer._hovered_hex_from_event(mock.Mock(
            xdata=float(gx), ydata=float(gy),
        ))

        self.assertEqual(result, hex_key)

    def test_context_draws_all_visible_uid_points_without_rescaling(self):
        traj_df = pd.DataFrame([
            {"uid": 7, "idx_o": i, "x_o": i, "y_o": 0, "z_o": -i,
             "x_d": i + 1, "y_d": 0, "z_d": -(i + 1)}
            for i in range(25)
        ])
        state = mock.Mock(uid=7, start=(12, 0, -12), end=(13, 0, -13))
        renderer = LabelPath.PathRenderer.__new__(LabelPath.PathRenderer)
        renderer.fig, renderer.ax = plt.subplots()
        renderer.ax.set_xlim(0, 25)
        renderer.ax.set_ylim(-1, 1)
        renderer.traj_df = traj_df
        renderer.current_idx = 12
        renderer.labeled_modes = {}
        before = (renderer.ax.get_xlim(), renderer.ax.get_ylim())

        with mock.patch.object(LabelPath, "hex_to_mercator", side_effect=lambda x, y, z: (x, y)), \
             mock.patch.object(LabelPath, "mercator_wgs84_to_gcj02", side_effect=lambda x, y: (x, y)):
            renderer._draw_context_points(state)

        visible_points = sum(len(collection.get_offsets()) for collection in renderer.ax.collections)
        self.assertEqual(visible_points, 24)
        self.assertEqual((renderer.ax.get_xlim(), renderer.ax.get_ylim()), before)
        plt.close(renderer.fig)

    def test_segment_info_shows_uid_od_progress(self):
        renderer = LabelPath.PathRenderer.__new__(LabelPath.PathRenderer)
        renderer.fig, renderer.ax = plt.subplots()
        renderer.ax_info = renderer.ax
        renderer.state = mock.Mock(
            uid=7,
            row=pd.Series({"dist_d": 120.5, "time_d": 8, "velocity_d": 54.2}),
        )
        renderer.traj_df = pd.DataFrame({"uid": [2, 7, 7, 2, 7]})
        renderer.current_idx = 2

        renderer._draw_segment_info()

        self.assertEqual(renderer._uid_od_progress(), (2, 3))
        self.assertIn("UID OD:   2 / 3", renderer.ax.texts[-1].get_text())
        plt.close(renderer.fig)

    def test_right_info_cards_share_width_style_and_compact_spacing(self):
        renderer = LabelPath.PathRenderer.__new__(LabelPath.PathRenderer)
        renderer.fig, renderer.ax_info = plt.subplots()

        renderer._init_info_panel()

        cards = list(renderer._info_cards.values())
        self.assertEqual(len(cards), 4)
        self.assertTrue(all(card.get_width() == cards[0].get_width() for card in cards))
        self.assertTrue(all(card.get_facecolor() == cards[0].get_facecolor() for card in cards))
        ordered = sorted(cards, key=lambda card: card.get_y())
        gaps = [
            ordered[index + 1].get_y()
            - (ordered[index].get_y() + ordered[index].get_height())
            for index in range(len(ordered) - 1)
        ]
        self.assertTrue(all(0.0 < gap <= 0.025 for gap in gaps))
        plt.close(renderer.fig)

    def test_l2_is_display_only_road_layer(self):
        hex_key = (1, -2, 1)
        road_sets = LabelPath.hex_mapdata_to_road_sets({
            hex_key: {"code": 1 << 5},
        })

        self.assertIn("L2", LabelPath.DISPLAY_MODE_LIST)
        self.assertNotIn("L2", LabelPath.MODE_LIST)
        self.assertEqual(road_sets["L2"], {hex_key})
        self.assertTrue(all(
            hex_key not in road_sets[mode] for mode in LabelPath.MODE_LIST
        ))

    def test_road_buttons_toggle_layer_without_rescaling_and_persist_on_redraw(self):
        renderer = LabelPath.PathRenderer.__new__(LabelPath.PathRenderer)
        renderer.fig, renderer.ax = plt.subplots()
        renderer.ax_info = renderer.ax
        renderer.road_visibility = {
            mode: mode != "L2" for mode in LabelPath.DISPLAY_MODE_LIST
        }
        renderer._road_artists = {}
        renderer._road_buttons = {}
        renderer._init_road_toggle_buttons()

        button_positions = [
            renderer._road_buttons[mode].ax.get_position()
            for mode in LabelPath.DISPLAY_MODE_LIST
        ]
        self.assertEqual(len({round(position.x0, 6) for position in button_positions}), 1)
        self.assertEqual(len({round(position.y0, 6) for position in button_positions}), 6)
        self.assertFalse(renderer.road_visibility["L2"])
        self.assertIn("关", renderer._road_buttons["L2"].label.get_text())

        renderer.ax.set_xlim(10, 20)
        renderer.ax.set_ylim(30, 40)
        artist = renderer.ax.scatter([15], [35])
        renderer._road_artists["TG"] = artist
        before = (renderer.ax.get_xlim(), renderer.ax.get_ylim())

        renderer._toggle_road_layer("TG")

        self.assertFalse(renderer.road_visibility["TG"])
        self.assertFalse(artist.get_visible())
        self.assertIn("关", renderer._road_buttons["TG"].label.get_text())
        self.assertEqual((renderer.ax.get_xlim(), renderer.ax.get_ylim()), before)

        l2_artist = renderer.ax.scatter([16], [36])
        l2_artist.set_visible(False)
        renderer._road_artists["L2"] = l2_artist
        renderer._toggle_road_layer("L2")
        self.assertTrue(renderer.road_visibility["L2"])
        self.assertTrue(l2_artist.get_visible())
        self.assertIn("开", renderer._road_buttons["L2"].label.get_text())
        self.assertEqual((renderer.ax.get_xlim(), renderer.ax.get_ylim()), before)

        renderer._road_artists = {}
        renderer._road_display = {
            "TG": (
                np.asarray([15.0]), np.asarray([35.0]),
                np.asarray([15.0]), np.asarray([35.0]),
            ),
        }
        renderer._mx_min, renderer._mx_max = 10, 20
        renderer._my_min, renderer._my_max = 30, 40
        renderer._build_hex_road_overlay(None)

        self.assertNotIn("TG", renderer._road_artists)
        self.assertEqual((renderer.ax.get_xlim(), renderer.ax.get_ylim()), before)
        plt.close(renderer.fig)

    def test_path_mode_keeps_exactly_one_road_network_active(self):
        renderer = LabelPath.PathRenderer.__new__(LabelPath.PathRenderer)
        renderer.fig, renderer.ax = plt.subplots()
        renderer.annotation_mode = LabelPath.ANNOTATION_PATH
        renderer.active_path_mode = None
        renderer.road_visibility = {
            mode: True for mode in LabelPath.DISPLAY_MODE_LIST
        }
        renderer.road_sets = {
            mode: {(index, -index, 0)}
            for index, mode in enumerate(LabelPath.DISPLAY_MODE_LIST, 1)
        }
        renderer.state = mock.Mock(path_history=[(0, 0, 0)])
        renderer._road_artists = {
            mode: renderer.ax.scatter([index], [index])
            for index, mode in enumerate(LabelPath.DISPLAY_MODE_LIST, 1)
        }
        renderer._road_buttons = {}
        renderer._road_hover = None
        renderer.path_line = mock.Mock()
        renderer.refresh = mock.Mock()

        renderer.set_active_path_mode("TG")

        self.assertEqual(renderer.active_path_mode, "TG")
        self.assertIs(renderer.state.multi_mapdata, renderer.road_sets["TG"])
        self.assertEqual(
            [mode for mode, visible in renderer.road_visibility.items() if visible],
            ["TG"],
        )
        renderer.state.clear_path.assert_called_once_with()

        renderer.state.path_history = [(1, -1, 0)]
        renderer.set_active_path_mode("GG")
        self.assertEqual(
            [mode for mode, visible in renderer.road_visibility.items() if visible],
            ["GG"],
        )
        self.assertFalse(renderer.road_visibility["L2"])
        self.assertEqual(renderer.state.clear_path.call_count, 2)
        plt.close(renderer.fig)

    def test_clickable_label_buttons_show_1_to_6_mapping_and_share_callback(self):
        renderer = LabelPath.PathRenderer.__new__(LabelPath.PathRenderer)
        renderer.fig = plt.figure()
        renderer._label_buttons = {}
        renderer.label_select_callback = mock.Mock()

        renderer._init_label_buttons()

        self.assertEqual(set(renderer._label_buttons), set(LabelPath.LABEL_OPTIONS.values()))
        for key, mode in LabelPath.LABEL_OPTIONS.items():
            self.assertTrue(renderer._label_buttons[mode].label.get_text().startswith(f"{key}  {mode}"))
        renderer._request_label_mode("GSD")
        renderer.label_select_callback.assert_called_once_with("GSD")
        plt.close(renderer.fig)

    def test_single_button_switches_between_mode_and_path_workflows(self):
        renderer = LabelPath.PathRenderer.__new__(LabelPath.PathRenderer)
        renderer.fig = plt.figure()
        renderer.annotation_mode = LabelPath.ANNOTATION_MODE
        renderer.annotation_toggle_callback = mock.Mock()
        renderer._annotation_toggle_button = None

        renderer._init_annotation_toggle_button()
        self.assertIn("模式标注", renderer._annotation_toggle_button.label.get_text())
        renderer._request_annotation_toggle()
        renderer.annotation_toggle_callback.assert_called_once_with()

        renderer.annotation_mode = LabelPath.ANNOTATION_PATH
        renderer._update_annotation_toggle_style()
        self.assertIn("路径标注", renderer._annotation_toggle_button.label.get_text())
        plt.close(renderer.fig)

    def test_uid_od_list_uses_saved_mode_colors_and_highlights_current_row(self):
        traj_df = pd.DataFrame([
            {"uid": 7, "idx_o": i, "x_o": i, "y_o": 0, "z_o": -i,
             "x_d": i + 1, "y_d": 0, "z_d": -(i + 1)}
            for i in range(8)
        ])
        renderer = LabelPath.PathRenderer.__new__(LabelPath.PathRenderer)
        renderer.fig, renderer.ax_uid_list = plt.subplots()
        renderer.traj_df = traj_df
        renderer.state = mock.Mock(uid=7)
        renderer.current_idx = 3
        renderer.labeled_modes = {(7, 1): "TG", (7, 5): "GG"}
        renderer.ignored_points = {(7, 4)}
        renderer._uid_segment_positions = {7: np.arange(8, dtype=np.int32)}
        renderer._uid_list_uid = None
        renderer._uid_list_offset = 0
        renderer._uid_list_hit_rows = []

        renderer._draw_uid_segment_list(ensure_current=True)

        rendered_colors = [
            color[:3]
            for collection in renderer.ax_uid_list.collections
            for color in collection.get_facecolors()
        ]
        self.assertTrue(any(np.allclose(color, LabelPath.MODE_COLORS["TG"])
                            for color in rendered_colors))
        self.assertTrue(any(np.allclose(color, LabelPath.MODE_COLORS["GG"])
                            for color in rendered_colors))
        self.assertTrue(any(patch.get_edgecolor()[2] > 0.7
                            for patch in renderer.ax_uid_list.patches))
        self.assertTrue(any(text.get_text() == "忽略"
                            for text in renderer.ax_uid_list.texts))
        self.assertEqual(len(renderer._uid_list_hit_rows), 8)
        plt.close(renderer.fig)

    def test_uid_od_list_scrolls_and_clicks_an_absolute_segment(self):
        traj_df = pd.DataFrame([
            {"uid": 9, "idx_o": i, "x_o": i, "y_o": 0, "z_o": -i,
             "x_d": i + 1, "y_d": 0, "z_d": -(i + 1)}
            for i in range(40)
        ])
        renderer = LabelPath.PathRenderer.__new__(LabelPath.PathRenderer)
        renderer.fig, renderer.ax_uid_list = plt.subplots()
        renderer.traj_df = traj_df
        renderer.state = mock.Mock(uid=9)
        renderer.current_idx = 0
        renderer.labeled_modes = {}
        renderer._uid_segment_positions = {9: np.arange(40, dtype=np.int32)}
        renderer._uid_list_uid = None
        renderer._uid_list_offset = 0
        renderer._uid_list_hit_rows = []
        renderer.segment_select_callback = mock.Mock()
        renderer._draw_uid_segment_list(ensure_current=True)

        renderer._on_uid_list_scroll(mock.Mock(
            inaxes=renderer.ax_uid_list, step=-1, button="down",
        ))
        self.assertEqual(renderer._uid_list_offset, LabelPath.UID_LIST_SCROLL_STEP)

        y_min, y_max, entry = renderer._uid_list_hit_rows[0]
        renderer._on_uid_list_click(mock.Mock(
            inaxes=renderer.ax_uid_list, ydata=(y_min + y_max) / 2, button=1,
        ))
        renderer.segment_select_callback.assert_called_once_with(entry["position"])
        self.assertEqual(entry["position"], LabelPath.UID_LIST_SCROLL_STEP)

        renderer._toggle_ignored_point = mock.Mock()
        renderer._on_uid_list_click(mock.Mock(
            inaxes=renderer.ax_uid_list, ydata=(y_min + y_max) / 2, button=3,
        ))
        renderer._toggle_ignored_point.assert_called_once_with(
            entry["position"], point_key=entry["key"], destination=False,
        )
        plt.close(renderer.fig)

    def test_uid_point_list_includes_and_right_clicks_terminal_point(self):
        traj_df = pd.DataFrame([
            {"uid": 9, "idx_o": i, "idx_d": i + 1,
             "x_o": i, "y_o": 0, "z_o": -i,
             "x_d": i + 1, "y_d": 0, "z_d": -(i + 1)}
            for i in range(3)
        ])
        renderer = LabelPath.PathRenderer.__new__(LabelPath.PathRenderer)
        renderer.fig, renderer.ax_uid_list = plt.subplots()
        renderer.traj_df = traj_df
        renderer.state = mock.Mock(uid=9)
        renderer.current_idx = 0
        renderer.labeled_modes = {}
        renderer.ignored_points = set()
        renderer._uid_segment_positions = {9: np.arange(3, dtype=np.int32)}
        renderer._uid_list_uid = None
        renderer._uid_list_offset = 0
        renderer._uid_list_hit_rows = []
        renderer._toggle_ignored_point = mock.Mock()

        renderer._draw_uid_segment_list(ensure_current=True)

        self.assertEqual(len(renderer._uid_list_hit_rows), 4)
        y_min, y_max, terminal = renderer._uid_list_hit_rows[-1]
        self.assertEqual(terminal, {
            "position": 2, "key": (9, 3), "destination": True,
        })
        renderer._on_uid_list_click(mock.Mock(
            inaxes=renderer.ax_uid_list,
            ydata=(y_min + y_max) / 2,
            button=3,
        ))
        renderer._toggle_ignored_point.assert_called_once_with(
            2, point_key=(9, 3), destination=True,
        )
        plt.close(renderer.fig)

    def test_saved_path_list_scrolls_and_opens_recorded_window(self):
        renderer = LabelPath.PathRenderer.__new__(LabelPath.PathRenderer)
        renderer.fig, renderer.ax_path_list = plt.subplots()
        renderer.state = mock.Mock(uid=7)
        renderer.current_idx = 0
        renderer.traj_df = pd.DataFrame([
            {"uid": 7, "idx_o": 100 + index} for index in range(35)
        ])
        renderer._uid_segment_positions = {7: np.arange(35, dtype=np.int32)}
        renderer.labeled_paths = {
            (7, segment_id): [(0, 0, 0), (1, -1, 0)]
            for segment_id in range(1, 36)
        }
        renderer.labeled_path_modes = {
            key: ("TG" if key[1] % 2 else "GG")
            for key in renderer.labeled_paths
        }
        renderer.labeled_path_metadata = {
            (7, segment_id): {
                "anchor_idx_o": 99 + segment_id,
                "steps": segment_id,
            }
            for segment_id in range(1, 36)
        }
        renderer.selected_path_key = None
        renderer.annotation_mode = LabelPath.ANNOTATION_PATH
        renderer.set_active_path_mode = mock.Mock()
        renderer._path_list_uid = None
        renderer._path_list_offset = 0
        renderer._path_list_hit_rows = []
        renderer.segment_select_callback = mock.Mock()
        renderer._draw_uid_path_list(ensure_current=True)

        renderer._on_path_list_scroll(mock.Mock(
            inaxes=renderer.ax_path_list, step=-1, button="down",
        ))
        self.assertEqual(renderer._path_list_offset, LabelPath.PATH_LIST_SCROLL_STEP)

        y_min, y_max, key = renderer._path_list_hit_rows[0]
        renderer._on_path_list_click(mock.Mock(
            inaxes=renderer.ax_path_list,
            ydata=(y_min + y_max) / 2,
            button=1,
        ))

        self.assertEqual(key, (7, 6))
        self.assertEqual(renderer.selected_path_key, key)
        renderer.set_active_path_mode.assert_called_once_with("GG")
        renderer.segment_select_callback.assert_called_once_with(5)
        plt.close(renderer.fig)

    def test_saved_path_in_memory_keys_are_compacted_for_one_uid(self):
        renderer = LabelPath.PathRenderer.__new__(LabelPath.PathRenderer)
        renderer.labeled_paths = {
            (7, 2): [(0, 0, 0)],
            (7, 4): [(1, -1, 0)],
            (8, 3): [(2, -2, 0)],
        }
        renderer.labeled_path_modes = {
            (7, 2): "TG", (7, 4): "GG", (8, 3): "TS",
        }
        renderer.labeled_path_metadata = {
            (7, 2): {"steps": 1},
            (7, 4): {"steps": 2},
            (8, 3): {"steps": 3},
        }
        renderer.selected_path_key = (7, 4)
        renderer.editing_path_key = (7, 2)

        key_remap = renderer.renumber_labeled_paths(7)

        self.assertEqual(key_remap, {(7, 2): (7, 1), (7, 4): (7, 2)})
        self.assertEqual(
            set(renderer.labeled_paths), {(7, 1), (7, 2), (8, 3)},
        )
        self.assertEqual(renderer.labeled_path_modes[(7, 2)], "GG")
        self.assertEqual(renderer.labeled_path_metadata[(7, 1)]["steps"], 1)
        self.assertEqual(renderer.selected_path_key, (7, 2))
        self.assertEqual(renderer.editing_path_key, (7, 1))

    def test_uid_navigation_distinguishes_od_completion_and_saved_paths(self):
        traj_df = pd.DataFrame([
            {"uid": uid, "idx_o": idx_o}
            for uid in (10, 20, 30)
            for idx_o in range(2)
        ])
        renderer = LabelPath.PathRenderer.__new__(LabelPath.PathRenderer)
        renderer.fig, renderer.ax_uid_nav = plt.subplots()
        renderer.traj_df = traj_df
        renderer.state = mock.Mock(uid=20)
        renderer.current_idx = 2
        renderer.labeled_modes = {
            (10, 0): "TG", (10, 1): "TS", (20, 0): "GG",
        }
        renderer._uid_segment_positions = {
            10: np.asarray([0, 1]), 20: np.asarray([2, 3]), 30: np.asarray([4, 5]),
        }
        renderer._uid_nav_values = np.asarray([10, 20, 30])
        renderer._uid_nav_index = {10: 0, 20: 1, 30: 2}
        renderer._uid_resolved_counts = {10: 2, 20: 1, 30: 0}
        renderer._uid_path_counts = {10: 0, 20: 1, 30: 2}
        renderer.excluded_uids = {30}
        renderer._uid_nav_offset = 0
        renderer._uid_nav_hit_rows = []

        renderer._draw_uid_navigation_list(ensure_current=True)

        facecolors = renderer.ax_uid_nav.collections[0].get_facecolors()
        self.assertEqual(facecolors[0, 3], 1.0)
        self.assertEqual(facecolors[1, 3], 0.0)
        self.assertEqual(len(facecolors), 2)
        path_markers = renderer.ax_uid_nav.collections[1]
        self.assertEqual(len(path_markers.get_offsets()), 2)
        self.assertTrue(np.allclose(
            path_markers.get_facecolors()[0, :3],
            matplotlib.colors.to_rgb("#00a7b5"),
        ))
        self.assertEqual(len(renderer.ax_uid_nav.collections[2].get_offsets()), 1)
        self.assertIn("◆ 有路径", renderer.ax_uid_nav.texts[0].get_text())
        self.assertTrue(any(patch.get_edgecolor()[2] > 0.7
                            for patch in renderer.ax_uid_nav.patches))
        plt.close(renderer.fig)

    def test_uid_navigation_orders_status_groups_then_numeric_uid(self):
        renderer = LabelPath.PathRenderer.__new__(LabelPath.PathRenderer)
        renderer._uid_nav_values = np.asarray(
            [42, 7, 30, 3, 5, 25, 18], dtype=np.int64,
        )
        renderer._uid_segment_positions = {
            uid: np.asarray([0, 1], dtype=np.int32)
            for uid in renderer._uid_nav_values
        }
        renderer._uid_resolved_counts = {
            42: 0, 7: 2, 30: 2, 3: 2, 5: 2, 25: 0, 18: 1,
        }
        renderer._uid_path_counts = {
            42: 0, 7: 1, 30: 1, 3: 0, 5: 2, 25: 1, 18: 0,
        }
        renderer.excluded_uids = {30}

        ordered = renderer._ordered_uid_navigation_values()

        self.assertEqual(
            ordered.tolist(),
            [5, 7, 3, 30, 18, 25, 42],
        )
        self.assertEqual(
            renderer._uid_nav_index,
            {5: 0, 7: 1, 3: 2, 30: 3, 18: 4, 25: 5, 42: 6},
        )

        # Once UID 3 receives a path it moves into the first group, where
        # numeric UID order is still preserved.
        renderer._uid_path_counts[3] = 1
        self.assertEqual(
            renderer._ordered_uid_navigation_values().tolist(),
            [3, 5, 7, 30, 18, 25, 42],
        )

    def test_uid_search_opens_first_unlabeled_od_and_reveals_ignored_uid(self):
        traj_df = pd.DataFrame([
            {"uid": 10, "idx_o": 0},
            {"uid": 10, "idx_o": 1},
            {"uid": 20, "idx_o": 0},
            {"uid": 20, "idx_o": 1},
            {"uid": 30, "idx_o": 0},
        ])
        renderer = LabelPath.PathRenderer.__new__(LabelPath.PathRenderer)
        renderer.fig = mock.Mock()
        renderer.traj_df = traj_df
        renderer.state = mock.Mock(uid=10)
        renderer.current_idx = 0
        renderer.annotation_mode = LabelPath.ANNOTATION_MODE
        renderer.labeled_modes = {(20, 0): "TG"}
        renderer.ignored_points = set()
        renderer.excluded_uids = {30}
        renderer._uid_segment_positions = {
            10: np.asarray([0, 1]),
            20: np.asarray([2, 3]),
            30: np.asarray([4]),
        }
        renderer._uid_nav_values = np.asarray([30, 20, 10])
        renderer._uid_nav_index = {30: 0, 20: 1, 10: 2}
        renderer._uid_resolved_counts = {10: 0, 20: 1, 30: 0}
        renderer._uid_path_counts = {10: 0, 20: 0, 30: 0}
        renderer._uid_nav_offset = 0
        renderer._draw_uid_navigation_list = mock.Mock()
        renderer.segment_select_callback = mock.Mock()

        renderer._search_uid("20")

        renderer.segment_select_callback.assert_called_once_with(3)
        renderer._draw_uid_navigation_list.assert_called_with(ensure_current=False)

        renderer.segment_select_callback.reset_mock()
        renderer._search_uid("30")
        renderer.segment_select_callback.assert_not_called()
        self.assertEqual(renderer._uid_nav_index[30], 0)

    def test_uid_search_box_is_labeled_and_blocks_annotation_shortcuts(self):
        renderer = LabelPath.PathRenderer.__new__(LabelPath.PathRenderer)
        renderer.fig, renderer.ax_uid_nav = plt.subplots()
        renderer._uid_search_box = None

        renderer._init_uid_search_box()

        self.assertEqual(renderer._uid_search_box.label.get_text(), "搜索：")
        renderer._uid_search_box.capturekeystrokes = True
        state = mock.Mock(path_history=[])
        controller = LabelPath.LabelController(
            state, renderer, "unused", batch_mode=True, current_idx=0,
            navigate_callback=mock.Mock(),
        )
        with mock.patch.object(controller, "save_mode") as save_mode:
            controller.on_key(mock.Mock(key="1"))
        save_mode.assert_not_called()
        plt.close(renderer.fig)

    def test_uid_navigation_scrolls_and_opens_first_unlabeled_od(self):
        traj_df = pd.DataFrame([
            {"uid": uid, "idx_o": 0} for uid in range(40)
        ])
        renderer = LabelPath.PathRenderer.__new__(LabelPath.PathRenderer)
        renderer.fig, renderer.ax_uid_nav = plt.subplots()
        renderer.traj_df = traj_df
        renderer.state = mock.Mock(uid=0)
        renderer.current_idx = 0
        renderer.labeled_modes = {}
        renderer._uid_segment_positions = {
            uid: np.asarray([uid], dtype=np.int32) for uid in range(40)
        }
        renderer._uid_nav_values = np.arange(40, dtype=np.int64)
        renderer._uid_nav_index = {uid: uid for uid in range(40)}
        renderer._uid_resolved_counts = {uid: 0 for uid in range(40)}
        renderer._uid_nav_offset = 0
        renderer._uid_nav_hit_rows = []
        renderer.segment_select_callback = mock.Mock()
        renderer._draw_uid_navigation_list(ensure_current=True)

        renderer._on_uid_nav_scroll(mock.Mock(
            inaxes=renderer.ax_uid_nav, step=-1, button="down",
        ))
        self.assertEqual(renderer._uid_nav_offset, LabelPath.UID_NAV_SCROLL_STEP)

        y_min, y_max, uid = renderer._uid_nav_hit_rows[0]
        renderer._on_uid_nav_click(mock.Mock(
            inaxes=renderer.ax_uid_nav, ydata=(y_min + y_max) / 2, button=1,
        ))
        renderer.segment_select_callback.assert_called_once_with(uid)
        self.assertEqual(uid, LabelPath.UID_NAV_SCROLL_STEP)

        renderer._toggle_excluded_uid = mock.Mock()
        renderer._on_uid_nav_click(mock.Mock(
            inaxes=renderer.ax_uid_nav, ydata=(y_min + y_max) / 2, button=3,
        ))
        renderer._toggle_excluded_uid.assert_called_once_with(uid)
        plt.close(renderer.fig)

    def test_number_key_saves_mode_directly_and_navigates(self):
        state = mock.Mock(path_history=[(0, 0, 0)])
        renderer = mock.Mock()
        navigate = mock.Mock()
        controller = LabelPath.LabelController(
            state, renderer, "unused", batch_mode=True, current_idx=4,
            navigate_callback=navigate,
        )

        with mock.patch.object(controller, "_finalize") as finalize, \
             mock.patch.object(LabelPath.plt, "close") as close:
            controller.on_key(mock.Mock(key="1"))

        finalize.assert_called_once_with("TG")
        navigate.assert_called_once_with(1)
        close.assert_not_called()

    def test_path_workflow_disables_mode_keys_and_enter_saves_path(self):
        row = pd.Series({"uid": 7, "idx_o": 3})
        state = mock.Mock(
            row=row, uid=7,
            path_history=[(0, 0, 0), (1, -1, 0)], step_count=1,
        )
        renderer = mock.Mock(
            labeled_modes={(7, 3): "TG"}, labeled_paths={},
            ignored_points=set(), excluded_uids=set(),
            active_path_mode="TG",
        )
        navigate = mock.Mock()
        controller = LabelPath.LabelController(
            state, renderer, "unused", batch_mode=True, current_idx=4,
            navigate_callback=navigate,
        )
        controller.annotation_mode = LabelPath.ANNOTATION_PATH
        renderer.annotation_mode = LabelPath.ANNOTATION_PATH

        with mock.patch.object(controller, "_finalize") as finalize_mode, \
             mock.patch.object(
                 controller, "_finalize_path", return_value=(7, 1),
             ) as finalize_path:
            controller.on_key(mock.Mock(key="1"))
            controller.on_key(mock.Mock(key="enter"))

        finalize_mode.assert_not_called()
        finalize_path.assert_called_once_with("TG")
        navigate.assert_not_called()
        state.clear_path.assert_called_once_with()
        renderer.show_segment.assert_called_once_with(state, 4)

    def test_r_resets_selected_saved_path_for_in_place_redraw(self):
        state = mock.Mock(
            uid=7, row=pd.Series({"uid": 7, "idx_o": 3}), path_history=[],
        )
        renderer = mock.Mock(
            labeled_paths={(7, 2): [(0, 0, 0), (1, -1, 0)]},
            selected_path_key=(7, 2), editing_path_key=None,
        )
        controller = LabelPath.LabelController(
            state, renderer, "unused", batch_mode=True, current_idx=4,
        )
        controller.annotation_mode = LabelPath.ANNOTATION_PATH
        renderer.annotation_mode = LabelPath.ANNOTATION_PATH

        controller.on_key(mock.Mock(key="r"))

        self.assertEqual(renderer.editing_path_key, (7, 2))
        state.clear_path.assert_called_once_with()
        renderer.show_segment.assert_called_once_with(state, 4)

    def test_backspace_deletes_selected_path_only_when_no_route_is_pending(self):
        state = mock.Mock(
            uid=7, row=pd.Series({"uid": 7, "idx_o": 3}), path_history=[],
        )
        renderer = mock.Mock(
            labeled_paths={(7, 2): [(0, 0, 0), (1, -1, 0)]},
            selected_path_key=(7, 2), editing_path_key=None,
            export_date="20260811", excluded_uids=set(),
        )
        controller = LabelPath.LabelController(
            state, renderer, "unused", batch_mode=True, current_idx=4,
        )
        controller.annotation_mode = LabelPath.ANNOTATION_PATH
        renderer.annotation_mode = LabelPath.ANNOTATION_PATH

        with mock.patch.object(
            LabelPath, "delete_truth_path_segment", return_value=("paths.csv", 1),
        ) as delete_path, mock.patch.object(
            controller, "_refresh_path_copy",
        ) as refresh_copy:
            controller.on_key(mock.Mock(key="backspace"))

        delete_path.assert_called_once_with("unused", (7, 2))
        renderer.clear_labeled_path.assert_called_once_with((7, 2))
        renderer.renumber_labeled_paths.assert_called_once_with(7)
        refresh_copy.assert_called_once_with()
        renderer.show_segment.assert_called_once_with(state, 4)

    def test_backspace_clears_pending_route_before_touching_selected_path(self):
        state = mock.Mock(
            uid=7, row=pd.Series({"uid": 7, "idx_o": 3}),
            path_history=[(0, 0, 0), (1, -1, 0)],
        )
        renderer = mock.Mock(
            labeled_paths={(7, 2): [(2, -2, 0), (3, -3, 0)]},
            selected_path_key=(7, 2), editing_path_key=None,
        )
        controller = LabelPath.LabelController(
            state, renderer, "unused", batch_mode=True, current_idx=4,
        )
        controller.annotation_mode = LabelPath.ANNOTATION_PATH

        with mock.patch.object(LabelPath, "delete_truth_path_segment") as delete_path:
            controller.on_key(mock.Mock(key="backspace"))

        state.set_path_start.assert_called_once_with((0, 0, 0))
        delete_path.assert_not_called()

    def test_path_uses_active_road_network_and_starts_with_no_forced_path(self):
        row = pd.Series({
            "uid": 7, "idx_o": 3, "order": 7, "mode": "",
            "x_o": 0, "y_o": 0, "z_o": 0,
            "x_d": 2, "y_d": -2, "z_d": 0,
        })
        all_roads = {(9, -9, 0)}
        tg_roads = {(5, -5, 0), (6, -6, 0)}
        state = LabelPath.LabelState(row, all_roads)
        renderer = mock.Mock(
            labeled_modes={(7, 3): "TG"},
            labeled_paths={(7, 1): [(5, -5, 0), (6, -6, 0)]},
            active_path_mode="TG",
            road_sets={"TG": tg_roads},
        )
        controller = LabelPath.LabelController(
            state, renderer, "unused", batch_mode=True, current_idx=4,
        )
        controller.annotation_mode = LabelPath.ANNOTATION_PATH
        controller._prepare_state_for_workflow()

        self.assertIs(state.multi_mapdata, tg_roads)
        self.assertEqual(state.path_history, [])
        self.assertEqual(state.step_count, 0)

    def test_manual_path_start_is_selected_on_map_not_forced_to_signal(self):
        row = pd.Series({
            "uid": 7, "idx_o": 3, "order": 7, "mode": "",
            "x_o": 0, "y_o": 0, "z_o": 0,
            "x_d": 2, "y_d": -2, "z_d": 0,
        })
        road_cells = {(5, -5, 0), (6, -6, 0)}
        state = LabelPath.LabelState(
            row, road_cells,
            hex_grid={(5, -5, 0): {}, (6, -6, 0): {}},
        )
        renderer = mock.Mock(labeled_paths={}, labeled_modes={(7, 3): "TG"})
        controller = LabelPath.LabelController(
            state, renderer, "unused", batch_mode=True, current_idx=4,
        )
        controller.annotation_mode = LabelPath.ANNOTATION_PATH
        state.clear_path()

        self.assertTrue(controller.select_path_start((5, -5, 0)))
        self.assertEqual(state.cur, (5, -5, 0))
        self.assertEqual(state.path_history, [(5, -5, 0)])
        self.assertNotEqual(state.path_history[0], state.start)
        renderer.refresh.assert_called_once_with()

        self.assertTrue(controller.select_path_start((6, -6, 0)))
        self.assertEqual(state.path_history, [(5, -5, 0), (6, -6, 0)])
        controller.on_key(mock.Mock(key="backspace"))
        self.assertEqual(state.path_history, [(5, -5, 0)])

    def test_main_view_left_click_requests_manual_path_start(self):
        renderer = LabelPath.PathRenderer.__new__(LabelPath.PathRenderer)
        renderer.fig, renderer.ax = plt.subplots()
        renderer.annotation_mode = LabelPath.ANNOTATION_PATH
        renderer.path_start_select_callback = mock.Mock()
        event = mock.Mock(inaxes=renderer.ax, button=1, xdata=10.0, ydata=20.0)

        with mock.patch.object(
                renderer, "_hovered_hex_from_event", return_value=(5, -5, 0)):
            renderer._on_main_view_click(event)

        renderer.path_start_select_callback.assert_called_once_with((5, -5, 0))
        plt.close(renderer.fig)

    def test_saved_paths_for_current_uid_are_drawn_without_rescaling(self):
        renderer = LabelPath.PathRenderer.__new__(LabelPath.PathRenderer)
        renderer.fig, renderer.ax = plt.subplots()
        renderer.annotation_mode = LabelPath.ANNOTATION_MODE
        renderer.labeled_paths = {
            (7, 1): [(0, 0, 0), (1, -1, 0)],
            (8, 1): [(2, -2, 0), (3, -3, 0)],
        }
        renderer.labeled_path_modes = {(7, 1): "TG", (8, 1): "GG"}
        state = mock.Mock(uid=7, row=pd.Series({"uid": 7, "idx_o": 0}))
        renderer.ax.set_xlim(10, 20)
        renderer.ax.set_ylim(30, 40)
        before = renderer.ax.get_xlim(), renderer.ax.get_ylim()

        with mock.patch.object(
                LabelPath, "hex_to_mercator",
                return_value=(np.asarray([11.0, 12.0]), np.asarray([31.0, 32.0]))), \
             mock.patch.object(
                 LabelPath, "mercator_wgs84_to_gcj02",
                 side_effect=lambda x, y: (np.asarray(x), np.asarray(y))):
            renderer._draw_saved_paths(state)

        self.assertEqual(len(renderer.ax.collections), 3)
        self.assertEqual((renderer.ax.get_xlim(), renderer.ax.get_ylim()), before)
        plt.close(renderer.fig)

    def test_selected_saved_path_is_emphasized_instead_of_latest(self):
        renderer = LabelPath.PathRenderer.__new__(LabelPath.PathRenderer)
        renderer.fig, renderer.ax = plt.subplots()
        renderer.annotation_mode = LabelPath.ANNOTATION_PATH
        renderer.labeled_paths = {
            (7, 1): [(0, 0, 0), (1, -1, 0)],
            (7, 2): [(1, -1, 0), (2, -2, 0)],
        }
        renderer.labeled_path_modes = {
            (7, 1): "TG", (7, 2): "GG",
        }
        renderer.selected_path_key = (7, 1)
        state = mock.Mock(uid=7)
        renderer.ax.set_xlim(10, 20)
        renderer.ax.set_ylim(30, 40)
        before = renderer.ax.get_xlim(), renderer.ax.get_ylim()

        with mock.patch.object(
                LabelPath, "hex_to_mercator",
                return_value=(np.asarray([11.0, 12.0]), np.asarray([31.0, 32.0]))), \
             mock.patch.object(
                 LabelPath, "mercator_wgs84_to_gcj02",
                 side_effect=lambda x, y: (np.asarray(x), np.asarray(y))):
            renderer._draw_saved_paths(state)

        widths = [
            float(np.max(collection.get_linewidths()))
            for collection in renderer.ax.collections
            if len(collection.get_linewidths())
        ]
        self.assertIn(7.0, widths)
        self.assertIn(4.0, widths)
        selected_collection = next(
            collection for collection in renderer.ax.collections
            if len(collection.get_linewidths())
            and float(np.max(collection.get_linewidths())) == 4.0
        )
        self.assertTrue(np.allclose(
            selected_collection.get_colors()[0, :3],
            LabelPath.MODE_COLORS["TG"],
        ))
        self.assertEqual((renderer.ax.get_xlim(), renderer.ax.get_ylim()), before)
        plt.close(renderer.fig)

    def test_path_save_writes_independent_user_defined_segment_csv(self):
        output_dir = LabelPath.os.path.join(
            LabelPath.os.getcwd(), "tests", "_runtime_path_output",
        )
        mode_csv = LabelPath.os.path.join(output_dir, "traj_labeled.csv")
        path_csv = LabelPath.os.path.join(output_dir, LabelPath.PATH_LABEL_FILENAME)
        try:
            pd.DataFrame([{
                "uid": 7, "idx_o": 3, "idx_d": 4, "mode": "TG",
            }]).to_csv(mode_csv, index=False)
            row = pd.Series({
                "uid": 7, "idx_o": 3, "idx_d": 4, "order": 7, "mode": "",
                "x_o": 0, "y_o": 0, "z_o": 0,
                "x_d": 1, "y_d": -1, "z_d": 0,
            })
            state = LabelPath.LabelState(row, {(0, 0, 0), (1, -1, 0)})
            state.apply_move(0)
            renderer = mock.Mock(
                labeled_modes={(7, 3): "TG"}, labeled_paths={},
                labeled_path_modes={}, export_date=None, excluded_uids=set(),
            )
            controller = LabelPath.LabelController(
                state, renderer, output_dir, batch_mode=False, current_idx=0,
            )

            self.assertEqual(controller._finalize_path("TG"), (7, 1))
            mode_saved = pd.read_csv(mode_csv, keep_default_na=False)
            self.assertNotIn("traj", mode_saved.columns)
            path_saved = pd.read_csv(path_csv, keep_default_na=False).iloc[0]
            self.assertEqual(path_saved["mode"], "TG")
            self.assertEqual(int(path_saved["anchor_idx_o"]), 3)
            self.assertEqual(
                (int(path_saved["start_x"]), int(path_saved["start_y"]),
                 int(path_saved["start_z"])),
                (0, 0, 0),
            )
            self.assertEqual(
                (int(path_saved["end_x"]), int(path_saved["end_y"]),
                 int(path_saved["end_z"])),
                (1, -1, 0),
            )
            self.assertEqual(
                json.loads(path_saved["traj"]), [[0, 0, 0], [1, -1, 0]],
            )
            self.assertEqual(int(path_saved["steps"]), 1)
        finally:
            for csv_path in (mode_csv, path_csv):
                for suffix in ("", ".tmp"):
                    target = csv_path + suffix
                    if LabelPath.os.path.exists(target):
                        LabelPath.os.remove(target)

    def test_path_redraw_updates_same_segment_id_through_controller(self):
        output_dir = LabelPath.os.path.join(
            LabelPath.os.getcwd(), "tests", "_runtime_path_output",
        )
        path_csv = LabelPath.os.path.join(output_dir, LabelPath.PATH_LABEL_FILENAME)
        try:
            _, segment_key = LabelPath.write_truth_path_segment(
                output_dir, 7, "TG", 3,
                [(0, 0, 0), (1, -1, 0)], 1.0, 1,
            )
            row = pd.Series({
                "uid": 7, "idx_o": 8, "idx_d": 9, "order": 7, "mode": "",
                "x_o": 4, "y_o": -4, "z_o": 0,
                "x_d": 6, "y_d": -6, "z_d": 0,
            })
            state = LabelPath.LabelState(
                row, {(4, -4, 0), (5, -5, 0), (6, -6, 0)},
            )
            self.assertTrue(state.set_path_start((4, -4, 0)))
            self.assertTrue(state.restore_path([
                (4, -4, 0), (5, -5, 0), (6, -6, 0),
            ]))
            renderer = mock.Mock(
                labeled_modes={(7, 8): "GG"},
                labeled_paths={segment_key: [(0, 0, 0), (1, -1, 0)]},
                labeled_path_modes={segment_key: "TG"},
                editing_path_key=segment_key,
                export_date=None, excluded_uids=set(),
            )
            controller = LabelPath.LabelController(
                state, renderer, output_dir, batch_mode=False, current_idx=0,
            )

            self.assertEqual(controller._finalize_path("GG"), segment_key)

            saved = pd.read_csv(path_csv)
            self.assertEqual(len(saved), 1)
            self.assertEqual(int(saved.iloc[0]["segment_id"]), segment_key[1])
            self.assertEqual(saved.iloc[0]["mode"], "GG")
            self.assertEqual(int(saved.iloc[0]["anchor_idx_o"]), 8)
            self.assertEqual(
                json.loads(saved.iloc[0]["traj"]),
                [[4, -4, 0], [5, -5, 0], [6, -6, 0]],
            )
            self.assertIsNone(renderer.editing_path_key)
        finally:
            for suffix in ("", ".tmp"):
                target = path_csv + suffix
                if LabelPath.os.path.exists(target):
                    LabelPath.os.remove(target)

    def test_od_mode_change_does_not_rewrite_independent_path_truth(self):
        output_dir = LabelPath.os.path.join(
            LabelPath.os.getcwd(), "tests", "_runtime_path_output",
        )
        mode_csv = LabelPath.os.path.join(output_dir, "traj_labeled.csv")
        path_csv = LabelPath.os.path.join(output_dir, LabelPath.PATH_LABEL_FILENAME)
        try:
            traj_df = pd.DataFrame([
                {"uid": 7, "idx_o": 3, "idx_d": 4, "order": 7, "mode": "",
                 "x_o": 0, "y_o": 0, "z_o": 0,
                 "x_d": 1, "y_d": -1, "z_d": 0},
                {"uid": 7, "idx_o": 4, "idx_d": 5, "order": 7, "mode": "",
                 "x_o": 1, "y_o": -1, "z_o": 0,
                 "x_d": 2, "y_d": -2, "z_d": 0},
            ])
            pd.DataFrame([
                {"uid": 7, "idx_o": 3, "idx_d": 4, "mode": "TG"},
                {"uid": 7, "idx_o": 4, "idx_d": 5, "mode": "TG"},
            ]).to_csv(mode_csv, index=False)
            renderer = LabelPath.PathRenderer.__new__(LabelPath.PathRenderer)
            renderer.traj_df = traj_df
            renderer.output_dir = output_dir
            renderer.labeled_modes = {(7, 3): "TG", (7, 4): "TG"}
            renderer.labeled_paths = {}
            renderer.labeled_path_modes = {}
            renderer.ignored_points = set()
            renderer.excluded_uids = set()
            renderer.export_date = None
            renderer._uid_resolved_counts = {7: 2}
            renderer.refresh_uid_segment_list = mock.Mock()
            LabelPath.write_truth_path_segment(
                output_dir, 7, "TG", 3,
                [(0, 0, 0), (1, -1, 0), (2, -2, 0)], 1.0, 2,
            )
            renderer.labeled_paths, renderer.labeled_path_modes = \
                LabelPath.load_labeled_path_data(output_dir)
            state = LabelPath.LabelState(traj_df.iloc[1], set())
            controller = LabelPath.LabelController(
                state, renderer, output_dir, batch_mode=False, current_idx=1,
            )

            self.assertTrue(controller._finalize("TG"))
            self.assertEqual(len(pd.read_csv(path_csv)), 1)

            self.assertTrue(controller._finalize("GG"))
            changed_modes = pd.read_csv(mode_csv).sort_values("idx_o")
            self.assertEqual(changed_modes["mode"].tolist(), ["TG", "GG"])
            saved_path = pd.read_csv(path_csv)
            self.assertEqual(len(saved_path), 1)
            self.assertEqual(saved_path.iloc[0]["mode"], "TG")
            self.assertEqual(
                json.loads(saved_path.iloc[0]["traj"]),
                [[0, 0, 0], [1, -1, 0], [2, -2, 0]],
            )
        finally:
            for csv_path in (mode_csv, path_csv):
                for suffix in ("", ".tmp"):
                    target = csv_path + suffix
                    if LabelPath.os.path.exists(target):
                        LabelPath.os.remove(target)

    def test_ignored_current_point_cannot_be_labeled(self):
        state = mock.Mock(
            row=pd.Series({"uid": 7, "idx_o": 3}),
            path_history=[(0, 0, 0)],
        )
        renderer = mock.Mock(ignored_points={(7, 3)})
        controller = LabelPath.LabelController(
            state, renderer, "unused", batch_mode=True, current_idx=4,
            navigate_callback=mock.Mock(),
        )

        with mock.patch.object(controller, "_finalize") as finalize:
            controller.save_mode("TG")

        finalize.assert_not_called()
        controller.navigate_callback.assert_not_called()

    def test_excluded_current_uid_cannot_be_labeled(self):
        state = mock.Mock(
            uid=7,
            row=pd.Series({"uid": 7, "idx_o": 3}),
            path_history=[(0, 0, 0)],
        )
        renderer = mock.Mock(ignored_points=set(), excluded_uids={7})
        controller = LabelPath.LabelController(
            state, renderer, "unused", batch_mode=True, current_idx=4,
            navigate_callback=mock.Mock(),
        )

        with mock.patch.object(controller, "_finalize") as finalize:
            controller.save_mode("TG")

        finalize.assert_not_called()
        controller.navigate_callback.assert_not_called()

    def test_enter_no_longer_starts_or_saves_a_label(self):
        state = mock.Mock(path_history=[(0, 0, 0)])
        renderer = mock.Mock()
        controller = LabelPath.LabelController(
            state, renderer, "unused", batch_mode=True, current_idx=4,
            navigate_callback=mock.Mock(),
        )

        with mock.patch.object(controller, "_finalize") as finalize:
            controller.on_key(mock.Mock(key="enter"))

        finalize.assert_not_called()
        controller.navigate_callback.assert_not_called()

    def test_batch_controller_goes_back_in_same_window(self):
        state = mock.Mock(path_history=[(0, 0, 0)])
        renderer = mock.Mock()
        navigate = mock.Mock()
        controller = LabelPath.LabelController(
            state, renderer, "unused", batch_mode=True, current_idx=4,
            navigate_callback=navigate,
        )

        with mock.patch.object(LabelPath.plt, "close") as close:
            controller.on_key(mock.Mock(key="backspace"))

        navigate.assert_called_once_with(-1)
        close.assert_not_called()


if __name__ == "__main__":
    unittest.main()
