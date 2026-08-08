import unittest
from unittest import mock

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

    def test_normalize_storage_sorts_active_records(self):
        output_dir = LabelPath.os.path.join(
            LabelPath.os.getcwd(), "tests", "_runtime_ignore_output",
        )
        active_path = LabelPath.os.path.join(output_dir, "traj_labeled.csv")
        try:
            pd.DataFrame([
                {"uid": 7, "idx_o": 1, "idx_d": 2, "mode": "TS"},
                {"uid": 7, "idx_o": 0, "idx_d": 1, "mode": "TG"},
                {"uid": 6, "idx_o": 9, "idx_d": 10, "mode": "GG"},
            ]).to_csv(active_path, index=False)

            reordered = LabelPath.normalize_label_storage(output_dir)

            self.assertEqual(reordered, 1)
            active = pd.read_csv(active_path)
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
        plt.close(renderer.fig)

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

    def test_road_buttons_toggle_layer_without_rescaling_and_persist_on_redraw(self):
        renderer = LabelPath.PathRenderer.__new__(LabelPath.PathRenderer)
        renderer.fig, renderer.ax = plt.subplots()
        renderer.ax_info = renderer.ax
        renderer.road_visibility = {mode: True for mode in LabelPath.MODE_LIST}
        renderer._road_artists = {}
        renderer._road_buttons = {}
        renderer._init_road_toggle_buttons()

        button_positions = [
            renderer._road_buttons[mode].ax.get_position() for mode in LabelPath.MODE_LIST
        ]
        self.assertEqual(len({round(position.x0, 6) for position in button_positions}), 1)
        self.assertEqual(len({round(position.y0, 6) for position in button_positions}), 5)

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

        self.assertFalse(renderer._road_artists["TG"].get_visible())
        self.assertEqual((renderer.ax.get_xlim(), renderer.ax.get_ylim()), before)
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

        y_min, y_max, target = renderer._uid_list_hit_rows[0]
        renderer._on_uid_list_click(mock.Mock(
            inaxes=renderer.ax_uid_list, ydata=(y_min + y_max) / 2, button=1,
        ))
        renderer.segment_select_callback.assert_called_once_with(target)
        self.assertEqual(target, LabelPath.UID_LIST_SCROLL_STEP)

        renderer._toggle_ignored_point = mock.Mock()
        renderer._on_uid_list_click(mock.Mock(
            inaxes=renderer.ax_uid_list, ydata=(y_min + y_max) / 2, button=3,
        ))
        renderer._toggle_ignored_point.assert_called_once_with(target)
        plt.close(renderer.fig)

    def test_uid_navigation_uses_solid_dot_only_for_completed_uid(self):
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
        renderer.excluded_uids = {30}
        renderer._uid_nav_offset = 0
        renderer._uid_nav_hit_rows = []

        renderer._draw_uid_navigation_list(ensure_current=True)

        facecolors = renderer.ax_uid_nav.collections[0].get_facecolors()
        self.assertEqual(facecolors[0, 3], 1.0)
        self.assertEqual(facecolors[1, 3], 0.0)
        self.assertEqual(len(facecolors), 2)
        self.assertEqual(len(renderer.ax_uid_nav.collections[1].get_offsets()), 1)
        self.assertTrue(any(patch.get_edgecolor()[2] > 0.7
                            for patch in renderer.ax_uid_nav.patches))
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
