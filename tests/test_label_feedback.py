import unittest
from unittest import mock

import matplotlib.pyplot as plt
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
        self.assertTrue(
            (collections[1].get_facecolors()[0, :3] == LabelPath.MODE_COLORS["TS"]).all()
        )
        self.assertTrue(
            (collections[2].get_facecolors()[0, :3] == LabelPath.MODE_COLORS["GG"]).all()
        )
        plt.close(renderer.fig)

    def test_segment_info_shows_uid_od_progress(self):
        renderer = LabelPath.PathRenderer.__new__(LabelPath.PathRenderer)
        renderer.fig, renderer.ax = plt.subplots()
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

    def test_batch_controller_navigates_without_closing_window(self):
        state = mock.Mock(path_history=[(0, 0, 0)])
        renderer = mock.Mock()
        navigate = mock.Mock()
        controller = LabelPath.LabelController(
            state, renderer, "unused", batch_mode=True, current_idx=4,
            navigate_callback=navigate,
        )
        controller.selecting_label = True

        with mock.patch.object(controller, "_finalize"), \
             mock.patch.object(LabelPath.plt, "close") as close:
            controller.on_key(mock.Mock(key="1"))

        navigate.assert_called_once_with(1, False)
        close.assert_not_called()

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

        navigate.assert_called_once_with(-1, True)
        close.assert_not_called()


if __name__ == "__main__":
    unittest.main()
