"""
Interactive path labeling tool — manually walk from start to end using arrow keys,
recording the labeled path step by step.

Usage:
  python LabelPath.py                     # start from index 0, label one by one
  python LabelPath.py --index 5           # label a single trajectory at index 5
  python LabelPath.py --batch             # label all trajectories in sequence
  python LabelPath.py --output my_labels  # custom output directory
"""

import os
import sys
import json
import pickle
import argparse

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 禁用 matplotlib 默认快捷键，避免与标注操作冲突
matplotlib.rcParams["keymap.quit"] = []
matplotlib.rcParams["keymap.quit_all"] = []
matplotlib.rcParams["keymap.save"] = []
matplotlib.rcParams["keymap.fullscreen"] = []
matplotlib.rcParams["keymap.home"] = []
matplotlib.rcParams["keymap.back"] = []
matplotlib.rcParams["keymap.forward"] = []
matplotlib.rcParams["keymap.pan"] = []
matplotlib.rcParams["keymap.zoom"] = []
matplotlib.rcParams["keymap.grid"] = []
matplotlib.rcParams["keymap.xscale"] = []
matplotlib.rcParams["keymap.yscale"] = []
# ---------------------------------------------------------------------------
from utils.geo_utils import (
    grid_to_mercator,
    grid_bounds_to_mercator,
    full_grid_bounds_mercator,
    mercator_to_grid,
    wgs84_to_hex,
    hex_to_wgs84,
    hex_to_mercator,
    hex_neighbors,
    hex_distance,
    hex_in_map,
    _init_hex_origin,
    mercator_wgs84_to_gcj02,
)
from utils.basemap import add_basemap, USE_BASEMAP

from utils.tools import (
    mapdata_to_modelmatrix,
    calculate_match_rate,
    HEX_MODE_BITS,
    hex_mapdata_to_road_sets,
    build_multi_mapdata_hex,
    calculate_match_rate_hex,
)

# ========================== Constants ==========================

# ========================== Quad Grid Constants ==========================

DX_DY = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
#         right     up        left       down

MAP_ROWS = 529
MAP_COLS = 564

# ========================== Hex Grid Constants ==========================

# 六方向向量（平顶六边形立方体坐标）
HEX_DIRS = {
    0: (1, -1, 0),   # E  / 右
    1: (1, 0, -1),   # NE / 东北
    2: (0, 1, -1),   # NW / 西北/上
    3: (-1, 1, 0),   # W  / 左
    4: (-1, 0, 1),   # SW / 西南
    5: (0, -1, 1),   # SE / 东南/下
}

VIEW_PADDING_METERS = 10000  # hex 模式视口边距（Mercator 米）
HEX_PKL_PATH = "data\hex_grid.pkl"

# ========================== Shared Constants ==========================

MODE_COLORS = {
    "TG": "purple",
    "GG": "blue",
    "GSD": "green",
    "TS": "red",
    "XD": "cyan",
}
MODE_LIST = ["GSD", "GG", "TS", "TG", "XD"]

DEFAULT_MAPDATA_PATH = "data/GridModesAdjacentRealworld.pkl"
DEFAULT_CSV_PATH = "data\dataset_20230917_nanjing_to_gaochun_lishui_with_hex_downsampled.csv"
DEFAULT_OUTPUT_DIR = "label_output"

DISTANCE_THRESHOLD = 1.0
VIEW_PADDING = 30

# Label options after saving (press 1-6 to select)
LABEL_OPTIONS = {
    "1": "GSD",
    "2": "GG",
    "3": "TS",
    "4": "TG",
    "5": "Mixed",
    "6": "Other",
}

# ========================== Quad Key Bindings ==========================
KEY_MAP = {
    "right": ("move", 0),
    "up":    ("move", 1),
    "left":  ("move", 2),
    "down":  ("move", 3),
    "d": ("move", 0),
    "w": ("move", 1),
    "a": ("move", 2),
    "s": ("move", 3),
    "backspace": ("undo", None),
    "ctrl+z":    ("undo", None),
    "r":         ("reset", None),
    "enter":     ("save", None),
}

# ========================== Hex Key Bindings ==========================
HEX_KEY_MAP = {
    "d": ("move", 1),  "right": ("move", 1),      # 右上
    "e": ("move", 0),                              # 右下
    "w": ("move", 5),  "up":    ("move", 5),       # 下
    "q": ("move", 4),                              # 左下
    "a": ("move", 3),  "left":  ("move", 3),       # 左上
    "s": ("move", 2),  "down":  ("move", 2),       # 上
    "backspace": ("undo", None),
    "ctrl+z":    ("undo", None),
    "r":         ("reset", None),
    "enter":     ("save", None),
}

# ========================== Data Loading ==========================

def load_mapdata(path=DEFAULT_MAPDATA_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_traj_csv(path=DEFAULT_CSV_PATH):
    return pd.read_csv(path)


def build_multi_mapdata(raw_mapdata, selected_mode):
    """Build multi_mapdata for the given mode(s), matching PathEnv.
    Returns ndarray of shape (564, 529).
    """
    mode_matrices = mapdata_to_modelmatrix(raw_mapdata, MAP_ROWS, MAP_COLS)
    if isinstance(selected_mode, str):
        selected_mode = [selected_mode]
    multi = np.zeros((MAP_COLS, MAP_ROWS), dtype=np.int32)
    for m in selected_mode:
        if m in mode_matrices:
            multi += np.array(mode_matrices[m], dtype=np.int32)
    return np.clip(multi, 0, 1)


# ========================== Hex Data Loading ==========================

def load_hex_mapdata(path=HEX_PKL_PATH):
    """加载六边形网格 pkl，触发原点初始化"""
    _init_hex_origin(path)
    with open(path, "rb") as f:
        return pickle.load(f)


# 原始点级数据缓存，供速度分布图使用
_RAW_POINT_DF = None


def get_raw_point_df():
    return _RAW_POINT_DF


def load_traj_csv_hex(path, sample_step=10):
    """读取六边形模式轨迹 CSV，按 uid 分组生成起终点记录。

    支持两种格式:
    1. 起终点格式: x_o,y_o,z_o,x_d,y_d,z_d,mode,order
    2. 点序列格式: uid,hex_x,hex_y,hex_z,...  → 按 uid 分组，每 sample_step 行采样生成段
    """
    global _RAW_POINT_DF
    df = pd.read_csv(path)
    # 检测格式: 有 hex_x/hex_y/hex_z 列 → 点序列格式
    if "hex_x" in df.columns and "uid" in df.columns:
        _RAW_POINT_DF = df  # 缓存原始数据
        records = []
        for uid, group in df.groupby("uid", sort=False):
            group = group.reset_index(drop=True)
            sampled_indices = list(range(0, len(group), sample_step))
            for i in range(len(sampled_indices) - 1):
                a = group.iloc[sampled_indices[i]]
                b = group.iloc[sampled_indices[i + 1]]
                # 间隔取行：对区间内的 time / dist 求和，velocity 重算
                seg_rows = group.iloc[sampled_indices[i] + 1 : sampled_indices[i + 1] + 1]
                time_sum = float(seg_rows["time_value"].sum()) if "time_value" in df.columns else 0.0
                dist_sum = float(seg_rows["dist_value"].sum()) if "dist_value" in df.columns else 0.0
                if time_sum > 0 and dist_sum > 0:
                    velocity_d = dist_sum / time_sum * 3.6  # m/s → km/h
                elif "velocity" in df.columns:
                    velocity_d = float(b["velocity"])
                else:
                    velocity_d = 0.0

                records.append({
                    "x_o": int(a["hex_x"]), "y_o": int(a["hex_y"]), "z_o": int(a["hex_z"]),
                    "x_d": int(b["hex_x"]), "y_d": int(b["hex_y"]), "z_d": int(b["hex_z"]),
                    "mode": str(a["attribution"]) if "attribution" in df.columns else "ALL",
                    "order": int(uid),
                    "uid": int(uid),
                    "idx_o": int(a["idx"]) if "idx" in df.columns else int(a.name),
                    "idx_d": int(b["idx"]) if "idx" in df.columns else int(b.name),
                    "stime_o": int(a["stime"]) if "stime" in df.columns else 0,
                    "lat_o": float(a["lat"]) if "lat" in df.columns else 0.0,
                    "lon_o": float(a["lon"]) if "lon" in df.columns else 0.0,
                    "time_d": time_sum,
                    "dist_d": dist_sum,
                    "velocity_d": velocity_d,
                })
        return pd.DataFrame(records)
    return df


# ========================== State ==========================

class LabelState:
    """Holds labeling state for a single trajectory (quad or hex)."""

    def __init__(self, row, multi_mapdata, is_hex=False, hex_grid=None):
        self.row = row
        self.order = int(row["order"]) if "order" in row.index else 0
        self.mode = str(row["mode"]).strip()
        self.uid = int(row["uid"]) if "uid" in row.index else self.order
        self.is_hex = is_hex
        self.hex_grid = hex_grid

        if is_hex:
            self.start = (int(row["x_o"]), int(row["y_o"]), int(row["z_o"]))
            self.end = (int(row["x_d"]), int(row["y_d"]), int(row["z_d"]))
        else:
            self.start_x = float(row["locx_o"])
            self.start_y = float(row["locy_o"])
            self.end_x = float(row["locx_d"])
            self.end_y = float(row["locy_d"])

        self.multi_mapdata = multi_mapdata

        if is_hex:
            self.cur = self.start
            self.path_history = [self.cur]
        else:
            self.cur_x = int(round(self.start_x))
            self.cur_y = int(round(self.start_y))
            self.path_history = [(self.cur_x, self.cur_y)]
        self.step_count = 0

    @property
    def reached(self):
        if self.is_hex:
            return hex_distance(self.cur, self.end) <= DISTANCE_THRESHOLD
        dist = abs(self.cur_x - self.end_x) + abs(self.cur_y - self.end_y)
        return dist <= DISTANCE_THRESHOLD

    @property
    def remaining_dist(self):
        if self.is_hex:
            return hex_distance(self.cur, self.end)
        return abs(self.cur_x - self.end_x) + abs(self.cur_y - self.end_y)

    def current_match_rate(self):
        if len(self.path_history) <= 1:
            return 0.0
        if self.is_hex:
            return calculate_match_rate_hex(self.path_history, self.multi_mapdata)
        return calculate_match_rate(self.path_history, self.multi_mapdata)

    def can_move(self, *args):
        if self.is_hex:
            dx, dy, dz = args
            nx, ny, nz = self.cur[0] + dx, self.cur[1] + dy, self.cur[2] + dz
            return hex_in_map(nx, ny, nz, self.hex_grid)
        dx, dy = args
        nx = self.cur_x + dx
        ny = self.cur_y + dy
        return 0 <= nx < MAP_COLS and 0 <= ny < MAP_ROWS

    def apply_move(self, action_id):
        if self.is_hex:
            dx, dy, dz = HEX_DIRS[action_id]
            self.cur = (self.cur[0] + dx, self.cur[1] + dy, self.cur[2] + dz)
            self.path_history.append(self.cur)
        else:
            dx, dy = DX_DY[action_id]
            self.cur_x += dx
            self.cur_y += dy
            self.path_history.append((self.cur_x, self.cur_y))
        self.step_count += 1

    def undo(self):
        if len(self.path_history) > 1:
            self.path_history.pop()
            if self.is_hex:
                self.cur = self.path_history[-1]
            else:
                self.cur_x, self.cur_y = self.path_history[-1]
            self.step_count = max(0, self.step_count - 1)
            return True
        return False

    def reset(self):
        if self.is_hex:
            self.cur = self.start
            self.path_history = [self.cur]
        else:
            self.cur_x = int(round(self.start_x))
            self.cur_y = int(round(self.start_y))
            self.path_history = [(self.cur_x, self.cur_y)]
        self.step_count = 0


# ========================== Renderer ==========================

class PathRenderer:
    """Manages the matplotlib figure and incremental updates (quad or hex)."""

    def __init__(self, state: LabelState, raw_mapdata, road_sets=None,
                 traj_df=None, current_idx=None, output_dir=None):
        self.state = state
        self.is_hex = state.is_hex
        self.road_sets = road_sets
        self.traj_df = traj_df
        self.current_idx = current_idx
        self.output_dir = output_dir

        if self.is_hex:
            # 左右分栏：左侧地图，右侧速度分布
            self.fig = plt.figure(figsize=(16, 9))
            gs = self.fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.02)
            self.ax = self.fig.add_subplot(gs[0])
            self.ax_hist = self.fig.add_subplot(gs[1])
            title = "LabelPath — Hex Grid"
        else:
            self.fig, self.ax = plt.subplots(figsize=(12, 9))
            self.ax_hist = None
            title = "LabelPath — Interactive Path Labeling"
        self.fig.canvas.manager.set_window_title(title)

        if self.is_hex:
            self._init_hex_view(state, raw_mapdata)
        else:
            self._init_quad_view(state, raw_mapdata)

        self.ax.set_xlabel("Web Mercator X (EPSG:3857)")
        self.ax.set_ylabel("Web Mercator Y (EPSG:3857)")
        self.ax.grid(False)

        # ---- 出行模式颜色图例 ----
        mode_handles = [
            Line2D([0], [0], color="purple", lw=3, label="TG"),
            Line2D([0], [0], color="blue", lw=3, label="GG"),
            Line2D([0], [0], color="green", lw=3, label="GSD"),
            Line2D([0], [0], color="red", lw=3, label="TS"),
        ]
        self.ax.legend(
            handles=mode_handles, loc="lower right",
            fontsize=7, handlelength=1.5, borderpad=0.4, labelspacing=0.3,
        )

        self._update_title()
        self._draw_legend_box()
        self._draw_segment_info()
        self.fig.tight_layout()

    # ======================== Quad View Init ========================

    def _init_quad_view(self, state, raw_mapdata):
        x_vals = [state.start_x, state.end_x]
        y_vals = [state.start_y, state.end_y]
        x_center = (min(x_vals) + max(x_vals)) / 2
        y_center = (min(y_vals) + max(y_vals)) / 2
        x_range = max(abs(max(x_vals) - min(x_vals)), 20)
        y_range = max(abs(max(y_vals) - min(y_vals)), 20)

        grid_xmin = max(0, x_center - x_range / 2 - VIEW_PADDING)
        grid_xmax = min(MAP_COLS, x_center + x_range / 2 + VIEW_PADDING)
        grid_ymin = max(0, y_center - y_range / 2 - VIEW_PADDING)
        grid_ymax = min(MAP_ROWS, y_center + y_range / 2 + VIEW_PADDING)

        mx_min, mx_max, my_min, my_max = grid_bounds_to_mercator(
            grid_xmin, grid_xmax, grid_ymin, grid_ymax
        )
        # GCJ-02 偏移 → 高德瓦片对齐
        gcj_x, gcj_y = mercator_wgs84_to_gcj02(
            [mx_min, mx_max, mx_min, mx_max],
            [my_min, my_max, my_max, my_min],
        )
        self.ax.set_xlim(gcj_x.min(), gcj_x.max())
        self.ax.set_ylim(gcj_y.min(), gcj_y.max())
        self.ax.set_aspect("equal")

        if USE_BASEMAP:
            try:
                add_basemap(self.ax, alpha=0.8)
            except Exception as e:
                print(f"  [WARN] basemap load failed: {e}")

        road_rgb = self._build_road_rgb(raw_mapdata)
        full_mx_min, full_mx_max, full_my_min, full_my_max = full_grid_bounds_mercator()
        gcj_fx, gcj_fy = mercator_wgs84_to_gcj02(
            [full_mx_min, full_mx_max, full_mx_min, full_mx_max],
            [full_my_min, full_my_max, full_my_max, full_my_min],
        )
        self.ax.imshow(
            road_rgb,
            extent=[gcj_fx.min(), gcj_fx.max(), gcj_fy.min(), gcj_fy.max()],
            origin="lower",
            alpha=0.45,
            zorder=2,
        )

        start_mx, start_my = grid_to_mercator(state.start_x, state.start_y)
        start_mx, start_my = mercator_wgs84_to_gcj02(start_mx, start_my)
        end_mx, end_my = grid_to_mercator(state.end_x, state.end_y)
        end_mx, end_my = mercator_wgs84_to_gcj02(end_mx, end_my)

        self.start_handle = self.ax.scatter(
            start_mx, start_my,
            c="limegreen", marker="o", s=60,
            edgecolors="darkgreen", linewidths=1.2, zorder=5, label="Start",
        )
        self.end_handle = self.ax.scatter(
            end_mx, end_my,
            c="red", marker="X", s=60,
            edgecolors="darkred", linewidths=1.2, zorder=5, label="End",
        )

        (self.path_line,) = self.ax.plot(
            [], [], "-",
            color="crimson", linewidth=2.5, alpha=0.85, zorder=3, label="Path",
        )

        cursor_mx, cursor_my = grid_to_mercator(state.cur_x, state.cur_y)
        cursor_mx, cursor_my = mercator_wgs84_to_gcj02(cursor_mx, cursor_my)
        self.cursor = self.ax.scatter(
            cursor_mx, cursor_my,
            c="cyan", marker="o", s=25,
            edgecolors="darkblue", linewidths=1.5, zorder=6, label="Cursor",
        )

    # ======================== Hex View Init ========================

    def _init_hex_view(self, state, raw_mapdata):
        # 从起止点 WGS84 坐标计算视口（Mercator）
        start_lon, start_lat = hex_to_wgs84(*state.start)
        end_lon, end_lat = hex_to_wgs84(*state.end)

        from utils.geo_utils import _wgs84_to_merc
        start_mx, start_my = _wgs84_to_merc.transform(start_lon, start_lat)
        end_mx, end_my = _wgs84_to_merc.transform(end_lon, end_lat)

        x_center = (start_mx + end_mx) / 2
        y_center = (start_my + end_my) / 2
        x_range = max(abs(end_mx - start_mx), 12000)
        y_range = max(abs(end_my - start_my), 12000)

        pad = VIEW_PADDING_METERS
        # 保存原始 WGS84 Mercator 范围（供道路叠加层过滤使用）
        self._mx_min = x_center - x_range / 2 - pad
        self._mx_max = x_center + x_range / 2 + pad
        self._my_min = y_center - y_range / 2 - pad
        self._my_max = y_center + y_range / 2 + pad

        # GCJ-02 偏移 → 高德瓦片对齐
        gcj_x = [self._mx_min, self._mx_max, self._mx_min, self._mx_max]
        gcj_y = [self._my_min, self._my_max, self._my_max, self._my_min]
        gcj_mx, gcj_my = mercator_wgs84_to_gcj02(gcj_x, gcj_y)
        self.ax.set_xlim(gcj_mx.min(), gcj_mx.max())
        self.ax.set_ylim(gcj_my.min(), gcj_my.max())
        self.ax.set_aspect("equal")

        if USE_BASEMAP:
            try:
                add_basemap(self.ax, alpha=0.8)
            except Exception as e:
                print(f"  [WARN] basemap load failed: {e}")

        # 道路叠加层：每个模式用散点图
        if self.road_sets is not None:
            self._build_hex_road_overlay(raw_mapdata)

        # 起终点标记（GCJ-02 偏移）
        start_mx, start_my = mercator_wgs84_to_gcj02(start_mx, start_my)
        end_mx, end_my = mercator_wgs84_to_gcj02(end_mx, end_my)
        self.start_handle = self.ax.scatter(
            start_mx, start_my,
            c="limegreen", marker="o", s=20,
            edgecolors="darkgreen", linewidths=1.2, zorder=5, label="Start",
        )
        self.end_handle = self.ax.scatter(
            end_mx, end_my,
            c="red", marker="X", s=20,
            edgecolors="darkred", linewidths=1.2, zorder=5, label="End",
        )

        (self.path_line,) = self.ax.plot(
            [], [], "-",
            color="crimson", linewidth=2.5, alpha=0.85, zorder=3, label="Path",
        )

        cursor_mx, cursor_my = hex_to_mercator(*state.cur)
        cursor_mx, cursor_my = mercator_wgs84_to_gcj02(cursor_mx, cursor_my)
        self.cursor = self.ax.scatter(
            cursor_mx, cursor_my,
            c="cyan", marker="o", s=20,
            edgecolors="darkblue", linewidths=1.5, zorder=6, label="Cursor",
        )

        # ---- 上下文参考点（前一段起点 / 后一段终点）----
        self._draw_context_points(state)

        # ---- 右侧：速度分布直方图 ----
        self._draw_velocity_hist(state)

    def _draw_context_points(self, state):
        """绘制同一 uid 相邻段的参考点及已标注路径"""
        if self.traj_df is None or self.current_idx is None:
            return

        idx = self.current_idx
        traj_df = self.traj_df

        # 前一段的起点（前一个点）
        if idx > 0:
            prev_row = traj_df.iloc[idx - 1]
            if int(prev_row.get("uid", -1)) == state.uid:
                if state.is_hex:
                    pt = (int(prev_row["x_o"]), int(prev_row["y_o"]), int(prev_row["z_o"]))
                    mx, my = hex_to_mercator(*pt)
                else:
                    mx, my = grid_to_mercator(float(prev_row["locx_o"]), float(prev_row["locy_o"]))
                mx, my = mercator_wgs84_to_gcj02(mx, my)
                self.ax.scatter(
                    mx, my,
                    c="orange", marker="D", s=20,
                    edgecolors="darkorange", linewidths=1, zorder=4,
                )
                self._draw_labeled_path(state, prev_row)

        # 后一段的终点（后一个点）
        if idx < len(traj_df) - 1:
            next_row = traj_df.iloc[idx + 1]
            if int(next_row.get("uid", -1)) == state.uid:
                if state.is_hex:
                    pt = (int(next_row["x_d"]), int(next_row["y_d"]), int(next_row["z_d"]))
                    mx, my = hex_to_mercator(*pt)
                else:
                    mx, my = grid_to_mercator(float(next_row["locx_d"]), float(next_row["locy_d"]))
                mx, my = mercator_wgs84_to_gcj02(mx, my)
                self.ax.scatter(
                    mx, my,
                    c="deepskyblue", marker="D", s=20,
                    edgecolors="blue", linewidths=1, zorder=4,
                )
                self._draw_labeled_path(state, next_row)

    def _draw_labeled_path(self, state, adj_row):
        """如果相邻段已被标注，将其路径画在图上"""
        uid_val = int(adj_row.get("uid", -1))
        idx_o_val = adj_row.get("idx_o", None)
        if idx_o_val is None:
            return
        labeled_csv = os.path.join(self.output_dir, "traj_labeled.csv")
        if not os.path.exists(labeled_csv):
            return
        try:
            labeled_df = pd.read_csv(labeled_csv, encoding="utf-8")
        except Exception:
            return
        if labeled_df.empty:
            return
        match = labeled_df[
            (labeled_df["uid"] == uid_val) & (labeled_df["idx_o"] == int(idx_o_val))
        ]
        if match.empty:
            return
        traj_str = match.iloc[0].get("traj", "")
        if not traj_str or not isinstance(traj_str, str):
            return
        try:
            pts = json.loads(traj_str)
        except (json.JSONDecodeError, TypeError):
            return
        if len(pts) < 2:
            return
        if state.is_hex:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            zs = [p[2] for p in pts]
            merc_x, merc_y = hex_to_mercator(xs, ys, zs)
        else:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            merc_x, merc_y = grid_to_mercator(xs, ys)
        merc_x, merc_y = mercator_wgs84_to_gcj02(merc_x, merc_y)
        self.ax.plot(
            merc_x, merc_y, "--",
            color="crimson", linewidth=2, alpha=0.5, zorder=3,
        )

    def _draw_velocity_hist(self, state):
        """在右侧子图绘制当前 uid 的速度分布直方图"""
        raw = get_raw_point_df()
        if raw is None or self.ax_hist is None:
            return
        uid_data = raw[(raw["uid"] == state.uid) & (raw["attribution"] != "origin")]
        velocities = uid_data["velocity"].dropna()
        velocities = velocities[velocities >= 0]

        self.ax_hist.clear()
        if len(velocities) > 0:
            self.ax_hist.hist(velocities, bins=25, color="steelblue",
                              edgecolor="white", alpha=0.85)
            self.ax_hist.axvline(velocities.median(), color="red", ls="--", lw=1.2,
                                 label=f'median={velocities.median():.1f}')
            mean_v = velocities.mean()
            self.ax_hist.axvline(mean_v, color="orange", ls="--", lw=1.2,
                                 label=f'mean={mean_v:.1f}')
            self.ax_hist.legend(fontsize=7, loc="upper right")
        self.ax_hist.set_xlabel("Velocity", fontsize=9)
        self.ax_hist.set_ylabel("Count", fontsize=9)
        self.ax_hist.set_title(f"UID {state.uid}\nn={len(velocities)}", fontsize=10)
        self.ax_hist.tick_params(labelsize=8)

    def _build_hex_road_overlay(self, hex_grid):
        """六边形模式道路叠加层 —— 视口范围内的散点图"""
        mode_rgb = {
            "TG":  (0.65, 0.00, 0.65),
            "GG":  (0.00, 0.45, 1.00),
            "GSD": (0.00, 0.75, 0.00),
            "TS":  (1.00, 0.00, 0.00),
        }

        # 视口 Mercator 四角 → WGS84 → 近似 hex 坐标范围
        from utils.geo_utils import _merc_to_wgs84
        corners_mx = [self._mx_min, self._mx_max, self._mx_max, self._mx_min]
        corners_my = [self._my_min, self._my_min, self._my_max, self._my_max]
        corners_lon, corners_lat = _merc_to_wgs84.transform(corners_mx, corners_my)
        hex_xs, hex_ys, hex_zs = wgs84_to_hex(corners_lon, corners_lat)

        margin = 3
        x_lo, x_hi = int(np.min(hex_xs)) - margin, int(np.max(hex_xs)) + margin
        y_lo, y_hi = int(np.min(hex_ys)) - margin, int(np.max(hex_ys)) + margin
        z_lo, z_hi = int(np.min(hex_zs)) - margin, int(np.max(hex_zs)) + margin

        for mode_name in MODE_LIST:
            if mode_name not in self.road_sets:
                continue
            road_set = self.road_sets[mode_name]
            # 收集视口范围内的道路 hex
            mx_list, my_list = [], []
            for x in range(x_lo, x_hi + 1):
                for y in range(y_lo, y_hi + 1):
                    z = -x - y
                    if not (z_lo <= z <= z_hi):
                        continue
                    if (x, y, z) not in road_set:
                        continue
                    if (x, y, z) not in hex_grid:
                        continue
                    # 检查是否在 Mercator 视口内
                    mx, my = hex_to_mercator(x, y, z)
                    if self._mx_min <= mx <= self._mx_max and self._my_min <= my <= self._my_max:
                        mx_list.append(mx)
                        my_list.append(my)

            if mx_list:
                r, g, b = mode_rgb.get(mode_name, (0.5, 0.5, 0.5))
                gx, gy = mercator_wgs84_to_gcj02(mx_list, my_list)
                self.ax.scatter(
                    gx, gy,
                    c=[(r, g, b)], s=6, alpha=0.5,
                    marker='h', zorder=2, label=mode_name,
                )

    def _build_road_rgb(self, raw_mapdata, resolution=500):
        """Build an RGB image resampled to a regular Mercator (EPSG:3857) grid.

        The Beijing 1954 GK grid is rotated ~4.5° relative to the Mercator
        axes at this longitude.  A plain imshow with a rectangular extent
        cannot represent that rotation, producing km-scale offsets from web map
        tiles.  Resampling onto a regular Mercator grid fixes the alignment.
        """
        matrices = mapdata_to_modelmatrix(raw_mapdata, MAP_ROWS, MAP_COLS)

        # ---- build the full-resolution source grid (564, 529) ----
        mode_rgb = {
            "TG":  (0.65, 0.00, 0.65),  # purple
            "GG":  (0.00, 0.45, 1.00),  # blue
            "GSD": (0.00, 0.75, 0.00),  # green
            "TS":  (1.00, 0.00, 0.00),  # red
        }

        src = np.ones((MAP_COLS, MAP_ROWS, 3), dtype=np.float32)
        for mode_name in MODE_LIST:
            if mode_name not in matrices:
                continue
            layer = np.array(matrices[mode_name], dtype=np.float32)
            mask = layer > 0
            if not mask.any():
                continue
            r, g, b = mode_rgb.get(mode_name, (0.5, 0.5, 0.5))
            src[mask, 0] = np.minimum(src[mask, 0], r)
            src[mask, 1] = np.minimum(src[mask, 1], g)
            src[mask, 2] = np.minimum(src[mask, 2], b)

        # ---- Mercator target grid ----
        mx_min, mx_max, my_min, my_max = full_grid_bounds_mercator()
        ncols = max(1, int((mx_max - mx_min) / resolution))
        nrows = max(1, int((my_max - my_min) / resolution))

        # pixel *centers* in Mercator (so extent is simply [mx_min, mx_max, …])
        merc_x = mx_min + (np.arange(ncols) + 0.5) * (mx_max - mx_min) / ncols
        merc_y = my_min + (np.arange(nrows) + 0.5) * (my_max - my_min) / nrows
        merc_xg, merc_yg = np.meshgrid(merc_x, merc_y)  # (nrows, ncols)

        # back-project to grid indices
        gx, gy = mercator_to_grid(merc_xg, merc_yg)
        gx_idx = np.clip(np.round(gx).astype(int), 0, MAP_COLS - 1)
        gy_idx = np.clip(np.round(gy).astype(int), 0, MAP_ROWS - 1)

        # nearest-neighbour sample from the source grid  src[col, row]
        sampled = src[gx_idx, gy_idx]  # (nrows, ncols, 3)
        return sampled

    def _update_title(self):
        state = self.state
        match = state.current_match_rate()
        reached = "ARRIVED" if state.reached else "moving"
        title = (
            f"Mode: {state.mode} | Steps: {state.step_count} | "
            f"Dist: {state.remaining_dist:.1f} | "
            f"Match: {match:.2%} | {reached}"
        )
        self.ax.set_title(title, fontsize=11, fontfamily="monospace")

    def _draw_legend_box(self):
        text = (
            "Keys:\n"
            "  Arrow / QWEASD   move\n"
            "  Backspace        undo\n"
            "  R                reset\n"
            "  Enter            save & label"
        )
        self.ax.text(
            0.02, 0.98, text,
            transform=self.ax.transAxes,
            fontsize=8, fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
        )

    def _draw_segment_info(self):
        """六边形模式：在右上角显示当前段的 dist / time / velocity"""
        state = self.state
        if not state.is_hex:
            return
        row = state.row

        def _fmt(val, fmt_spec):
            try:
                v = float(val)
            except (ValueError, TypeError):
                return "-"
            return format(v, fmt_spec)

        dist_str = _fmt(row.get("dist_d", ""), ".1f")
        time_str = _fmt(row.get("time_d", ""), ".0f")

        vel_str = _fmt(row.get("velocity_d", ""), ".2f")

        text = (
            "Segment Info:\n"
            f"  dist:     {dist_str} m\n"
            f"  time:     {time_str} s\n"
            f"  velocity: {vel_str} km/h"
        )
        self.ax.text(
            0.98, 0.98, text,
            transform=self.ax.transAxes,
            fontsize=8, fontfamily="monospace",
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.9),
        )

    def show_label_prompt(self):
        """Show the label selection prompt after Enter is pressed."""
        self.ax.set_title(
            "SELECT LABEL:  [1] GSD  [2] GG  [3] TS  [4] TG  [5] Mixed  [6] Other",
            fontsize=12, fontfamily="monospace", color="darkblue",
        )
        self.fig.canvas.draw_idle()

    def refresh(self):
        """Incremental update: path line + cursor position."""
        state = self.state
        if self.is_hex:
            xs = [p[0] for p in state.path_history]
            ys = [p[1] for p in state.path_history]
            zs = [p[2] for p in state.path_history]
            merc_x, merc_y = hex_to_mercator(xs, ys, zs)
        else:
            xs = [p[0] for p in state.path_history]
            ys = [p[1] for p in state.path_history]
            merc_x, merc_y = grid_to_mercator(xs, ys)
        merc_x, merc_y = mercator_wgs84_to_gcj02(merc_x, merc_y)
        self.path_line.set_data(merc_x, merc_y)

        if self.is_hex:
            cursor_mx, cursor_my = hex_to_mercator(*state.cur)
        else:
            cursor_mx, cursor_my = grid_to_mercator(state.cur_x, state.cur_y)
        cursor_mx, cursor_my = mercator_wgs84_to_gcj02(cursor_mx, cursor_my)
        self.cursor.set_offsets([[cursor_mx, cursor_my]])
        self._update_title()
        self.fig.canvas.draw_idle()


# ========================== Controller ==========================

class LabelController:
    """Coordinates state, renderer, and keyboard input."""

    def __init__(self, state: LabelState, renderer: PathRenderer,
                 output_dir: str, batch_mode: bool, current_idx: int,
                 start_in_label_mode: bool = False):
        self.state = state
        self.renderer = renderer
        self.output_dir = output_dir
        self.batch_mode = batch_mode
        self.current_idx = current_idx
        self.saved = False
        self.selecting_label = False
        self.next_requested = False
        self.go_back_requested = False

        if start_in_label_mode:
            self.selecting_label = True
            self.renderer.show_label_prompt()
            print(f"  Select label: 1=GSD 2=GG 3=TS 4=TG 5=Mixed 6=Other")
            print(f"  (Backspace to re-edit path)")

    def on_key(self, event):
        if event.key is None:
            return

        key = event.key.lower()

        # --- label selection mode: 1-5 to pick, backspace to cancel ---
        if self.selecting_label:
            if key in LABEL_OPTIONS:
                label = LABEL_OPTIONS[key]
                self._finalize(label)
                self.selecting_label = False
                self.saved = True
                print(f"  [LABELED] #{self.current_idx} -> {label}")
                if self.batch_mode:
                    self.next_requested = True
                    plt.close(self.renderer.fig)
                else:
                    self.renderer.ax.set_title(
                        self.renderer.ax.get_title() + f" [{label}]",
                        fontsize=11, fontfamily="monospace",
                    )
                    self.renderer.fig.canvas.draw_idle()
            elif key == "backspace":
                # Cancel label selection, return to path editing
                self.selecting_label = False
                self.renderer._update_title()
                self.renderer.fig.canvas.draw_idle()
                print(f"  Label selection cancelled, back to path editing")
            return

        keymap = HEX_KEY_MAP if self.state.is_hex else KEY_MAP
        if key not in keymap:
            return

        action, arg = keymap[key]

        if action == "move":
            if self.state.is_hex:
                if self.state.can_move(*HEX_DIRS[arg]):
                    self.state.apply_move(arg)
                    self.renderer.refresh()
            else:
                dx, dy = DX_DY[arg]
                if self.state.can_move(dx, dy):
                    self.state.apply_move(arg)
                    self.renderer.refresh()

        elif action == "undo":
            if len(self.state.path_history) <= 1:
                # No steps taken — go back to previous trajectory's label
                if self.current_idx > 0:
                    self.go_back_requested = True
                    plt.close(self.renderer.fig)
                    print(f"  Going back to re-label previous trajectory #{self.current_idx - 1}")
                else:
                    print(f"  Already at first trajectory, cannot go back")
            elif self.state.undo():
                self.renderer.refresh()

        elif action == "reset":
            self.state.reset()
            self.renderer.refresh()

        elif action == "save":
            self.selecting_label = True
            self.renderer.show_label_prompt()
            print(f"  Select label: 1=GSD 2=GG 3=TS 4=TG 5=Mixed 6=Other")
            print(f"  (Backspace to cancel)")

    def _finalize(self, label):
        """Write the complete record (path + label) to CSV and PNG."""
        state = self.state
        os.makedirs(self.output_dir, exist_ok=True)

        csv_path = os.path.join(self.output_dir, "traj_labeled.csv")

        if state.is_hex:
            traj_list = [[int(p[0]), int(p[1]), int(p[2])] for p in state.path_history]
        else:
            traj_list = [[float(p[0]), float(p[1])] for p in state.path_history]
        match_rate = state.current_match_rate()

        row = state.row

        # 全量原始字段（排除 order 和 mode）
        skip_cols = {"order", "mode"}
        record = {}
        for col in row.index:
            if col not in skip_cols:
                record[col] = row[col]

        record["success"] = 1 if state.reached else 0
        record["match"] = match_rate
        record["steps"] = state.step_count
        record["traj"] = json.dumps(traj_list, ensure_ascii=False)
        record["mode"] = label

        df_new = pd.DataFrame([record])
        # uid, idx_o, idx_d 放前三列
        front_cols = [c for c in ["uid", "idx_o", "idx_d"] if c in df_new.columns]
        other_cols = [c for c in df_new.columns if c not in front_cols]
        df_new = df_new[front_cols + other_cols]
        if os.path.exists(csv_path):
            df_new.to_csv(csv_path, mode="a", index=False, header=False, encoding="utf-8")
        else:
            df_new.to_csv(csv_path, index=False, encoding="utf-8")

        png_name = f"ep_{self.current_idx:04d}_order_{state.order}_{label}.png"
        png_path = os.path.join(self.output_dir, png_name)
        self.renderer.fig.savefig(png_path, bbox_inches="tight", dpi=150)
        print(f"  -> CSV: {csv_path}")
        print(f"  -> PNG: {png_path}")


# ========================== Main Loop ==========================

def run_single(state, raw_mapdata, output_dir, batch_mode, idx,
               start_in_label_mode=False, road_sets=None, traj_df=None):
    """Run labeling for one trajectory. Returns (next_idx, keep_going)."""
    renderer = PathRenderer(state, raw_mapdata, road_sets=road_sets,
                            traj_df=traj_df, current_idx=idx,
                            output_dir=output_dir)
    controller = LabelController(
        state, renderer, output_dir, batch_mode, idx,
        start_in_label_mode=start_in_label_mode,
    )

    renderer.fig.canvas.mpl_connect("key_press_event", controller.on_key)

    def on_close(event):
        if not controller.saved:
            print(f"  [WARN] window closed, #{idx} not saved")

    renderer.fig.canvas.mpl_connect("close_event", on_close)

    if not start_in_label_mode:
        print(f"\n{'='*60}")
        print(f"#{idx}  order={state.order}  mode={state.mode}")
        if state.is_hex:
            print(f"Start: {state.start}  ->  End: {state.end}")
            print(f"Keys: W/A/S/D/Q/E=move  Backspace=undo  R=reset  Enter=save & label")
        else:
            print(f"Start: ({state.start_x}, {state.start_y})  ->  End: ({state.end_x}, {state.end_y})")
            print(f"Keys: Arrows/WASD=move  Backspace=undo  R=reset  Enter=save & label")
        print(f"{'='*60}")
    else:
        print(f"\n#{idx}  order={state.order}  mode={state.mode}  [RE-LABEL]")

    plt.show(block=True)

    if controller.next_requested:
        return idx + 1, True
    elif controller.go_back_requested:
        return max(0, idx - 1), True
    else:
        return idx, False


def main():
    parser = argparse.ArgumentParser(description="Interactive path labeling tool")
    parser.add_argument("--index", type=int, default=None,
                        help="label a single trajectory (0-based index)")
    parser.add_argument("--batch", action="store_true",
                        help="label all trajectories in sequence")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="output directory (default: label_output)")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH,
                        help="path to trajectory CSV")
    parser.add_argument("--mapdata", type=str, default=DEFAULT_MAPDATA_PATH,
                        help="path to map pickle file")
    parser.add_argument("--grid", type=str, default="hex",
                        choices=["quad", "hex"],
                        help="grid type: quad (1000m quadrilateral) or hex (200m hexagon)")
    parser.add_argument("--sample-step", type=int, default=1,
                        help="sampling interval for point-sequence CSV (default: 10)")
    args = parser.parse_args()

    is_hex = (args.grid == "hex")

    if is_hex:
        # ---- 六边形模式 ----
        if args.mapdata == DEFAULT_MAPDATA_PATH:
            args.mapdata = HEX_PKL_PATH
        print("Loading hex grid map data...")
        raw_mapdata = load_hex_mapdata(args.mapdata)
        print(f"  Hex cells: {len(raw_mapdata):,}")
        print("Building road sets...")
        road_sets = hex_mapdata_to_road_sets(raw_mapdata)
        for m in MODE_LIST:
            print(f"  {m}: {len(road_sets[m]):,} cells")
        print("Loading trajectory data...")
        traj_df = load_traj_csv_hex(args.csv, sample_step=args.sample_step)
        print(f"Total trajectories: {len(traj_df)}")

        def make_state(row):
            mode = str(row["mode"]).strip()
            if mode not in MODE_LIST:
                mode = "ALL"
            if mode == "ALL":
                multi = set().union(*road_sets.values())
            else:
                multi = build_multi_mapdata_hex(road_sets, mode)
            return LabelState(row, multi, is_hex=True, hex_grid=raw_mapdata)
    else:
        # ---- 四边形模式（原有逻辑）----
        print("Loading map data...")
        raw_mapdata = load_mapdata(args.mapdata)
        print("Loading trajectory data...")
        traj_df = load_traj_csv(args.csv)
        print(f"Total trajectories: {len(traj_df)}")
        road_sets = None

        def make_state(row):
            mode = str(row["mode"]).strip()
            multi = build_multi_mapdata(raw_mapdata, mode)
            return LabelState(row, multi)

    output_dir = args.output

    if args.index is not None:
        if args.index < 0 or args.index >= len(traj_df):
            print(f"Error: index out of range [0, {len(traj_df)-1}]")
            sys.exit(1)
        state = make_state(traj_df.iloc[args.index])
        run_single(state, raw_mapdata, output_dir, batch_mode=False,
                   idx=args.index, road_sets=road_sets, traj_df=traj_df)

    elif args.batch:
        idx = 0
        while idx < len(traj_df):
            state = make_state(traj_df.iloc[idx])
            next_idx, keep_going = run_single(
                state, raw_mapdata, output_dir, batch_mode=True,
                idx=idx, road_sets=road_sets, traj_df=traj_df,
            )
            if not keep_going:
                print(f"Labeling stopped, completed up to #{idx}")
                break
            start_label = (next_idx < idx)
            idx = next_idx
            if start_label:
                state = make_state(traj_df.iloc[idx])
                next_idx, keep_going = run_single(
                    state, raw_mapdata, output_dir, batch_mode=True,
                    idx=idx, road_sets=road_sets, traj_df=traj_df,
                    start_in_label_mode=True,
                )
                if not keep_going:
                    break
                idx = next_idx
        if idx >= len(traj_df):
            print(f"\nAll {len(traj_df)} trajectories labeled!")

    else:
        # Prompt user for starting index
        while True:
            try:
                user_input = input(
                    f"Enter starting index [0-{len(traj_df)-1}], or press Enter for 0: "
                ).strip()
                if user_input == "":
                    start_idx = 0
                else:
                    start_idx = int(user_input)
                if 0 <= start_idx < len(traj_df):
                    break
                print(f"  Error: index out of range [0, {len(traj_df)-1}]")
            except ValueError:
                print(f"  Error: please enter a valid integer")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                sys.exit(0)

        idx = start_idx
        while idx < len(traj_df):
            state = make_state(traj_df.iloc[idx])
            next_idx, keep_going = run_single(
                state, raw_mapdata, output_dir, batch_mode=True,
                idx=idx, road_sets=road_sets, traj_df=traj_df,
            )
            if not keep_going:
                print(f"Labeling stopped, completed up to #{idx}")
                break
            start_label = (next_idx < idx)
            idx = next_idx
            if start_label:
                state = make_state(traj_df.iloc[idx])
                next_idx, keep_going = run_single(
                    state, raw_mapdata, output_dir, batch_mode=True,
                    idx=idx, road_sets=road_sets, traj_df=traj_df,
                    start_in_label_mode=True,
                )
                if not keep_going:
                    break
                idx = next_idx
        if idx >= len(traj_df):
            print(f"\nAll {len(traj_df)} trajectories labeled!")


if __name__ == "__main__":
    main()
