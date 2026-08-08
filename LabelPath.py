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
import argparse
import warnings

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

# 中文字体：图例/标题等含中文（如"高铁/铁路/国道..."），需指定支持中文的字体，
# 否则显示为方框。优先用 Windows 自带的微软雅黑，缺失时回退 SimHei。
_font_mgr = matplotlib.font_manager.fontManager
_zh_fonts = ["Microsoft YaHei", "SimHei", "Microsoft JhengHei"]
_available = {f.name for f in _font_mgr.ttflist}
_zh_ok = [f for f in _zh_fonts if f in _available]
if _zh_ok:
    matplotlib.rcParams["font.sans-serif"] = _zh_ok + matplotlib.rcParams["font.sans-serif"]
    matplotlib.rcParams["axes.unicode_minus"] = False  # 负号正常显示
# ---------------------------------------------------------------------------
from utils.geo_utils import (
    wgs84_to_hex,
    hex_to_wgs84,
    hex_to_mercator,
    hex_distance,
    hex_in_map,
    _init_hex_origin,
    get_hex_grid,
    mercator_wgs84_to_gcj02,
)
from utils.basemap import add_basemap, USE_BASEMAP
from utils.osm_pois import (
    CATEGORY_LABELS as POI_CATEGORY_LABELS,
    CATEGORY_SUBWAY,
    CATEGORY_TOLL,
    CATEGORY_TRAIN,
    DEFAULT_POI_PATH,
    group_pois_by_hex,
    load_osm_pois,
)

from utils.tools import (
    hex_mapdata_to_road_sets,
    calculate_match_rate_hex,
    decode_road_code,
    MODE_LIST,
    MODE_LABELS,
    ROAD_KEY_TO_MODE,
)

# ========================== Constants ==========================

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
HEX_PKL_PATH = r"data\hex_grid_2025.pkl"

DEFAULT_CSV_PATH = r"data\dataset_multicity_20230917_unpacked.csv"
DEFAULT_OUTPUT_DIR = "label_output"

DISTANCE_THRESHOLD = 1.0

# Label options after saving (press 1-6 to select)
# 与路网渲染分组一致：TG/TS/DT/GG/GSD；6=Other
LABEL_OPTIONS = {
    "1": "TG",
    "2": "TS",
    "3": "DT",
    "4": "GG",
    "5": "GSD",
    "6": "Other",
}

# 路网渲染分组配色（RGB），legend 与 overlay 共用，保证图例与路网颜色一致
MODE_COLORS = {
    "TG":  (0.65, 0.00, 0.65),  # 高铁   紫
    "TS":  (0.95, 0.45, 0.00),  # 铁路   橙
    "DT":  (0.00, 0.45, 1.00),  # 地铁   蓝
    "GG":  (1.00, 0.00, 0.00),  # 高速   红
    "GSD": (0.00, 0.75, 0.00),  # 国/省/环 绿
}

# 已标注参考点沿用路网配色；Other 没有对应路网，使用中性灰。
LABELED_POINT_COLORS = {
    **MODE_COLORS,
    "Other": (0.38, 0.38, 0.38),
}

# 视觉层级：当前起终信令点最醒目，较远上下文与静态辅助层主动退后。
ROAD_OVERLAY_ALPHA = 0.20
POI_ALPHA = 0.42
CONTEXT_ALPHA = 0.88

# OSM 点位使用独立形状，避免与已有线状路网颜色混淆。
POI_STYLES = {
    CATEGORY_SUBWAY: {
        "label": "地铁站", "marker": "o", "color": "#00a8ff", "size": 78,
    },
    CATEGORY_TRAIN: {
        "label": "火车站", "marker": "^", "color": "#ff8c00", "size": 92,
    },
    CATEGORY_TOLL: {
        "label": "高速收费站", "marker": "P", "color": "#c000ff", "size": 96,
    },
}
POI_LABEL_LIMIT = 45


def _label_prompt_str():
    """由 LABEL_OPTIONS 生成标签选择提示文本，避免硬编码漂移。"""
    return "  ".join(f"[{k}] {v}" for k, v in LABEL_OPTIONS.items())


def _annotation_key(row):
    """Return the stable (uid, idx_o) key shared by source and label CSVs."""
    try:
        return int(row["uid"]), int(row["idx_o"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _canonical_label_mode(value):
    text = str(value).strip()
    for mode in LABELED_POINT_COLORS:
        if text.casefold() == mode.casefold():
            return mode
    return None


def load_labeled_modes(output_dir):
    """Load labels already saved by this annotation run for map feedback."""
    csv_path = os.path.join(output_dir, "traj_labeled.csv")
    if not os.path.exists(csv_path):
        return {}
    try:
        labeled = pd.read_csv(csv_path, encoding="utf-8")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"  [WARN] Failed to load existing labels for map feedback: {exc}")
        return {}

    required = {"uid", "idx_o", "mode"}
    if not required.issubset(labeled.columns):
        print(f"  [WARN] Existing label CSV lacks columns: {sorted(required - set(labeled.columns))}")
        return {}

    result = {}
    for _, row in labeled.iterrows():
        key = _annotation_key(row)
        mode = _canonical_label_mode(row["mode"])
        if key is not None and mode is not None:
            result[key] = mode
    return result

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

def load_hex_mapdata(path=HEX_PKL_PATH):
    """加载六边形网格 pkl，触发原点/查表/仿射初始化。

    复用 geo_utils 内部已加载的 pkl，避免重复读取 2.2GB 文件。
    """
    return get_hex_grid(path)


# 原始点级数据缓存，供速度分布图使用
_RAW_POINT_DF = None
_RAW_VELOCITIES_BY_UID = {}

# 不保存轨迹开关（运行时按 N 切换）：开启后保存时 traj(路径)留空，其余字段照常写入
_NO_TRAJ_MODE = True


def get_raw_point_df():
    return _RAW_POINT_DF


def _build_segments_from_point_df(df, sample_step):
    """Vectorized point-sequence to OD conversion.

    The previous implementation iterated over every one of the 270k+ ODs with
    ``group.iloc``.  This keeps the same sampling and interval-sum semantics,
    but shifts sampled rows and cumulative sums in Pandas/NumPy.
    """
    if sample_step <= 0:
        raise ValueError("sample_step must be positive")
    if df.empty:
        return pd.DataFrame()

    uid_groups = df.groupby("uid", sort=False)
    group_position = uid_groups.cumcount()
    sampled_mask = (group_position % int(sample_step)) == 0
    sampled = df.loc[sampled_mask].copy()
    sampled["_group_position"] = group_position.loc[sampled_mask].to_numpy()
    sampled_groups = sampled.groupby("uid", sort=False)
    following = sampled_groups.shift(-1)
    valid = following["hex_x"].notna()
    if not valid.any():
        return pd.DataFrame()

    origin = sampled.loc[valid]
    destination = following.loc[valid]

    def _interval_sum(column):
        if column not in df.columns:
            return np.zeros(int(valid.sum()), dtype=float)
        values = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
        cumulative = values.groupby(df["uid"], sort=False).cumsum()
        sampled_cumulative = cumulative.loc[sampled.index]
        next_cumulative = sampled_cumulative.groupby(sampled["uid"], sort=False).shift(-1)
        return (next_cumulative - sampled_cumulative).loc[valid].to_numpy(dtype=float)

    time_sum = _interval_sum("time_value")
    dist_sum = _interval_sum("dist_value")
    if "velocity" in destination.columns:
        fallback_velocity = pd.to_numeric(
            destination["velocity"], errors="coerce",
        ).fillna(0.0).to_numpy(dtype=float)
    else:
        fallback_velocity = np.zeros(len(origin), dtype=float)
    velocity_d = np.where(
        (time_sum > 0) & (dist_sum > 0),
        dist_sum / time_sum * 3.6,
        fallback_velocity,
    )

    def _origin_values(column, default=0, dtype=None):
        if column not in origin.columns:
            return np.full(len(origin), default, dtype=dtype)
        values = origin[column]
        return values.to_numpy(dtype=dtype) if dtype is not None else values.to_numpy()

    def _destination_values(column, default=0, dtype=None):
        if column not in destination.columns:
            return np.full(len(destination), default, dtype=dtype)
        values = destination[column]
        return values.to_numpy(dtype=dtype) if dtype is not None else values.to_numpy()

    idx_o = (
        _origin_values("idx", dtype=np.int64)
        if "idx" in origin.columns
        else _origin_values("_group_position", dtype=np.int64)
    )
    idx_d = (
        _destination_values("idx", dtype=np.int64)
        if "idx" in destination.columns
        else _destination_values("_group_position", dtype=np.int64)
    )
    source_mode = (
        origin["attribution"].astype(str).to_numpy()
        if "attribution" in origin.columns
        else np.full(len(origin), "ALL", dtype=object)
    )

    return pd.DataFrame({
        "x_o": _origin_values("hex_x", dtype=np.int64),
        "y_o": _origin_values("hex_y", dtype=np.int64),
        "z_o": _origin_values("hex_z", dtype=np.int64),
        "x_d": _destination_values("hex_x", dtype=np.int64),
        "y_d": _destination_values("hex_y", dtype=np.int64),
        "z_d": _destination_values("hex_z", dtype=np.int64),
        "mode": source_mode,
        "order": _origin_values("uid", dtype=np.int64),
        "uid": _origin_values("uid", dtype=np.int64),
        "idx_o": idx_o,
        "idx_d": idx_d,
        "stime_o": _origin_values("stime", dtype=np.int64),
        "lat_o": _origin_values("lat", dtype=float),
        "lon_o": _origin_values("lon", dtype=float),
        "time_d": time_sum,
        "dist_d": dist_sum,
        "velocity_d": velocity_d,
    })


def load_traj_csv_hex(path, sample_step=10):
    """读取六边形模式轨迹 CSV，按 uid 分组生成起终点记录。

    支持两种格式:
    1. 起终点格式: x_o,y_o,z_o,x_d,y_d,z_d,mode,order
    2. 点序列格式: uid,hex_x,hex_y,hex_z,...  → 按 uid 分组，每 sample_step 行采样生成段
    """
    global _RAW_POINT_DF, _RAW_VELOCITIES_BY_UID
    df = pd.read_csv(path)
    # 列名兼容: 不同数据集标识列命名不同，统一为 uid
    if "traj_id" in df.columns and "uid" not in df.columns:
        df = df.rename(columns={"traj_id": "uid"})
    # stime 可能是日期字符串，统一转为整数时间戳
    if "stime" in df.columns and df["stime"].dtype == object:
        df["stime"] = pd.to_datetime(df["stime"]).astype("int64") // 10**9
    # uid 统一为整数（traj_id 可能是 "20230917_0000"，需与 records 中 int(uid) 类型一致，
    # 否则速度分布图按 uid 匹配时类型不一致导致取不到数据）
    if "uid" in df.columns:
        df["uid"] = df["uid"].astype(str).str.replace("_", "").astype("int64")
    # 检测格式: 有 hex_x/hex_y/hex_z 列 → 点序列格式
    if "hex_x" in df.columns and "uid" in df.columns:
        _RAW_POINT_DF = df  # 缓存原始数据
        if "velocity" in df.columns:
            valid_velocity = pd.to_numeric(df["velocity"], errors="coerce").ge(0)
            if "attribution" in df.columns:
                valid_velocity &= df["attribution"].ne("origin")
            _RAW_VELOCITIES_BY_UID = {
                int(uid): group["velocity"].dropna().to_numpy(dtype=float)
                for uid, group in df.loc[valid_velocity].groupby("uid", sort=False)
            }
        else:
            _RAW_VELOCITIES_BY_UID = {}
        return _build_segments_from_point_df(df, sample_step)
    return df


# ========================== State ==========================

class LabelState:
    """Holds labeling state for a single trajectory (hex)."""

    def __init__(self, row, multi_mapdata, hex_grid=None):
        self.row = row
        self.order = int(row["order"]) if "order" in row.index else 0
        self.mode = str(row.get("mode", "ALL")).strip()
        self.uid = int(row["uid"]) if "uid" in row.index else self.order
        self.hex_grid = hex_grid

        self.start = (int(row["x_o"]), int(row["y_o"]), int(row["z_o"]))
        self.end = (int(row["x_d"]), int(row["y_d"]), int(row["z_d"]))

        self.multi_mapdata = multi_mapdata

        self.cur = self.start
        self.path_history = [self.cur]
        self.step_count = 0

    @property
    def reached(self):
        return hex_distance(self.cur, self.end) <= DISTANCE_THRESHOLD

    @property
    def remaining_dist(self):
        return hex_distance(self.cur, self.end)

    def current_match_rate(self):
        if len(self.path_history) <= 1:
            return 0.0
        return calculate_match_rate_hex(self.path_history, self.multi_mapdata)

    def can_move(self, dx, dy, dz):
        nx, ny, nz = self.cur[0] + dx, self.cur[1] + dy, self.cur[2] + dz
        return hex_in_map(nx, ny, nz, self.hex_grid)

    def apply_move(self, action_id):
        dx, dy, dz = HEX_DIRS[action_id]
        self.cur = (self.cur[0] + dx, self.cur[1] + dy, self.cur[2] + dz)
        self.path_history.append(self.cur)
        self.step_count += 1

    def undo(self):
        if len(self.path_history) > 1:
            self.path_history.pop()
            self.cur = self.path_history[-1]
            self.step_count = max(0, self.step_count - 1)
            return True
        return False

    def reset(self):
        self.cur = self.start
        self.path_history = [self.cur]
        self.step_count = 0


# ========================== Renderer ==========================

class PathRenderer:
    """Manages the matplotlib figure and incremental updates (hex)."""

    def __init__(self, state: LabelState, raw_mapdata, road_sets=None,
                 traj_df=None, current_idx=None, output_dir=None, pois=None,
                 labeled_modes=None):
        self.raw_mapdata = raw_mapdata
        self.road_sets = road_sets
        self.traj_df = traj_df
        self.output_dir = output_dir
        self.labeled_modes = labeled_modes if labeled_modes is not None else {}
        self.pois = pois or []
        self.pois_by_hex = group_pois_by_hex(self.pois)
        self._prepare_static_indexes()

        # 左右分栏：左侧地图，右侧速度分布
        self.fig = plt.figure(figsize=(16, 9))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.02)
        self.ax = self.fig.add_subplot(gs[0])
        self.ax_hist = self.fig.add_subplot(gs[1])
        self.fig.canvas.manager.set_window_title("LabelPath — Hex Grid")
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_poi_hover)
        self.show_segment(state, current_idx, initial=True)

    def _prepare_static_indexes(self):
        """Project and index immutable layers once for the whole annotation run."""
        self._road_display = {}
        if self.road_sets:
            for mode_name in MODE_LIST:
                road_set = self.road_sets.get(mode_name)
                if not road_set:
                    continue
                coordinates = np.asarray(tuple(road_set), dtype=np.int32)
                mx, my = hex_to_mercator(
                    coordinates[:, 0], coordinates[:, 1], coordinates[:, 2],
                )
                gx, gy = mercator_wgs84_to_gcj02(mx, my)
                self._road_display[mode_name] = (
                    np.asarray(mx), np.asarray(my), np.asarray(gx), np.asarray(gy),
                )

        self._poi_index = {}
        for category in POI_STYLES:
            records = [record for record in self.pois if record["category"] == category]
            if not records:
                continue
            self._poi_index[category] = (
                records,
                np.fromiter((record["_mercator_x"] for record in records), dtype=float),
                np.fromiter((record["_mercator_y"] for record in records), dtype=float),
            )

        self._uid_od_current = None
        self._uid_od_total = None
        if self.traj_df is not None and "uid" in self.traj_df.columns:
            uid_values = pd.to_numeric(self.traj_df["uid"], errors="coerce")
            grouped = uid_values.groupby(uid_values, sort=False)
            self._uid_od_current = (grouped.cumcount() + 1).to_numpy(dtype=np.int32)
            self._uid_od_total = grouped.transform("size").to_numpy(dtype=np.int32)
            self._uid_segment_positions = {
                int(uid): np.asarray(positions, dtype=np.int32)
                for uid, positions in grouped.indices.items()
                if not pd.isna(uid)
            }
        else:
            self._uid_segment_positions = {}

    def show_segment(self, state, current_idx, initial=False):
        """Render another OD in the existing figure instead of reopening it."""
        self.state = state
        self.current_idx = current_idx
        self._poi_scatter_meta = []
        self._poi_hover = None
        self.ax.clear()

        self._init_hex_view(state, self.raw_mapdata)

        self.ax.set_xlabel("Web Mercator X (EPSG:3857)")
        self.ax.set_ylabel("Web Mercator Y (EPSG:3857)")
        self.ax.grid(False)

        # ---- 出行模式颜色图例（路网渲染分组）----
        mode_handles = [
            Line2D([0], [0], color=MODE_COLORS[m], lw=3,
                   label=f"{m} {MODE_LABELS.get(m, '')}")
            for m in MODE_LIST
        ]
        poi_handles = [
            Line2D(
                [0], [0], linestyle="None", marker=style["marker"],
                markerfacecolor=style["color"], markeredgecolor="white",
                markeredgewidth=1.0, markersize=8,
                label=style["label"],
            )
            for category, style in POI_STYLES.items()
            if category in self._poi_index
        ]
        self.ax.legend(
            handles=mode_handles + poi_handles, loc="lower right",
            fontsize=7, handlelength=1.5, borderpad=0.4, labelspacing=0.3,
        )

        self._update_title()
        self._draw_legend_box()
        self._draw_segment_info()
        self._init_cell_info()
        if initial:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="This figure includes Axes that are not compatible",
                )
                self.fig.tight_layout()
        self.fig.canvas.draw_idle()

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

        # OSM POI 层：只从本地缓存读取，不在标注过程中发起网络请求。
        if self.pois:
            self._draw_osm_pois()

        # 起终点标记（GCJ-02 偏移）
        start_mx, start_my = mercator_wgs84_to_gcj02(start_mx, start_my)
        end_mx, end_my = mercator_wgs84_to_gcj02(end_mx, end_my)
        self.ax.scatter(
            start_mx, start_my,
            c="white", marker="o", s=92,
            edgecolors="white", linewidths=1.5, alpha=0.88, zorder=7.8,
        )
        self.start_handle = self.ax.scatter(
            start_mx, start_my,
            c="#39d353", marker="o", s=58,
            edgecolors="darkgreen", linewidths=1.4, zorder=8, label="Start",
        )
        self.ax.scatter(
            end_mx, end_my,
            c="white", marker="o", s=142,
            edgecolors="white", linewidths=1.8, alpha=0.92, zorder=8.05,
        )
        self.end_handle = self.ax.scatter(
            end_mx, end_my,
            c="#ff2538", marker="X", s=102,
            edgecolors="#8b0015", linewidths=1.4, zorder=8.2, label="End",
        )
        (self.path_line,) = self.ax.plot(
            [], [], "-",
            color="crimson", linewidth=2.5, alpha=0.85, zorder=3, label="Path",
        )

        cursor_mx, cursor_my = hex_to_mercator(*state.cur)
        cursor_mx, cursor_my = mercator_wgs84_to_gcj02(cursor_mx, cursor_my)
        self.cursor = self.ax.scatter(
            cursor_mx, cursor_my,
            c="cyan", marker="o", s=48,
            edgecolors="darkblue", linewidths=1.8, zorder=8.3, label="Cursor",
        )

        # ---- 上下文参考点（前一段起点 / 后一段终点）----
        self._draw_context_points(state)

        # ---- 右侧：速度分布直方图 ----
        self._draw_velocity_hist(state)

    def _draw_context_points(self, state):
        """绘制同一 UID 中落在当前视口内的全部前后参考点。

        视口只由当前 OD 起终点决定。上下文先按完整 UID 轨迹投影，再按
        已有坐标轴范围裁剪；绘制完成后显式恢复范围，保证参考点永远不会
        触发缩放。
        """
        if self.traj_df is None or self.current_idx is None:
            return

        traj_df = self.traj_df
        idx = self.current_idx
        positions_by_uid = getattr(self, "_uid_segment_positions", None)
        if positions_by_uid is None:
            uid_values = pd.to_numeric(traj_df["uid"], errors="coerce")
            grouped = uid_values.groupby(uid_values, sort=False)
            positions_by_uid = {
                int(uid): np.asarray(positions, dtype=np.int32)
                for uid, positions in grouped.indices.items()
                if not pd.isna(uid)
            }
            self._uid_segment_positions = positions_by_uid
        uid_positions = positions_by_uid.get(state.uid, np.empty(0, dtype=np.int32))
        previous_positions = uid_positions[uid_positions < idx]
        next_positions = uid_positions[uid_positions > idx]

        def _project_rows(positions, prefix):
            if not len(positions):
                return np.empty(0), np.empty(0)
            rows = traj_df.iloc[positions]
            mx, my = hex_to_mercator(
                rows[f"x_{prefix}"].to_numpy(dtype=np.int64),
                rows[f"y_{prefix}"].to_numpy(dtype=np.int64),
                rows[f"z_{prefix}"].to_numpy(dtype=np.int64),
            )
            gx, gy = mercator_wgs84_to_gcj02(mx, my)
            return np.asarray(gx), np.asarray(gy)

        prev_x, prev_y = _project_rows(previous_positions, "o")
        next_x, next_y = _project_rows(next_positions, "d")
        start_x, start_y = hex_to_mercator(*state.start)
        start_x, start_y = mercator_wgs84_to_gcj02(start_x, start_y)
        end_x, end_y = hex_to_mercator(*state.end)
        end_x, end_y = mercator_wgs84_to_gcj02(end_x, end_y)

        x_limits = self.ax.get_xlim()
        y_limits = self.ax.get_ylim()
        x_min, x_max = sorted(x_limits)
        y_min, y_max = sorted(y_limits)

        # 完整轨迹链交给 Matplotlib 视口裁剪；参考点只提交视口内的部分。
        for chain_x, chain_y in (
            (np.append(prev_x, float(start_x)), np.append(prev_y, float(start_y))),
            (np.insert(next_x, 0, float(end_x)), np.insert(next_y, 0, float(end_y))),
        ):
            if len(chain_x) >= 2:
                self.ax.plot(
                    chain_x, chain_y, "--",
                    color="dimgray", linewidth=1.0,
                    alpha=0.58, zorder=5.6, scalex=False, scaley=False,
                )

        def _saved_color(row_idx, fallback):
            row_i = traj_df.iloc[row_idx]
            mode = self.labeled_modes.get(_annotation_key(row_i))
            return LABELED_POINT_COLORS.get(mode, fallback)

        def _draw_visible(positions, xs, ys, fallback, edge, near_position, near_size):
            if not len(positions):
                return
            visible = (
                (xs >= x_min) & (xs <= x_max)
                & (ys >= y_min) & (ys <= y_max)
            )
            near = visible & (positions == near_position)
            far = visible & ~near
            if far.any():
                far_positions = positions[far]
                self.ax.scatter(
                    xs[far], ys[far],
                    c=[_saved_color(int(row_idx), fallback) for row_idx in far_positions],
                    marker="D", s=28, edgecolors=edge, linewidths=0.9,
                    alpha=CONTEXT_ALPHA, zorder=6.6,
                )
            if near.any():
                near_index = int(np.flatnonzero(near)[0])
                row_idx = int(positions[near_index])
                self.ax.scatter(
                    [xs[near_index]], [ys[near_index]],
                    c=[_saved_color(row_idx, fallback)], marker="D",
                    s=near_size, edgecolors=edge, linewidths=1.5,
                    alpha=CONTEXT_ALPHA, zorder=7.25,
                )

        nearest_previous = int(previous_positions[-1]) if len(previous_positions) else -1
        nearest_next = int(next_positions[0]) if len(next_positions) else -1
        _draw_visible(
            previous_positions, prev_x, prev_y,
            "#d8a24a", "#7a4b00", nearest_previous, 42,
        )
        _draw_visible(
            next_positions, next_x, next_y,
            "deepskyblue", "#005bbb", nearest_next, 46,
        )

        # 明确恢复，防止任何新增 artist 改变当前 OD 的视图范围。
        self.ax.set_xlim(x_limits)
        self.ax.set_ylim(y_limits)

    def _draw_velocity_hist(self, state):
        """在右侧子图绘制当前 uid 的速度分布直方图"""
        if self.ax_hist is None:
            return
        if getattr(self, "_hist_uid", None) == state.uid:
            return
        self._hist_uid = state.uid
        velocities = _RAW_VELOCITIES_BY_UID.get(state.uid, np.empty(0, dtype=float))

        self.ax_hist.clear()
        if len(velocities) > 0:
            self.ax_hist.hist(velocities, bins=25, color="steelblue",
                              edgecolor="white", alpha=0.85)
            median_v = float(np.median(velocities))
            self.ax_hist.axvline(median_v, color="red", ls="--", lw=1.2,
                                 label=f'median={median_v:.1f}')
            mean_v = float(np.mean(velocities))
            self.ax_hist.axvline(mean_v, color="orange", ls="--", lw=1.2,
                                 label=f'mean={mean_v:.1f}')
            self.ax_hist.legend(fontsize=7, loc="upper right")
        self.ax_hist.set_xlabel("Velocity", fontsize=9)
        self.ax_hist.set_ylabel("Count", fontsize=9)
        self.ax_hist.set_title(f"UID {state.uid}\nn={len(velocities)}", fontsize=10)
        self.ax_hist.tick_params(labelsize=8)

    def _build_hex_road_overlay(self, hex_grid):
        """六边形模式道路叠加层 —— 视口范围内的散点图（按 6 分组配色）"""
        if self._road_display:
            for mode_name in MODE_LIST:
                projected = self._road_display.get(mode_name)
                if projected is None:
                    continue
                mx, my, gx, gy = projected
                visible = (
                    (mx >= self._mx_min) & (mx <= self._mx_max)
                    & (my >= self._my_min) & (my <= self._my_max)
                )
                if not visible.any():
                    continue
                color = MODE_COLORS.get(mode_name, (0.5, 0.5, 0.5))
                self.ax.scatter(
                    gx[visible], gy[visible],
                    c=[color], s=6, alpha=ROAD_OVERLAY_ALPHA,
                    marker="h", zorder=2, label=mode_name,
                )
            return

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
                r, g, b = MODE_COLORS.get(mode_name, (0.5, 0.5, 0.5))
                gx, gy = mercator_wgs84_to_gcj02(mx_list, my_list)
                self.ax.scatter(
                    gx, gy,
                    c=[(r, g, b)], s=6, alpha=ROAD_OVERLAY_ALPHA,
                    marker='h', zorder=2, label=mode_name,
                )

    def _draw_osm_pois(self):
        """Draw cached OSM stations/toll points inside the current viewport."""
        visible = []
        visible_by_category = {}
        for category, (records, mx, my) in self._poi_index.items():
            mask = (
                (mx >= self._mx_min) & (mx <= self._mx_max)
                & (my >= self._my_min) & (my <= self._my_max)
            )
            indices = np.flatnonzero(mask)
            if not len(indices):
                continue
            category_records = [records[int(index)] for index in indices]
            visible_by_category[category] = category_records
            visible.extend(category_records)
        if not visible:
            return

        for category, style in POI_STYLES.items():
            category_records = visible_by_category.get(category, [])
            if not category_records:
                continue
            xs = [record["_display_x"] for record in category_records]
            ys = [record["_display_y"] for record in category_records]
            artist = self.ax.scatter(
                xs, ys,
                c=style["color"], marker=style["marker"], s=style["size"],
                edgecolors="#303030", linewidths=0.8, alpha=POI_ALPHA,
                zorder=3.6, label=style["label"],
            )
            self._poi_scatter_meta.append((artist, category_records))

        # Dense city-centre views remain readable: labels are shown directly
        # only for a modest number of POIs; hover and cell info always work.
        named_visible = [
            record for record in visible
            if record.get("name") and record.get("name") != "未命名"
        ]
        if len(named_visible) <= POI_LABEL_LIMIT:
            labeled = set()
            for record in named_visible:
                label_key = (
                    round(record["_display_x"], 1), round(record["_display_y"], 1),
                    record.get("name", "未命名"),
                )
                if label_key in labeled:
                    continue
                labeled.add(label_key)
                self.ax.annotate(
                    record.get("name", "未命名"),
                    (record["_display_x"], record["_display_y"]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=6.5, fontweight="bold", color="#444444", zorder=3.7,
                    bbox=dict(boxstyle="round,pad=0.14", facecolor="white", alpha=0.46,
                              edgecolor="none"),
                )

        self._poi_hover = self.ax.annotate(
            "", xy=(0, 0), xytext=(8, 8), textcoords="offset points",
            fontsize=8, zorder=8,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#555555", alpha=0.95),
        )
        self._poi_hover.set_visible(False)
        self.ax.text(
            0.5, 0.005, "POI © OpenStreetMap contributors",
            transform=self.ax.transAxes, horizontalalignment="center",
            verticalalignment="bottom", fontsize=6, color="#555555", zorder=7,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.72,
                      edgecolor="none"),
        )

    def _on_poi_hover(self, event):
        """Show category and OSM name when hovering over a POI marker."""
        if self._poi_hover is None:
            return
        if event.inaxes is self.ax:
            for artist, records in reversed(self._poi_scatter_meta):
                contains, details = artist.contains(event)
                indices = details.get("ind", []) if contains else []
                if len(indices):
                    record = records[int(indices[0])]
                    self._poi_hover.xy = (record["_display_x"], record["_display_y"])
                    category = POI_CATEGORY_LABELS.get(record["category"], record["category"])
                    self._poi_hover.set_text(f"{category}: {record.get('name', '未命名')}")
                    self._poi_hover.set_visible(True)
                    self.fig.canvas.draw_idle()
                    return
        if self._poi_hover.get_visible():
            self._poi_hover.set_visible(False)
            self.fig.canvas.draw_idle()

    def _update_title(self):
        state = self.state
        match = state.current_match_rate()
        reached = "ARRIVED" if state.reached else "moving"
        notraj = " | NO-TRAJ" if _NO_TRAJ_MODE else ""
        title = (
            f"Mode: {state.mode} | Steps: {state.step_count} | "
            f"Dist: {state.remaining_dist:.1f} | "
            f"Match: {match:.2%} | {reached}{notraj}"
        )
        self.ax.set_title(title, fontsize=11, fontfamily="monospace")

    def _draw_legend_box(self):
        text = (
            "Keys:\n"
            "  Arrow / QWEASD   move\n"
            "  Backspace        undo\n"
            "  R                reset\n"
            "  Enter            save & label\n"
            "  N                toggle no-traj"
        )
        self.ax.text(
            0.02, 0.98, text,
            transform=self.ax.transAxes,
            fontsize=8, fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
        )

    def _draw_segment_info(self):
        """在右上角显示当前段的 dist / time / velocity"""
        state = self.state
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
        uid_od_current, uid_od_total = self._uid_od_progress()
        uid_od_str = (
            f"{uid_od_current} / {uid_od_total}"
            if uid_od_current is not None else "-"
        )

        text = (
            "Segment Info:\n"
            f"  UID OD:   {uid_od_str}\n"
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

    def _uid_od_progress(self):
        """Return the current OD's 1-based position and total within this UID."""
        if self.current_idx is None:
            return None, None
        current_values = getattr(self, "_uid_od_current", None)
        total_values = getattr(self, "_uid_od_total", None)
        if current_values is None or total_values is None:
            if self.traj_df is None or "uid" not in self.traj_df.columns:
                return None, None
            uid_values = pd.to_numeric(self.traj_df["uid"], errors="coerce")
            grouped = uid_values.groupby(uid_values, sort=False)
            current_values = (grouped.cumcount() + 1).to_numpy(dtype=np.int32)
            total_values = grouped.transform("size").to_numpy(dtype=np.int32)
            self._uid_od_current = current_values
            self._uid_od_total = total_values
        if not 0 <= self.current_idx < len(current_values):
            return None, None
        return (
            int(current_values[self.current_idx]),
            int(total_values[self.current_idx]),
        )

    def _init_cell_info(self):
        """左下角：当前光标所在栅格的道路属性信息框（栅格可能复合多种道路）。"""
        # 不用 monospace：Windows 等宽字体无中文字形，会显示为方框，
        # 改用默认 sans-serif（已配置微软雅黑/SimHei）以正常显示中文。
        self.cell_info_text = self.ax.text(
            0.02, 0.02, "",
            transform=self.ax.transAxes,
            fontsize=8,
            verticalalignment="bottom", horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.92),
            zorder=7,
        )
        self._update_cell_info()

    def _update_cell_info(self):
        """刷新当前光标所在栅格的道路属性（细类 + 所属可视化分组）。

        一个栅格的 code 可能同时命中多种道路（如 高速+国道），叠加层只能
        显示一种颜色，这里把完整属性列出来辅助判断标注标签。
        """
        cur = self.state.cur
        try:
            code = self.state.hex_grid[cur]["code"]
        except (KeyError, IndexError, TypeError):
            code = 0

        hits = decode_road_code(code)
        if not hits:
            body = "  无道路"
        else:
            rendered, others = [], []
            for key, label in hits:
                mode = ROAD_KEY_TO_MODE.get(key)
                if mode:
                    rendered.append(f"[{mode}]{label}")
                else:
                    others.append(label)
            lines = []
            if rendered:
                lines.append("  " + "  ".join(rendered))
            if others:
                lines.append("  另含: " + " ".join(others))
            body = "\n".join(lines)

        poi_hits = self.pois_by_hex.get(cur, [])
        if poi_hits:
            poi_text = []
            seen = set()
            for record in poi_hits:
                item = (
                    POI_CATEGORY_LABELS.get(record["category"], record["category"]),
                    record.get("name", "未命名"),
                )
                if item not in seen:
                    seen.add(item)
                    poi_text.append(f"[{item[0]}]{item[1]}")
            body += "\n  地点: " + "  ".join(poi_text)

        self.cell_info_text.set_text(f"当前栅格 {cur}:\n{body}")

    def show_label_prompt(self):
        """Show the label selection prompt after Enter is pressed."""
        self.ax.set_title(
            "SELECT LABEL:  " + _label_prompt_str(),
            fontsize=12, fontfamily="monospace", color="darkblue",
        )
        self.fig.canvas.draw_idle()

    def refresh(self):
        """Incremental update: path line + cursor position."""
        state = self.state
        xs = [p[0] for p in state.path_history]
        ys = [p[1] for p in state.path_history]
        zs = [p[2] for p in state.path_history]
        merc_x, merc_y = hex_to_mercator(xs, ys, zs)
        merc_x, merc_y = mercator_wgs84_to_gcj02(merc_x, merc_y)
        self.path_line.set_data(merc_x, merc_y)

        cursor_mx, cursor_my = hex_to_mercator(*state.cur)
        cursor_mx, cursor_my = mercator_wgs84_to_gcj02(cursor_mx, cursor_my)
        self.cursor.set_offsets([[cursor_mx, cursor_my]])
        self._update_title()
        self._update_cell_info()
        self.fig.canvas.draw_idle()


# ========================== Controller ==========================

class LabelController:
    """Coordinates state, renderer, and keyboard input."""

    def __init__(self, state: LabelState, renderer: PathRenderer,
                 output_dir: str, batch_mode: bool, current_idx: int,
                 start_in_label_mode: bool = False, navigate_callback=None):
        self.state = state
        self.renderer = renderer
        self.output_dir = output_dir
        self.batch_mode = batch_mode
        self.current_idx = current_idx
        self.saved = False
        self.selecting_label = False
        self.next_requested = False
        self.go_back_requested = False
        self.navigate_callback = navigate_callback

        if start_in_label_mode:
            self.selecting_label = True
            self.renderer.show_label_prompt()
            print(f"  Select label: " + " ".join(f"{k}={v}" for k, v in LABEL_OPTIONS.items()))
            print(f"  (Backspace to re-edit path)")

    def set_segment(self, state, current_idx, start_in_label_mode=False):
        """Point the existing controller at a newly rendered OD."""
        self.state = state
        self.current_idx = current_idx
        self.saved = False
        self.selecting_label = bool(start_in_label_mode)
        self.next_requested = False
        self.go_back_requested = False
        if self.selecting_label:
            self.renderer.show_label_prompt()
            print(f"  Select label: " + " ".join(f"{k}={v}" for k, v in LABEL_OPTIONS.items()))
            print(f"  (Backspace to re-edit path)")

    def on_key(self, event):
        global _NO_TRAJ_MODE
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
                    if self.navigate_callback is not None:
                        self.navigate_callback(1, False)
                    else:
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

        # N：切换"不保存轨迹"模式——开启后保存时 traj(路径)留空，其余字段照常
        if key == "n":
            _NO_TRAJ_MODE = not _NO_TRAJ_MODE
            self.renderer._update_title()
            self.renderer.fig.canvas.draw_idle()
            print(f"  [no-traj mode: {'ON' if _NO_TRAJ_MODE else 'OFF'}]")
            return

        if key not in HEX_KEY_MAP:
            return

        action, arg = HEX_KEY_MAP[key]

        if action == "move":
            if self.state.can_move(*HEX_DIRS[arg]):
                self.state.apply_move(arg)
                self.renderer.refresh()

        elif action == "undo":
            if len(self.state.path_history) <= 1:
                # No steps taken — go back to previous trajectory's label
                if self.current_idx > 0:
                    self.go_back_requested = True
                    print(f"  Going back to re-label previous trajectory #{self.current_idx - 1}")
                    if self.navigate_callback is not None:
                        self.navigate_callback(-1, True)
                    else:
                        plt.close(self.renderer.fig)
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
            print(f"  Select label: " + " ".join(f"{k}={v}" for k, v in LABEL_OPTIONS.items()))
            print(f"  (Backspace to cancel)")

    def _finalize(self, label):
        """Write the complete record (path + label) to CSV and PNG.

        相同 OD（uid + idx_o）采用覆盖方式更新，而非新增一行。
        """
        state = self.state
        os.makedirs(self.output_dir, exist_ok=True)

        csv_path = os.path.join(self.output_dir, "traj_labeled.csv")

        traj_list = [[int(p[0]), int(p[1]), int(p[2])] for p in state.path_history]
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
        if _NO_TRAJ_MODE:
            # 不保存轨迹模式：仅留空 traj，其余字段照常写入
            record["traj"] = ""
        else:
            record["traj"] = json.dumps(traj_list, ensure_ascii=False)
        record["mode"] = label

        key = _annotation_key(record)
        canonical_mode = _canonical_label_mode(label)
        if key is not None and canonical_mode is not None:
            self.renderer.labeled_modes[key] = canonical_mode

        df_new = pd.DataFrame([record])

        # 读取已有标注，按 OD（uid + idx_o）去重后再追加，实现"覆盖"语义
        if os.path.exists(csv_path):
            try:
                existing = pd.read_csv(csv_path, encoding="utf-8")
            except Exception:
                existing = pd.DataFrame()
        else:
            existing = pd.DataFrame()

        key_cols = [c for c in ["uid", "idx_o"] if c in df_new.columns]
        if not existing.empty and key_cols and all(c in existing.columns for c in key_cols):
            mask = np.ones(len(existing), dtype=bool)
            for c in key_cols:
                mask &= (existing[c].astype(str) == str(record[c]))
            existing = existing[~mask]

        out_df = (pd.concat([existing, df_new], ignore_index=True)
                  if not existing.empty else df_new)

        # uid, idx_o, idx_d 放前三列
        front_cols = [c for c in ["uid", "idx_o", "idx_d"] if c in out_df.columns]
        other_cols = [c for c in out_df.columns if c not in front_cols]
        out_df = out_df[front_cols + other_cols]
        out_df.to_csv(csv_path, index=False, encoding="utf-8")

        print(f"  -> CSV: {csv_path}")
        if not _NO_TRAJ_MODE:
            png_name = f"ep_{self.current_idx:04d}_order_{state.order}_{label}.png"
            png_path = os.path.join(self.output_dir, png_name)
            self.renderer.fig.savefig(png_path, bbox_inches="tight", dpi=150)
            print(f"  -> PNG: {png_path}")


# ========================== Main Loop ==========================

def run_single(state, raw_mapdata, output_dir, batch_mode, idx,
               start_in_label_mode=False, road_sets=None, traj_df=None, pois=None,
               labeled_modes=None):
    """Run labeling for one trajectory. Returns (next_idx, keep_going)."""
    renderer = PathRenderer(state, raw_mapdata, road_sets=road_sets,
                            traj_df=traj_df, current_idx=idx,
                            output_dir=output_dir, pois=pois,
                            labeled_modes=labeled_modes)
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
        print(f"Start: {state.start}  ->  End: {state.end}")
        print(f"Keys: W/A/S/D/Q/E=move  Backspace=undo  R=reset  Enter=save & label  N=no-traj")
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


def _print_segment_status(idx, state, relabel=False):
    if relabel:
        print(f"\n#{idx}  order={state.order}  mode={state.mode}  [RE-LABEL]")
        return
    print(f"\n{'='*60}")
    print(f"#{idx}  order={state.order}  mode={state.mode}")
    print(f"Start: {state.start}  ->  End: {state.end}")
    print("Keys: W/A/S/D/Q/E=move  Backspace=undo  R=reset  "
          "Enter=save & label  N=no-traj")
    print(f"{'='*60}")


def run_continuous(start_idx, make_state, raw_mapdata, output_dir,
                   road_sets, traj_df, pois, labeled_modes):
    """Run a batch in one persistent Matplotlib window.

    Static data and offline map tiles stay cached in the same process. Moving
    between ODs clears and redraws the two axes without destroying the GUI
    window or reconnecting keyboard/mouse handlers.
    """
    state = make_state(traj_df.iloc[start_idx])
    renderer = PathRenderer(
        state, raw_mapdata, road_sets=road_sets, traj_df=traj_df,
        current_idx=start_idx, output_dir=output_dir, pois=pois,
        labeled_modes=labeled_modes,
    )
    controller = LabelController(
        state, renderer, output_dir, batch_mode=True, current_idx=start_idx,
    )

    def navigate(delta, start_in_label_mode=False):
        target = controller.current_idx + int(delta)
        if target >= len(traj_df):
            controller.saved = True
            print(f"\nAll {len(traj_df)} trajectories labeled!")
            plt.close(renderer.fig)
            return
        if target < 0:
            print("  Already at first trajectory, cannot go back")
            return

        next_state = make_state(traj_df.iloc[target])
        renderer.show_segment(next_state, target)
        controller.set_segment(next_state, target, start_in_label_mode)
        _print_segment_status(target, next_state, relabel=start_in_label_mode)

    controller.navigate_callback = navigate
    renderer.fig.canvas.mpl_connect("key_press_event", controller.on_key)

    def on_close(event):
        if not controller.saved:
            print(f"  [WARN] window closed, #{controller.current_idx} not saved")

    renderer.fig.canvas.mpl_connect("close_event", on_close)
    _print_segment_status(start_idx, state)
    plt.show(block=True)


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
    parser.add_argument("--mapdata", type=str, default=HEX_PKL_PATH,
                        help="path to hex grid pickle file")
    parser.add_argument("--sample-step", type=int, default=1,
                        help="sampling interval for point-sequence CSV (default: 1)")
    parser.add_argument("--poi-data", type=str, default=DEFAULT_POI_PATH,
                        help="offline OSM POI GeoJSON cache")
    parser.add_argument("--no-pois", action="store_true",
                        help="do not display OSM subway/train/toll POIs")
    args = parser.parse_args()

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
    pois = []
    if not args.no_pois:
        try:
            pois = load_osm_pois(args.poi_data, hex_grid=raw_mapdata)
            if pois:
                print(f"Loaded OSM POIs: {len(pois):,}")
                for category, style in POI_STYLES.items():
                    count = sum(record["category"] == category for record in pois)
                    print(f"  {style['label']}: {count:,}")
            else:
                print(f"  [WARN] OSM POI cache not found or empty: {args.poi_data}")
                print("         Run: python download_osm_pois.py")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"  [WARN] Failed to load OSM POI cache: {exc}")

    # 匹配率基于所有可见路网分组的并集（不再按行内 mode 过滤）。
    # 该集合对所有 OD 都相同，只构建一次，避免每次切换重复复制大集合。
    multi_mapdata = set().union(*road_sets.values()) if road_sets else set()

    def make_state(row):
        return LabelState(row, multi_mapdata, hex_grid=raw_mapdata)

    output_dir = args.output
    labeled_modes = load_labeled_modes(output_dir)
    if labeled_modes:
        print(f"Loaded existing labels for map feedback: {len(labeled_modes):,}")

    if args.index is not None:
        if args.index < 0 or args.index >= len(traj_df):
            print(f"Error: index out of range [0, {len(traj_df)-1}]")
            sys.exit(1)
        state = make_state(traj_df.iloc[args.index])
        run_single(state, raw_mapdata, output_dir, batch_mode=False,
                   idx=args.index, road_sets=road_sets, traj_df=traj_df, pois=pois,
                   labeled_modes=labeled_modes)

    else:
        if args.batch:
            start_idx = 0
        else:
            # Prompt user for starting index or uid.
            while True:
                try:
                    user_input = input(
                        f"Enter starting index [0-{len(traj_df)-1}], "
                        f"or uid (e.g. u123), or press Enter for 0: "
                    ).strip()
                    if user_input == "":
                        start_idx = 0
                        break
                    if user_input.lower().startswith("u"):
                        uid_val = int(user_input[1:])
                        if "uid" not in traj_df.columns:
                            print("  Error: CSV has no uid column")
                            continue
                        matches = traj_df.index[traj_df["uid"] == uid_val].tolist()
                        if not matches:
                            print(f"  Error: uid {uid_val} not found")
                            continue
                        start_idx = int(matches[0])
                        print(f"  uid {uid_val} -> index {start_idx}")
                        break
                    start_idx = int(user_input)
                    if 0 <= start_idx < len(traj_df):
                        break
                    print(f"  Error: index out of range [0, {len(traj_df)-1}]")
                except ValueError:
                    print("  Error: please enter a valid integer (index or u<uid>)")
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting.")
                    sys.exit(0)

        run_continuous(
            start_idx, make_state, raw_mapdata, output_dir,
            road_sets, traj_df, pois, labeled_modes,
        )


if __name__ == "__main__":
    main()
