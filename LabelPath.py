"""
Interactive mode/path labeling tool. In path workflow, click a start and end
hex to preview an automatic mode-network route, then press Enter to confirm.

Usage:
  python LabelPath.py                     # start from index 0, label one by one
  python LabelPath.py --index 5           # label a single trajectory at index 5
  python LabelPath.py --batch             # label all trajectories in sequence
  python LabelPath.py --output my_labels  # custom output directory
"""

import os
import sys
import json
import heapq
import re
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.widgets import Button, TextBox

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
    gcj02_to_wgs84,
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
    DISPLAY_MODE_LIST,
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
PATH_LABEL_FILENAME = "path_labeled.csv"
PATH_LABEL_COLUMNS = (
    "uid", "segment_id", "mode", "anchor_idx_o",
    "start_x", "start_y", "start_z", "end_x", "end_y", "end_z",
    "match", "steps", "traj",
)
IGNORED_POINT_FILENAME = "ignored_points.csv"
EXCLUDED_UID_FILENAME = "excluded_uids.csv"

DISTANCE_THRESHOLD = 1.0

ANNOTATION_MODE = "mode"
ANNOTATION_PATH = "path"
PATH_SNAP_MAX_RADIUS = 8
ROUTE_SEARCH_MAX_NODES = 300_000


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

ROAD_LAYER_COLORS = {
    **MODE_COLORS,
    "L2": (0.55, 0.38, 0.20),  # 二级公路 棕
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
UID_LIST_VISIBLE_ROWS = 28
UID_LIST_SCROLL_STEP = 5
UID_NAV_VISIBLE_ROWS = 28
UID_NAV_SCROLL_STEP = 5
PATH_LIST_VISIBLE_ROWS = 28
PATH_LIST_SCROLL_STEP = 5

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


def _annotation_key(row):
    """Return the stable (uid, idx_o) key shared by source and label CSVs."""
    try:
        return int(row["uid"]), int(row["idx_o"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _destination_annotation_key(row):
    """Return the stable key of an OD destination sampled point."""
    try:
        return int(row["uid"]), int(row["idx_d"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def segment_is_reviewable(traj_df, index, ignored_points=None):
    """Return whether an OD still has a non-ignored origin and endpoint.

    Ignored intermediate points are crossed by the existing effective-segment
    merge. An ignored terminal point is different: there is no later sampled
    point to merge to, so the final OD must be skipped until that point is
    restored.
    """
    ignored_points = ignored_points or set()
    if traj_df is None or not 0 <= int(index) < len(traj_df):
        return False
    cursor = int(index)
    row = traj_df.iloc[cursor]
    origin_key = _annotation_key(row)
    if origin_key is None or origin_key in ignored_points:
        return False
    uid = origin_key[0]

    while True:
        destination_key = _destination_annotation_key(row)
        if destination_key is None or destination_key not in ignored_points:
            return True
        next_cursor = cursor + 1
        if next_cursor >= len(traj_df):
            return False
        next_row = traj_df.iloc[next_cursor]
        if int(next_row["uid"]) != uid \
                or _annotation_key(next_row) != destination_key:
            return False
        cursor = next_cursor
        row = next_row


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


def _parse_saved_path(value):
    """Parse one JSON path cell into validated integer hex coordinates."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        raw_path = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(raw_path, list) or not raw_path:
        return None
    path = []
    for point in raw_path:
        if not isinstance(point, (list, tuple)) or len(point) != 3:
            return None
        try:
            path.append(tuple(int(value) for value in point))
        except (TypeError, ValueError, OverflowError):
            return None
    return path


def load_labeled_path_data(output_dir):
    """Load user-defined truth path segments and their selected road modes."""
    csv_path = os.path.join(output_dir, PATH_LABEL_FILENAME)
    if not os.path.exists(csv_path):
        return {}, {}
    try:
        labeled = pd.read_csv(csv_path, encoding="utf-8")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"  [WARN] Failed to load existing paths: {exc}")
        return {}, {}
    required = {"uid", "segment_id", "mode", "traj"}
    if not required.issubset(labeled.columns):
        return {}, {}

    paths = {}
    modes = {}
    for _, row in labeled.iterrows():
        mode = _canonical_label_mode(row["mode"])
        try:
            key = int(row["uid"]), int(row["segment_id"])
        except (TypeError, ValueError, OverflowError):
            continue
        if mode not in MODE_LIST:
            continue
        path = _parse_saved_path(row["traj"])
        if path is not None:
            paths[key] = path
            modes[key] = mode
    return paths, modes


def load_labeled_paths(output_dir, valid_segments=None):
    """Compatibility wrapper returning user-defined truth path geometries."""
    paths, _ = load_labeled_path_data(output_dir)
    return paths


def load_labeled_path_metadata(output_dir):
    """Load lightweight fields used by the current-UID saved-path list."""
    csv_path = os.path.join(output_dir, PATH_LABEL_FILENAME)
    if not os.path.exists(csv_path):
        return {}
    try:
        labeled = pd.read_csv(csv_path, encoding="utf-8")
    except (OSError, ValueError, pd.errors.ParserError):
        return {}
    required = {"uid", "segment_id", "anchor_idx_o", "steps"}
    if not required.issubset(labeled.columns):
        return {}
    metadata = {}
    for _, row in labeled.iterrows():
        try:
            key = int(row["uid"]), int(row["segment_id"])
            metadata[key] = {
                "anchor_idx_o": int(row["anchor_idx_o"]),
                "steps": int(row["steps"]),
                "match": float(row.get("match", np.nan)),
            }
        except (TypeError, ValueError, OverflowError):
            continue
    return metadata


def load_ignored_points(output_dir):
    """Load persisted sampled-point keys excluded from OD construction."""
    csv_path = os.path.join(output_dir, IGNORED_POINT_FILENAME)
    if not os.path.exists(csv_path):
        return set()
    try:
        ignored = pd.read_csv(csv_path, encoding="utf-8")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"  [WARN] Failed to load ignored points: {exc}")
        return set()
    if not {"uid", "idx"}.issubset(ignored.columns):
        print("  [WARN] Ignored point file lacks uid/idx columns")
        return set()
    result = set()
    for uid, point_idx in ignored[["uid", "idx"]].itertuples(index=False, name=None):
        try:
            result.add((int(uid), int(point_idx)))
        except (TypeError, ValueError, OverflowError):
            continue
    return result


def save_ignored_points(output_dir, ignored_points):
    """Persist ignored sampled-point keys atomically."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, IGNORED_POINT_FILENAME)
    temp_path = csv_path + ".tmp"
    rows = sorted((int(uid), int(point_idx)) for uid, point_idx in ignored_points)
    pd.DataFrame(rows, columns=["uid", "idx"]).to_csv(
        temp_path, index=False, encoding="utf-8",
    )
    os.replace(temp_path, csv_path)


def load_excluded_uids(output_dir):
    """Load UID trajectories omitted from the dated labeled-data copy."""
    csv_path = os.path.join(output_dir, EXCLUDED_UID_FILENAME)
    if not os.path.exists(csv_path):
        return set()
    try:
        excluded = pd.read_csv(csv_path, encoding="utf-8")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"  [WARN] Failed to load excluded UIDs: {exc}")
        return set()
    if "uid" not in excluded.columns:
        print("  [WARN] Excluded UID file lacks uid column")
        return set()
    result = set()
    for uid in excluded["uid"]:
        try:
            result.add(int(uid))
        except (TypeError, ValueError, OverflowError):
            continue
    return result


def save_excluded_uids(output_dir, excluded_uids):
    """Persist excluded UID trajectories atomically."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, EXCLUDED_UID_FILENAME)
    temp_path = csv_path + ".tmp"
    pd.DataFrame(
        sorted(int(uid) for uid in excluded_uids), columns=["uid"],
    ).to_csv(temp_path, index=False, encoding="utf-8")
    os.replace(temp_path, csv_path)


def derive_labeled_data_date(csv_path, traj_df=None):
    """Derive YYYYMMDD from the source filename, then from trajectory UIDs."""
    for match in re.findall(r"(?<!\d)(20\d{6})(?!\d)", os.path.basename(csv_path)):
        if not pd.isna(pd.to_datetime(match, format="%Y%m%d", errors="coerce")):
            return match
    if traj_df is not None and not traj_df.empty and "uid" in traj_df.columns:
        for uid in traj_df["uid"]:
            try:
                candidate = str(int(uid))[:8]
            except (TypeError, ValueError, OverflowError):
                continue
            if len(candidate) == 8 and not pd.isna(pd.to_datetime(
                    candidate, format="%Y%m%d", errors="coerce")):
                return candidate
    return pd.Timestamp.now().strftime("%Y%m%d")


def write_labeled_data_copy(output_dir, data_date, excluded_uids):
    """Write the sorted active-label copy without excluded UID trajectories."""
    source_path = os.path.join(output_dir, "traj_labeled.csv")
    export_path = os.path.join(output_dir, f"labeled_data_{data_date}.csv")
    if not os.path.exists(source_path):
        return export_path, 0
    labeled = pd.read_csv(source_path, encoding="utf-8")
    if excluded_uids and "uid" in labeled.columns:
        uid_values = pd.to_numeric(labeled["uid"], errors="coerce")
        labeled = labeled.loc[
            ~uid_values.isin(tuple(int(uid) for uid in excluded_uids))
        ]
    labeled = sort_labeled_records(labeled)
    export_temp = export_path + ".tmp"
    labeled.to_csv(export_temp, index=False, encoding="utf-8")
    os.replace(export_temp, export_path)
    return export_path, len(labeled)


def write_path_labeled_copy(output_dir, data_date, excluded_uids):
    """Write the dated user-defined path copy without excluded UIDs."""
    source_path = os.path.join(output_dir, PATH_LABEL_FILENAME)
    export_path = os.path.join(output_dir, f"path_labeled_{data_date}.csv")
    if not os.path.exists(source_path):
        return export_path, 0
    labeled = pd.read_csv(source_path, encoding="utf-8")
    if excluded_uids and "uid" in labeled.columns:
        uid_values = pd.to_numeric(labeled["uid"], errors="coerce")
        labeled = labeled.loc[
            ~uid_values.isin(tuple(int(uid) for uid in excluded_uids))
        ]
    sort_columns = [column for column in ("uid", "segment_id", "anchor_idx_o")
                    if column in labeled.columns]
    if sort_columns:
        labeled = labeled.sort_values(sort_columns, kind="mergesort")
    export_temp = export_path + ".tmp"
    labeled.to_csv(export_temp, index=False, encoding="utf-8")
    os.replace(export_temp, export_path)
    return export_path, len(labeled)


def write_truth_path_segment(output_dir, uid, mode, anchor_idx_o,
                             path, match, steps, segment_key=None):
    """Append a truth segment, or replace one selected segment in place."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, PATH_LABEL_FILENAME)
    if os.path.exists(csv_path):
        try:
            existing = pd.read_csv(csv_path, encoding="utf-8")
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame(columns=PATH_LABEL_COLUMNS)
    else:
        existing = pd.DataFrame(columns=PATH_LABEL_COLUMNS)
    existing = existing.reindex(columns=PATH_LABEL_COLUMNS)
    uid = int(uid)
    uid_values = pd.to_numeric(existing["uid"], errors="coerce")
    segment_values = pd.to_numeric(existing["segment_id"], errors="coerce")
    replace_mask = np.zeros(len(existing), dtype=bool)
    if segment_key is not None:
        selected_uid, segment_id = (int(value) for value in segment_key)
        if selected_uid != uid:
            raise ValueError("Selected path belongs to a different UID")
        replace_mask = (
            (uid_values == selected_uid) & (segment_values == segment_id)
        ).to_numpy()
        if not replace_mask.any():
            raise ValueError(
                f"Selected path uid={selected_uid} segment={segment_id} no longer exists"
            )
        existing = existing.loc[~replace_mask].copy()
    elif existing.empty:
        segment_id = 1
    else:
        uid_segments = segment_values.loc[uid_values == uid].dropna()
        segment_id = int(uid_segments.max()) + 1 if len(uid_segments) else 1

    path_values = [[int(value) for value in point] for point in path]
    start = path_values[0]
    end = path_values[-1]
    record = pd.DataFrame([{
        "uid": uid,
        "segment_id": segment_id,
        "mode": mode,
        "anchor_idx_o": int(anchor_idx_o),
        "start_x": start[0], "start_y": start[1], "start_z": start[2],
        "end_x": end[0], "end_y": end[1], "end_z": end[2],
        "match": float(match),
        "steps": int(steps),
        "traj": json.dumps(path_values, ensure_ascii=False),
    }])
    output = (
        record.copy() if existing.empty
        else pd.concat([existing, record], ignore_index=True)
    )
    output = output.reindex(columns=PATH_LABEL_COLUMNS)
    output = output.sort_values(
        ["uid", "segment_id"], kind="mergesort",
    ).reset_index(drop=True)
    temp_path = csv_path + ".tmp"
    output.to_csv(temp_path, index=False, encoding="utf-8")
    os.replace(temp_path, csv_path)
    return csv_path, (uid, segment_id)


def delete_truth_path_segment(output_dir, segment_key):
    """Delete one truth segment and compact that UID's segment IDs to 1..N."""
    csv_path = os.path.join(output_dir, PATH_LABEL_FILENAME)
    if not os.path.exists(csv_path):
        return csv_path, 0
    try:
        existing = pd.read_csv(csv_path, encoding="utf-8")
    except pd.errors.EmptyDataError:
        return csv_path, 0
    existing = existing.reindex(columns=PATH_LABEL_COLUMNS)
    selected_uid, segment_id = (int(value) for value in segment_key)
    uid_values = pd.to_numeric(existing["uid"], errors="coerce")
    segment_values = pd.to_numeric(existing["segment_id"], errors="coerce")
    delete_mask = (
        (uid_values == selected_uid) & (segment_values == segment_id)
    ).to_numpy()
    deleted_count = int(delete_mask.sum())
    if not deleted_count:
        return csv_path, 0

    output = existing.loc[~delete_mask].reindex(columns=PATH_LABEL_COLUMNS).copy()
    output_uid_values = pd.to_numeric(output["uid"], errors="coerce")
    output_segment_values = pd.to_numeric(
        output["segment_id"], errors="coerce",
    )
    uid_row_indices = output.index[output_uid_values == selected_uid].tolist()
    uid_row_indices.sort(key=lambda index: (
        pd.isna(output_segment_values.loc[index]),
        output_segment_values.loc[index]
        if not pd.isna(output_segment_values.loc[index]) else float("inf"),
        index,
    ))
    if uid_row_indices:
        output.loc[uid_row_indices, "segment_id"] = np.arange(
            1, len(uid_row_indices) + 1,
        )
    output = output.sort_values(
        ["uid", "segment_id"], kind="mergesort",
    ).reset_index(drop=True)
    temp_path = csv_path + ".tmp"
    output.to_csv(temp_path, index=False, encoding="utf-8")
    os.replace(temp_path, csv_path)
    return csv_path, deleted_count


def normalize_path_label_storage(output_dir):
    """Normalize only user-defined path records; never infer boundaries from ODs."""
    csv_path = os.path.join(output_dir, PATH_LABEL_FILENAME)
    if not os.path.exists(csv_path):
        return 0
    try:
        labeled = pd.read_csv(csv_path, encoding="utf-8")
    except pd.errors.EmptyDataError:
        return 0
    required = {"uid", "segment_id", "mode", "traj"}
    if labeled.empty:
        if tuple(labeled.columns) != PATH_LABEL_COLUMNS:
            pd.DataFrame(columns=PATH_LABEL_COLUMNS).to_csv(
                csv_path, index=False, encoding="utf-8",
            )
            return 1
        return 0
    if not required.issubset(labeled.columns):
        return 0
    keep = []
    for _, row in labeled.iterrows():
        mode = _canonical_label_mode(row.get("mode"))
        try:
            int(row["uid"])
            int(row["segment_id"])
        except (TypeError, ValueError, OverflowError):
            keep.append(False)
            continue
        keep.append(mode in MODE_LIST and _parse_saved_path(row.get("traj")) is not None)
    keep = np.asarray(keep, dtype=bool)
    removed = int((~keep).sum())
    active = labeled.loc[keep].reindex(columns=PATH_LABEL_COLUMNS).copy()
    active = active.sort_values(["uid", "segment_id"], kind="mergesort")
    changed = removed or list(labeled.columns) != list(PATH_LABEL_COLUMNS)
    if changed:
        temp_path = csv_path + ".tmp"
        active.to_csv(temp_path, index=False, encoding="utf-8")
        os.replace(temp_path, csv_path)
    return removed


def remove_labeled_keys(output_dir, keys):
    """Remove OD labels made invalid by an ignored/restored sampled point."""
    keys = {tuple(map(int, key)) for key in keys if key is not None}
    if not keys or not output_dir:
        return 0
    csv_path = os.path.join(output_dir, "traj_labeled.csv")
    if not os.path.exists(csv_path):
        return 0
    labeled = pd.read_csv(csv_path, encoding="utf-8")
    if labeled.empty or not {"uid", "idx_o"}.issubset(labeled.columns):
        return 0
    uid_values = pd.to_numeric(labeled["uid"], errors="coerce")
    idx_values = pd.to_numeric(labeled["idx_o"], errors="coerce")
    invalid_mask = np.fromiter(
        (
            not pd.isna(uid) and not pd.isna(idx_o)
            and (int(uid), int(idx_o)) in keys
            for uid, idx_o in zip(uid_values, idx_values)
        ),
        dtype=bool, count=len(labeled),
    )
    if not invalid_mask.any():
        return 0

    active = labeled.loc[~invalid_mask]
    active_temp = csv_path + ".tmp"
    active.to_csv(active_temp, index=False, encoding="utf-8")
    os.replace(active_temp, csv_path)
    return int(invalid_mask.sum())


def sort_labeled_records(labeled):
    """Return active labels in deterministic source-trajectory order."""
    if labeled is None or labeled.empty:
        return labeled
    records = labeled.reset_index(drop=True)
    sort_columns = [column for column in ("uid", "idx_o", "idx_d")
                    if column in records.columns]
    if not sort_columns:
        return records
    helper = pd.DataFrame(index=records.index)
    helper["_position"] = np.arange(len(records), dtype=np.int64)
    helper_columns = []
    for column in sort_columns:
        helper_column = f"_sort_{column}"
        helper[helper_column] = pd.to_numeric(records[column], errors="coerce")
        helper_columns.append(helper_column)
    order = helper.sort_values(
        helper_columns + ["_position"], kind="mergesort", na_position="last",
    ).index
    return records.iloc[order].reset_index(drop=True)


def normalize_label_storage(output_dir):
    """Keep OD mode truth separate from all path-specific fields."""
    csv_path = os.path.join(output_dir, "traj_labeled.csv")
    if not os.path.exists(csv_path):
        return 0
    active = pd.read_csv(csv_path, encoding="utf-8")
    path_columns = [
        column for column in ("path", "traj", "success", "match", "steps")
        if column in active.columns
    ]
    if path_columns:
        active = active.drop(columns=path_columns)
    ordered = sort_labeled_records(active)
    changed = int(
        bool(path_columns)
        or not ordered.equals(active.reset_index(drop=True))
    )
    if changed:
        active_temp = csv_path + ".tmp"
        ordered.to_csv(active_temp, index=False, encoding="utf-8")
        os.replace(active_temp, csv_path)
    return changed


def first_unlabeled_index(traj_df, labeled_modes, ignored_points=None,
                          excluded_uids=None):
    """Return the first unfinished OD, then the first reviewable OD."""
    ignored_points = ignored_points or set()
    excluded_uids = excluded_uids or set()
    if traj_df is None or traj_df.empty:
        return 0
    if not {"uid", "idx_o"}.issubset(traj_df.columns):
        return 0
    uid_values = pd.to_numeric(traj_df["uid"], errors="coerce").to_numpy()
    idx_o_values = pd.to_numeric(traj_df["idx_o"], errors="coerce").to_numpy()
    for position, (uid, idx_o) in enumerate(zip(uid_values, idx_o_values)):
        if not pd.isna(uid) and int(uid) in excluded_uids:
            continue
        key = None if pd.isna(uid) or pd.isna(idx_o) else (int(uid), int(idx_o))
        if not segment_is_reviewable(traj_df, position, ignored_points):
            continue
        if key is None or key not in labeled_modes:
            return position
    for position, (uid, idx_o) in enumerate(zip(uid_values, idx_o_values)):
        if not pd.isna(uid) and int(uid) in excluded_uids:
            continue
        key = None if pd.isna(uid) or pd.isna(idx_o) else (int(uid), int(idx_o))
        if segment_is_reviewable(traj_df, position, ignored_points):
            return position
    return 0


def nearest_road_hex(point, road_cells, max_radius=PATH_SNAP_MAX_RADIUS):
    """Snap a clicked hex to the nearest cell in the active mode road set."""
    if not road_cells:
        return None
    point = tuple(int(value) for value in point)
    if point in road_cells:
        return point
    x0, y0, z0 = point
    for radius in range(1, int(max_radius) + 1):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                dz = -dx - dy
                if max(abs(dx), abs(dy), abs(dz)) != radius:
                    continue
                candidate = (x0 + dx, y0 + dy, z0 + dz)
                if candidate in road_cells:
                    return candidate
    return None


def find_road_route(start, end, road_cells, max_nodes=ROUTE_SEARCH_MAX_NODES):
    """Find a shortest six-neighbour hex route on one mode-specific road set."""
    if not road_cells:
        return None
    start = tuple(int(value) for value in start)
    end = tuple(int(value) for value in end)
    if start not in road_cells or end not in road_cells:
        return None
    if start == end:
        return [start]

    frontier = [(hex_distance(start, end), 0, start)]
    best_cost = {start: 0}
    previous = {start: None}
    expanded = 0
    while frontier:
        _, cost, current = heapq.heappop(frontier)
        if cost != best_cost.get(current):
            continue
        if current == end:
            route = []
            cursor = current
            while cursor is not None:
                route.append(cursor)
                cursor = previous[cursor]
            route.reverse()
            return route
        expanded += 1
        if expanded > int(max_nodes):
            return None
        for dx, dy, dz in HEX_DIRS.values():
            neighbour = (
                current[0] + dx, current[1] + dy, current[2] + dz,
            )
            if neighbour not in road_cells:
                continue
            next_cost = cost + 1
            if next_cost >= best_cost.get(neighbour, sys.maxsize):
                continue
            best_cost[neighbour] = next_cost
            previous[neighbour] = current
            priority = next_cost + hex_distance(neighbour, end)
            heapq.heappush(frontier, (priority, next_cost, neighbour))
    return None


def next_nonignored_index(traj_df, ignored_points, current_idx, direction,
                          excluded_uids=None):
    """Return the next global row not ignored, or an out-of-range sentinel."""
    excluded_uids = excluded_uids or set()
    step = 1 if int(direction) >= 0 else -1
    target = int(current_idx) + step
    while 0 <= target < len(traj_df):
        row = traj_df.iloc[target]
        if int(row["uid"]) not in excluded_uids \
                and segment_is_reviewable(traj_df, target, ignored_points):
            return target
        target += step
    return len(traj_df) if step > 0 else -1


def effective_segment_row(traj_df, index, ignored_points=None):
    """Merge OD intervals across ignored intermediate sampled points."""
    ignored_points = ignored_points or set()
    index = int(index)
    row = traj_df.iloc[index].copy()
    uid = int(row["uid"])
    cursor = index
    total_time = float(pd.to_numeric(pd.Series([row.get("time_d", 0)]), errors="coerce").fillna(0).iloc[0])
    total_dist = float(pd.to_numeric(pd.Series([row.get("dist_d", 0)]), errors="coerce").fillna(0).iloc[0])
    last_row = row

    while cursor + 1 < len(traj_df):
        candidate = traj_df.iloc[cursor + 1]
        if int(candidate["uid"]) != uid or _annotation_key(candidate) not in ignored_points:
            break
        cursor += 1
        last_row = candidate
        total_time += float(pd.to_numeric(pd.Series([candidate.get("time_d", 0)]), errors="coerce").fillna(0).iloc[0])
        total_dist += float(pd.to_numeric(pd.Series([candidate.get("dist_d", 0)]), errors="coerce").fillna(0).iloc[0])

    if cursor != index:
        for column in ("x_d", "y_d", "z_d", "idx_d"):
            if column in row.index and column in last_row.index:
                row[column] = last_row[column]
        row["time_d"] = total_time
        row["dist_d"] = total_dist
        fallback_velocity = float(last_row.get("velocity_d", row.get("velocity_d", 0)) or 0)
        row["velocity_d"] = (
            total_dist / total_time * 3.6
            if total_time > 0 and total_dist > 0 else fallback_velocity
        )
    return row

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
        if self.cur is None:
            return False
        return hex_distance(self.cur, self.end) <= DISTANCE_THRESHOLD

    @property
    def remaining_dist(self):
        if self.cur is None:
            return None
        return hex_distance(self.cur, self.end)

    def current_match_rate(self):
        if len(self.path_history) <= 1:
            return 0.0
        return calculate_match_rate_hex(self.path_history, self.multi_mapdata)

    def can_move(self, dx, dy, dz):
        if self.cur is None:
            return False
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

    def clear_path(self):
        """Clear the manual path so its start can be selected on the map."""
        self.cur = None
        self.path_history = []
        self.step_count = 0

    def set_path_start(self, point):
        """Start a manual path at a user-selected hex, independent of signals."""
        point = tuple(int(value) for value in point)
        if self.hex_grid is not None and not hex_in_map(*point, self.hex_grid):
            return False
        self.cur = point
        self.path_history = [point]
        self.step_count = 0
        return True

    def restore_path(self, path):
        """Restore a previously saved path when revisiting an OD."""
        normalized = [tuple(int(value) for value in point) for point in (path or [])]
        if not normalized:
            self.clear_path()
            return False
        self.path_history = normalized
        self.cur = normalized[-1]
        self.step_count = max(0, len(normalized) - 1)
        return True


# ========================== Renderer ==========================

class PathRenderer:
    """Manages the matplotlib figure and incremental updates (hex)."""

    def __init__(self, state: LabelState, raw_mapdata, road_sets=None,
                 traj_df=None, current_idx=None, output_dir=None, pois=None,
                 labeled_modes=None, labeled_paths=None, labeled_path_modes=None,
                 labeled_path_metadata=None,
                 ignored_points=None, excluded_uids=None, export_date=None):
        self.raw_mapdata = raw_mapdata
        self.road_sets = road_sets
        self.traj_df = traj_df
        self.output_dir = output_dir
        self.labeled_modes = labeled_modes if labeled_modes is not None else {}
        self.labeled_paths = labeled_paths if labeled_paths is not None else {}
        self.labeled_path_modes = (
            labeled_path_modes if labeled_path_modes is not None else {}
        )
        self.labeled_path_metadata = (
            labeled_path_metadata if labeled_path_metadata is not None else {}
        )
        self.ignored_points = ignored_points if ignored_points is not None else set()
        self.excluded_uids = excluded_uids if excluded_uids is not None else set()
        self.export_date = export_date
        self.pois = pois or []
        self.pois_by_hex = group_pois_by_hex(self.pois)
        self.road_visibility = {
            mode: mode != "L2" for mode in DISPLAY_MODE_LIST
        }
        self._road_artists = {}
        self._road_buttons = {}
        self._label_buttons = {}
        self._annotation_toggle_button = None
        self.annotation_mode = ANNOTATION_MODE
        self.active_path_mode = None
        self._mode_workflow_road_visibility = None
        self.segment_select_callback = None
        self.label_select_callback = None
        self.annotation_toggle_callback = None
        self.path_start_select_callback = None
        self._uid_list_uid = None
        self._uid_list_offset = 0
        self._uid_list_hit_rows = []
        self._uid_nav_offset = 0
        self._uid_nav_hit_rows = []
        self._uid_search_box = None
        self._path_list_uid = None
        self._path_list_offset = 0
        self._path_list_hit_rows = []
        self.selected_path_key = None
        self.editing_path_key = None
        self._uid_point_sequence_cache = {}
        self._prepare_static_indexes()

        # 六栏：UID、当前 UID 已标路径、点/OD、地图、信息栏、速度分布。
        self.fig = plt.figure(figsize=(20, 9))
        gs = self.fig.add_gridspec(
            1, 6, width_ratios=[0.64, 0.68, 0.70, 3, 0.92, 1], wspace=0.025,
        )
        self.ax_uid_nav = self.fig.add_subplot(gs[0])
        self.ax_path_list = self.fig.add_subplot(gs[1])
        self.ax_uid_list = self.fig.add_subplot(gs[2])
        self.ax = self.fig.add_subplot(gs[3])
        self.ax_info = self.fig.add_subplot(gs[4])
        self.ax_hist = self.fig.add_subplot(gs[5])
        self.fig.canvas.manager.set_window_title("LabelPath — Hex Grid")
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_poi_hover)
        self.fig.canvas.mpl_connect("scroll_event", self._on_uid_list_scroll)
        self.fig.canvas.mpl_connect("scroll_event", self._on_uid_nav_scroll)
        self.fig.canvas.mpl_connect("scroll_event", self._on_path_list_scroll)
        self.fig.canvas.mpl_connect("button_press_event", self._on_uid_list_click)
        self.fig.canvas.mpl_connect("button_press_event", self._on_uid_nav_click)
        self.fig.canvas.mpl_connect("button_press_event", self._on_path_list_click)
        self.fig.canvas.mpl_connect("button_press_event", self._on_main_view_click)
        self.show_segment(state, current_idx, initial=True)
        # Bottom is reserved only for direct mode labels; road switches live in
        # the right-side legend card so the two actions cannot be confused.
        self.fig.subplots_adjust(bottom=0.115)
        self._init_uid_search_box()
        self._init_road_toggle_buttons()
        self._init_label_buttons()
        self._init_annotation_toggle_button()
        self.fig.canvas.draw_idle()

    def _prepare_static_indexes(self):
        """Project and index immutable layers once for the whole annotation run."""
        self._road_display = {}
        if self.road_sets:
            for mode_name in DISPLAY_MODE_LIST:
                if mode_name != "L2":
                    self._prepare_road_display_mode(mode_name)

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
            self._uid_nav_values = np.asarray(
                tuple(self._uid_segment_positions.keys()), dtype=np.int64,
            )
            self._uid_nav_index = {
                int(uid): index for index, uid in enumerate(self._uid_nav_values)
            }
            self._uid_resolved_counts = {uid: 0 for uid in self._uid_segment_positions}
            self._uid_path_counts = {uid: 0 for uid in self._uid_segment_positions}
            for uid, _segment_id in self.labeled_paths:
                uid = int(uid)
                if uid in self._uid_path_counts:
                    self._uid_path_counts[uid] += 1
            resolved_keys = set(self.labeled_modes) | set(self.ignored_points)
            for uid, positions in self._uid_segment_positions.items():
                self._uid_resolved_counts[uid] = sum(
                    _annotation_key(self.traj_df.iloc[int(position)])
                    in resolved_keys
                    or not segment_is_reviewable(
                        self.traj_df, int(position), self.ignored_points,
                    )
                    for position in positions
                )
        else:
            self._uid_segment_positions = {}
            self._uid_nav_values = np.empty(0, dtype=np.int64)
            self._uid_nav_index = {}
            self._uid_resolved_counts = {}
            self._uid_path_counts = {}

    def _prepare_road_display_mode(self, mode_name):
        """Project one road layer once; L2 uses this lazily on first enable."""
        if mode_name in self._road_display or not self.road_sets:
            return
        road_set = self.road_sets.get(mode_name)
        if not road_set:
            return
        coordinates = np.asarray(tuple(road_set), dtype=np.int32)
        mx, my = hex_to_mercator(
            coordinates[:, 0], coordinates[:, 1], coordinates[:, 2],
        )
        gx, gy = mercator_wgs84_to_gcj02(mx, my)
        self._road_display[mode_name] = (
            np.asarray(mx), np.asarray(my), np.asarray(gx), np.asarray(gy),
        )

    def show_segment(self, state, current_idx, initial=False):
        """Render another OD in the existing figure instead of reopening it."""
        self.state = state
        self.current_idx = current_idx
        if self.selected_path_key is not None \
                and int(self.selected_path_key[0]) != int(state.uid):
            self.selected_path_key = None
        if self.editing_path_key is not None \
                and int(self.editing_path_key[0]) != int(state.uid):
            self.editing_path_key = None
        self._poi_scatter_meta = []
        self._poi_hover = None
        self._point_scatter_meta = []
        self._point_hover = None
        self._road_hover = None
        self._road_artists = {}
        self.ax.clear()
        self._draw_uid_navigation_list(ensure_current=True)
        self._draw_uid_path_list(ensure_current=True)
        self._draw_uid_segment_list(ensure_current=True)

        self._init_hex_view(state, self.raw_mapdata)

        # Coordinates are not needed during annotation; keep the map uncluttered.
        self.ax.set_xlabel("")
        self.ax.set_ylabel("")
        self.ax.tick_params(
            axis="both", which="both", left=False, bottom=False,
            labelleft=False, labelbottom=False,
        )
        self.ax.xaxis.offsetText.set_visible(False)
        self.ax.yaxis.offsetText.set_visible(False)
        self.ax.grid(False)

        # Road colors are shown by the vertical toggle buttons in the legend
        # card. The conventional legend only needs the cached OSM POIs.
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
        self.ax_info.clear()
        self.ax_info.axis("off")
        self._init_info_panel()
        if poi_handles:
            self.ax_info.legend(
                handles=poi_handles, loc="upper left",
                bbox_to_anchor=(0.055, 0.088), borderaxespad=0.0,
                fontsize=6.8, handlelength=1.4, borderpad=0.0,
                labelspacing=0.20, frameon=False,
            )

        self._update_title()
        self._draw_legend_box()
        self._draw_segment_info()
        self._init_cell_info()
        self._update_annotation_controls()
        self._sync_path_artists()
        if initial:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="This figure includes Axes that are not compatible",
                )
                self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def set_segment_select_callback(self, callback):
        """Set the absolute-row navigation callback used by the UID list."""
        self.segment_select_callback = callback

    def set_label_select_callback(self, callback):
        """Set the callback shared by the clickable mode-label buttons."""
        self.label_select_callback = callback

    def set_annotation_toggle_callback(self, callback):
        """Set the callback used by the mode/path workflow switch."""
        self.annotation_toggle_callback = callback

    def set_path_start_select_callback(self, callback):
        """Set the callback for choosing a manual path start on the map."""
        self.path_start_select_callback = callback

    def set_labeled_mode(self, key, mode):
        """Update one saved label and the UID completion counter."""
        is_new_resolution = key not in self.labeled_modes and key not in self.ignored_points
        self.labeled_modes[key] = mode
        if is_new_resolution and key[0] in self._uid_resolved_counts:
            self._uid_resolved_counts[key[0]] += 1

    def set_labeled_path(self, key, path, mode, metadata=None):
        """Update one saved path and the UID path-completion counter."""
        is_new_path = key not in self.labeled_paths
        self.labeled_paths[key] = [tuple(point) for point in path]
        self.labeled_path_modes[key] = mode
        if metadata is not None:
            self.labeled_path_metadata[key] = dict(metadata)
        self.selected_path_key = key
        if is_new_path and key[0] in self._uid_path_counts:
            self._uid_path_counts[key[0]] += 1

    def clear_labeled_path(self, key):
        """Discard a path invalidated by a changed mode or ignored point."""
        labeled_paths = getattr(self, "labeled_paths", None)
        path_counts = getattr(self, "_uid_path_counts", {})
        if isinstance(labeled_paths, dict) and labeled_paths.pop(key, None) is not None \
                and key[0] in path_counts:
            path_counts[key[0]] = max(0, path_counts[key[0]] - 1)
        getattr(self, "labeled_path_modes", {}).pop(key, None)
        getattr(self, "labeled_path_metadata", {}).pop(key, None)
        if getattr(self, "selected_path_key", None) == key:
            self.selected_path_key = None
        if getattr(self, "editing_path_key", None) == key:
            self.editing_path_key = None

    def renumber_labeled_paths(self, uid):
        """Compact one UID's in-memory path keys after a persisted deletion."""
        uid = int(uid)
        old_keys = sorted(
            (
                key for key in getattr(self, "labeled_paths", {})
                if int(key[0]) == uid
            ),
            key=lambda key: int(key[1]),
        )
        key_remap = {
            key: (uid, segment_id)
            for segment_id, key in enumerate(old_keys, start=1)
        }
        if all(old_key == new_key for old_key, new_key in key_remap.items()):
            return key_remap
        for attribute in (
            "labeled_paths", "labeled_path_modes", "labeled_path_metadata",
        ):
            values = getattr(self, attribute, None)
            if not isinstance(values, dict):
                continue
            updated = {
                key_remap.get(key, key): value for key, value in values.items()
            }
            values.clear()
            values.update(updated)
        for attribute in ("selected_path_key", "editing_path_key"):
            key = getattr(self, attribute, None)
            if key in key_remap:
                setattr(self, attribute, key_remap[key])
        return key_remap

    def _point_sequence_number(self, uid, point_idx=None, position=None,
                               destination=False):
        """Return the 1-based point number shown in the current UID list."""
        uid = int(uid)
        cache = getattr(self, "_uid_point_sequence_cache", None)
        if cache is None:
            cache = {}
            self._uid_point_sequence_cache = cache
        if uid not in cache:
            positions = self._uid_segment_positions.get(
                uid, np.empty(0, dtype=np.int32),
            )
            point_numbers = {}
            position_numbers = {}
            for local_index, global_position in enumerate(positions):
                global_position = int(global_position)
                ordinal = local_index + 1
                position_numbers[global_position] = ordinal
                row = self.traj_df.iloc[global_position]
                for column, number in (("idx_o", ordinal), ("idx_d", ordinal + 1)):
                    try:
                        point_numbers.setdefault(int(row[column]), number)
                    except (KeyError, TypeError, ValueError, OverflowError):
                        continue
            cache[uid] = point_numbers, position_numbers
        point_numbers, position_numbers = cache[uid]
        try:
            number = point_numbers.get(int(point_idx))
        except (TypeError, ValueError, OverflowError):
            number = None
        if number is not None:
            return int(number)
        if position is None:
            return None
        number = position_numbers.get(int(position))
        if number is None:
            return None
        return int(number + (1 if destination else 0))

    def _register_point_hover(self, artist, xs, ys, sequence_numbers):
        """Register one trajectory-point scatter artist for hover lookup."""
        records = [
            (
                float(x), float(y),
                None if sequence is None else int(sequence),
            )
            for x, y, sequence in zip(
                np.atleast_1d(xs), np.atleast_1d(ys), sequence_numbers,
            )
        ]
        if any(record[2] is not None for record in records):
            if not hasattr(self, "_point_scatter_meta"):
                self._point_scatter_meta = []
            self._point_scatter_meta.append((artist, records))

    def _recount_uid_resolved(self, uid):
        positions = self._uid_segment_positions.get(uid, np.empty(0, dtype=np.int32))
        resolved_keys = set(self.labeled_modes) | set(self.ignored_points)
        self._uid_resolved_counts[uid] = sum(
            _annotation_key(self.traj_df.iloc[int(position)])
            in resolved_keys
            or not segment_is_reviewable(
                self.traj_df, int(position), self.ignored_points,
            )
            for position in positions
        )

    def _previous_nonignored_position(self, target):
        uid = int(self.traj_df.iloc[int(target)]["uid"])
        for position in range(int(target) - 1, -1, -1):
            row = self.traj_df.iloc[position]
            if int(row["uid"]) != uid:
                break
            if _annotation_key(row) not in self.ignored_points:
                return position
        return None

    def _uid_point_entries(self, uid):
        """Return origin points plus the UID's otherwise-missing final point."""
        positions = self._uid_segment_positions.get(
            int(uid), np.empty(0, dtype=np.int32),
        )
        entries = []
        for position in positions:
            position = int(position)
            key = _annotation_key(self.traj_df.iloc[position])
            if key is not None:
                entries.append({
                    "position": position,
                    "key": key,
                    "destination": False,
                })
        if len(positions):
            last_position = int(positions[-1])
            terminal_key = _destination_annotation_key(
                self.traj_df.iloc[last_position],
            )
            if terminal_key is not None \
                    and (not entries or terminal_key != entries[-1]["key"]):
                entries.append({
                    "position": last_position,
                    "key": terminal_key,
                    "destination": True,
                })
        return entries

    def _toggle_ignored_point(self, target, point_key=None, destination=False):
        """Ignore/restore one sampled point and invalidate changed adjacent ODs."""
        target = int(target)
        row = self.traj_df.iloc[target]
        key = point_key or (
            _destination_annotation_key(row) if destination else _annotation_key(row)
        )
        if key is None:
            return
        was_ignored = key in self.ignored_points
        if destination and _annotation_key(row) not in self.ignored_points:
            previous_position = target
        else:
            previous_position = self._previous_nonignored_position(target)
        previous_key = (
            _annotation_key(self.traj_df.iloc[previous_position])
            if previous_position is not None else None
        )
        affected_keys = {
            item for item in (
                (previous_key,) if destination else (key, previous_key)
            ) if item is not None
        }
        action = "restored" if was_ignored else "ignored"

        updated_points = set(self.ignored_points)
        if was_ignored:
            updated_points.remove(key)
        else:
            updated_points.add(key)

        # Persist the point state first. If removing the affected labels
        # fails, roll it back so the ignore file and active label CSV never
        # intentionally disagree.
        try:
            if self.output_dir:
                save_ignored_points(self.output_dir, updated_points)
            remove_labeled_keys(self.output_dir, affected_keys)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            if self.output_dir:
                try:
                    save_ignored_points(self.output_dir, self.ignored_points)
                except OSError as rollback_exc:
                    print(f"  [WARN] Failed to roll back ignored points: {rollback_exc}")
            print(f"  [WARN] Failed to update ignored point: {exc}")
            return
        for affected_key in affected_keys:
            self.labeled_modes.pop(affected_key, None)

        self.ignored_points.clear()
        self.ignored_points.update(updated_points)
        self._recount_uid_resolved(key[0])
        export_date = getattr(self, "export_date", None)
        if self.output_dir and export_date:
            try:
                write_labeled_data_copy(
                    self.output_dir, export_date,
                    getattr(self, "excluded_uids", set()),
                )
                write_path_labeled_copy(
                    self.output_dir, export_date,
                    getattr(self, "excluded_uids", set()),
                )
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                print(f"  [WARN] Failed to update labeled-data copy: {exc}")
        self.refresh_uid_segment_list()
        print(f"  [POINT {action.upper()}] uid={key[0]} idx={key[1]}")

        if self.segment_select_callback is None:
            return
        current_became_unreviewable = (
            not was_ignored
            and destination
            and previous_position == self.current_idx
        )
        if not was_ignored and (target == self.current_idx or current_became_unreviewable):
            next_target = next_nonignored_index(
                self.traj_df, self.ignored_points, target, 1,
                getattr(self, "excluded_uids", set()),
            )
            if next_target >= len(self.traj_df):
                next_target = next_nonignored_index(
                    self.traj_df, self.ignored_points, target, -1,
                    getattr(self, "excluded_uids", set()),
                )
            if 0 <= next_target < len(self.traj_df):
                self.segment_select_callback(next_target)
        elif previous_position == self.current_idx:
            # The current OD endpoint changed because its next point was
            # ignored/restored; rebuild this view immediately.
            self.segment_select_callback(self.current_idx)

    def _toggle_excluded_uid(self, uid):
        """Exclude/restore a complete UID trajectory from the dated copy."""
        uid = int(uid)
        was_excluded = uid in self.excluded_uids
        updated_uids = set(self.excluded_uids)
        if was_excluded:
            updated_uids.remove(uid)
        else:
            updated_uids.add(uid)
        try:
            if self.output_dir:
                save_excluded_uids(self.output_dir, updated_uids)
                if self.export_date:
                    write_labeled_data_copy(
                        self.output_dir, self.export_date, updated_uids,
                    )
                    write_path_labeled_copy(
                        self.output_dir, self.export_date, updated_uids,
                    )
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            if self.output_dir:
                try:
                    save_excluded_uids(self.output_dir, self.excluded_uids)
                except OSError as rollback_exc:
                    print(f"  [WARN] Failed to roll back excluded UIDs: {rollback_exc}")
            print(f"  [WARN] Failed to update excluded UID: {exc}")
            return

        self.excluded_uids.clear()
        self.excluded_uids.update(updated_uids)
        self._draw_uid_navigation_list(ensure_current=False)
        self.fig.canvas.draw_idle()
        action = "RESTORED" if was_excluded else "EXCLUDED"
        print(f"  [UID {action}] uid={uid}")

        if not was_excluded and uid == int(self.state.uid) \
                and self.segment_select_callback is not None:
            positions = self._uid_segment_positions.get(
                uid, np.empty(0, dtype=np.int32),
            )
            if len(positions):
                target = next_nonignored_index(
                    self.traj_df, self.ignored_points, int(positions[-1]), 1,
                    self.excluded_uids,
                )
                if target >= len(self.traj_df):
                    target = next_nonignored_index(
                        self.traj_df, self.ignored_points, int(positions[0]), -1,
                        self.excluded_uids,
                    )
                if 0 <= target < len(self.traj_df):
                    self.segment_select_callback(target)

    def _uid_navigation_group(self, uid):
        """Return the requested UID-list group in display priority order."""
        uid = int(uid)
        if uid in getattr(self, "excluded_uids", set()):
            return 3
        total_required = len(self._uid_segment_positions.get(uid, ()))
        resolved_count = getattr(self, "_uid_resolved_counts", {}).get(uid, 0)
        mode_complete = (
            total_required > 0
            and resolved_count >= total_required
        )
        has_path = getattr(self, "_uid_path_counts", {}).get(uid, 0) > 0
        if mode_complete and has_path:
            return 0
        if mode_complete:
            return 1
        if resolved_count > 0 or has_path:
            return 2
        return 4

    def _ordered_uid_navigation_values(self):
        """Sort UID groups, then sort numerically within every group."""
        uid_values = (
            int(uid) for uid in getattr(self, "_uid_nav_values", ())
        )
        ordered = sorted(
            set(uid_values),
            key=lambda uid: (self._uid_navigation_group(uid), uid),
        )
        self._uid_nav_values = np.asarray(ordered, dtype=np.int64)
        self._uid_nav_index = {
            int(uid): index for index, uid in enumerate(self._uid_nav_values)
        }
        return self._uid_nav_values

    def _init_uid_search_box(self):
        """Create an exact UID search field at the bottom of the UID column."""
        nav_bbox = self.ax_uid_nav.get_position()
        search_ax = self.fig.add_axes([
            nav_bbox.x0 + nav_bbox.width * 0.24,
            nav_bbox.y0 + nav_bbox.height * 0.012,
            nav_bbox.width * 0.70,
            nav_bbox.height * 0.042,
        ])
        self._uid_search_box = TextBox(
            search_ax, "搜索：", initial="",
            color="#f7f9fb", hovercolor="#ffffff",
        )
        self._uid_search_box.label.set_fontsize(7.4)
        self._uid_search_box.label.set_fontweight("bold")
        self._uid_search_box.text_disp.set_fontsize(7.4)
        search_ax.patch.set_edgecolor("#7f8c99")
        search_ax.patch.set_linewidth(0.9)
        self._uid_search_box.on_submit(self._search_uid)

    def _search_uid(self, text):
        """Reveal an exact UID and open its next reviewable OD when possible."""
        query = str(text).strip()
        if not query:
            return
        try:
            uid = int(query)
        except (TypeError, ValueError, OverflowError):
            print(f"  [UID SEARCH] 无效 UID：{query}")
            return

        uid_values = self._ordered_uid_navigation_values()
        if uid not in self._uid_segment_positions:
            print(f"  [UID SEARCH] 未找到 UID：{uid}")
            return

        local_index = self._uid_nav_index[uid]
        max_offset = max(0, len(uid_values) - UID_NAV_VISIBLE_ROWS)
        self._uid_nav_offset = min(
            max(0, local_index - UID_NAV_VISIBLE_ROWS // 2), max_offset,
        )
        self._draw_uid_navigation_list(ensure_current=False)
        self.fig.canvas.draw_idle()

        if uid in getattr(self, "excluded_uids", set()):
            print(f"  [UID SEARCH] UID {uid} 已忽略，可在列表中右键恢复")
            return
        target = self._first_unlabeled_position(uid)
        if target is None:
            print(f"  [UID SEARCH] UID {uid} 没有可查看的 OD")
            return
        print(f"  [UID SEARCH] UID {uid} -> OD #{target}")
        if target != self.current_idx and self.segment_select_callback is not None:
            self.segment_select_callback(target)

    def _draw_uid_navigation_list(self, ensure_current=False):
        """Draw the scrollable all-UID navigation list at the far left."""
        ax = self.ax_uid_nav
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        self._uid_nav_hit_rows = []

        uid_values = self._ordered_uid_navigation_values()
        total = len(uid_values)
        if not total:
            ax.text(0.5, 0.5, "没有可用 UID", ha="center", va="center", fontsize=9)
            return

        current_local = self._uid_nav_index.get(int(self.state.uid), 0)
        if ensure_current:
            if current_local < self._uid_nav_offset:
                self._uid_nav_offset = current_local
            elif current_local >= self._uid_nav_offset + UID_NAV_VISIBLE_ROWS:
                self._uid_nav_offset = current_local - UID_NAV_VISIBLE_ROWS + 1

        max_offset = max(0, total - UID_NAV_VISIBLE_ROWS)
        self._uid_nav_offset = min(max(0, self._uid_nav_offset), max_offset)
        stop = min(total, self._uid_nav_offset + UID_NAV_VISIBLE_ROWS)

        ax.text(
            0.5, 0.982,
            f"UID 列表  {current_local + 1} / {total}\n● OD完成   ◆ 有路径",
            ha="center", va="top", fontsize=7.7, fontweight="bold",
            transform=ax.transAxes,
        )

        top, bottom = 0.89, 0.120
        row_height = (top - bottom) / UID_NAV_VISIBLE_ROWS
        dot_xs, dot_ys = [], []
        dot_facecolors, dot_edgecolors = [], []
        path_xs, path_ys = [], []
        excluded_xs, excluded_ys = [], []
        excluded_uids = getattr(self, "excluded_uids", set())
        path_counts = getattr(self, "_uid_path_counts", {})
        for visible_row, local_index in enumerate(range(self._uid_nav_offset, stop)):
            uid = int(uid_values[local_index])
            y = top - (visible_row + 0.5) * row_height
            is_current = uid == int(self.state.uid)
            is_excluded = uid in excluded_uids
            if is_current:
                ax.add_patch(Rectangle(
                    (0.02, y - row_height * 0.46), 0.93, row_height * 0.92,
                    transform=ax.transAxes, facecolor="#e7f2ff",
                    edgecolor="#4c91d9", linewidth=1.0, zorder=0,
                ))

            if is_excluded:
                excluded_xs.append(0.10)
                excluded_ys.append(y)
            else:
                total_required = len(self._uid_segment_positions.get(uid, ()))
                progress_counts = self._uid_resolved_counts
                is_complete = (
                    total_required > 0
                    and progress_counts.get(uid, 0) >= total_required
                )
                dot_xs.append(0.10)
                dot_ys.append(y)
                dot_facecolors.append(
                    "#2ca02c" if is_complete else (1.0, 1.0, 1.0, 0.0)
                )
                dot_edgecolors.append("#1e6b1e" if is_complete else "#a5a5a5")
            if path_counts.get(uid, 0) > 0:
                path_xs.append(0.155)
                path_ys.append(y)
            ax.text(
                0.225, y, str(uid), ha="left", va="center", fontsize=7.1,
                fontweight="bold" if is_current else "normal",
                color="#8a8a8a" if is_excluded else (
                    "#005baa" if is_current else "#303030"
                ),
                fontstyle="italic" if is_excluded else "normal",
                transform=ax.transAxes,
            )
            self._uid_nav_hit_rows.append((
                y - row_height * 0.5, y + row_height * 0.5, uid,
            ))

        if dot_xs:
            ax.scatter(
                dot_xs, dot_ys, s=32, marker="o", facecolors=dot_facecolors,
                edgecolors=dot_edgecolors, linewidths=0.8,
                transform=ax.transAxes, zorder=2,
            )
        if path_xs:
            ax.scatter(
                path_xs, path_ys, s=29, marker="D", c="#00a7b5",
                edgecolors="#005f69", linewidths=0.75,
                transform=ax.transAxes, zorder=3,
            )
        if excluded_xs:
            ax.scatter(
                excluded_xs, excluded_ys, s=38, marker="x", c="#777777",
                linewidths=1.25, transform=ax.transAxes, zorder=2,
            )

        if total > UID_NAV_VISIBLE_ROWS:
            track_y, track_height = bottom, top - bottom
            thumb_height = track_height * UID_NAV_VISIBLE_ROWS / total
            scroll_range = track_height - thumb_height
            thumb_y = top - thumb_height - (
                scroll_range * self._uid_nav_offset / max_offset if max_offset else 0
            )
            ax.add_patch(Rectangle(
                (0.965, track_y), 0.012, track_height,
                transform=ax.transAxes, facecolor="#e1e1e1", edgecolor="none",
            ))
            ax.add_patch(Rectangle(
                (0.965, thumb_y), 0.012, thumb_height,
                transform=ax.transAxes, facecolor="#7f7f7f", edgecolor="none",
            ))

    def _on_uid_nav_scroll(self, event):
        if event.inaxes is not self.ax_uid_nav:
            return
        max_offset = max(0, len(self._uid_nav_values) - UID_NAV_VISIBLE_ROWS)
        if max_offset <= 0:
            return
        direction = getattr(event, "step", 0)
        if not direction:
            direction = 1 if event.button == "up" else -1
        delta = -UID_NAV_SCROLL_STEP if direction > 0 else UID_NAV_SCROLL_STEP
        next_offset = min(max(0, self._uid_nav_offset + delta), max_offset)
        if next_offset == self._uid_nav_offset:
            return
        self._uid_nav_offset = next_offset
        self._draw_uid_navigation_list(ensure_current=False)
        self.fig.canvas.draw_idle()

    def _first_unlabeled_position(self, uid):
        if int(uid) in getattr(self, "excluded_uids", set()):
            return None
        positions = self._uid_segment_positions.get(uid, np.empty(0, dtype=np.int32))
        ignored_points = getattr(self, "ignored_points", set())
        if not len(positions):
            return None
        if getattr(self, "annotation_mode", ANNOTATION_MODE) == ANNOTATION_PATH:
            for position in positions:
                if segment_is_reviewable(self.traj_df, int(position), ignored_points):
                    return int(position)
            return None
        for position in positions:
            key = _annotation_key(self.traj_df.iloc[int(position)])
            if key not in self.labeled_modes \
                    and segment_is_reviewable(self.traj_df, int(position), ignored_points):
                return int(position)
        for position in positions:
            if segment_is_reviewable(self.traj_df, int(position), ignored_points):
                return int(position)
        return None

    def _on_uid_nav_click(self, event):
        if event.inaxes is not self.ax_uid_nav or event.ydata is None:
            return
        if event.button not in (1, 3):
            return
        for y_min, y_max, uid in self._uid_nav_hit_rows:
            if y_min <= event.ydata <= y_max:
                if event.button == 3:
                    self._toggle_excluded_uid(uid)
                else:
                    target = self._first_unlabeled_position(uid)
                    if target is not None and target != self.current_idx \
                            and self.segment_select_callback is not None:
                        self.segment_select_callback(target)
                return

    def _uid_saved_path_keys(self, uid=None):
        """Return current-UID saved path keys in stable segment order."""
        uid = int(self.state.uid if uid is None else uid)
        return sorted(
            (
                key for key in getattr(self, "labeled_paths", {})
                if int(key[0]) == uid
            ),
            key=lambda key: int(key[1]),
        )

    def _draw_uid_path_list(self, ensure_current=False):
        """Draw saved paths for the current UID in a dedicated scroll list."""
        ax = self.ax_path_list
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        self._path_list_hit_rows = []

        uid = int(self.state.uid)
        path_keys = self._uid_saved_path_keys(uid)
        total = len(path_keys)
        if self._path_list_uid != uid:
            self._path_list_uid = uid
            self._path_list_offset = 0
        if not total:
            ax.text(
                0.5, 0.975, f"UID {uid}\n已标路径 0",
                ha="center", va="top", fontsize=8.2, fontweight="bold",
                transform=ax.transAxes,
            )
            ax.text(
                0.5, 0.52, "当前 UID\n暂无已标路径",
                ha="center", va="center", fontsize=8.2, color="#777777",
                transform=ax.transAxes,
            )
            return

        selected_key = getattr(self, "selected_path_key", None)
        selected_local = (
            path_keys.index(selected_key) if selected_key in path_keys else None
        )
        if ensure_current and selected_local is not None:
            if selected_local < self._path_list_offset:
                self._path_list_offset = selected_local
            elif selected_local >= self._path_list_offset + PATH_LIST_VISIBLE_ROWS:
                self._path_list_offset = selected_local - PATH_LIST_VISIBLE_ROWS + 1

        max_offset = max(0, total - PATH_LIST_VISIBLE_ROWS)
        self._path_list_offset = min(max(0, self._path_list_offset), max_offset)
        stop = min(total, self._path_list_offset + PATH_LIST_VISIBLE_ROWS)
        selected_text = f"选中 {selected_local + 1}/{total}" if selected_local is not None \
            else f"共 {total} 条"
        ax.text(
            0.5, 0.975, f"UID {uid}\n已标路径  {selected_text}",
            ha="center", va="top", fontsize=8.2, fontweight="bold",
            transform=ax.transAxes,
        )

        top, bottom = 0.89, 0.055
        row_height = (top - bottom) / PATH_LIST_VISIBLE_ROWS
        marker_xs, marker_ys, marker_colors = [], [], []
        metadata = getattr(self, "labeled_path_metadata", {})
        modes = getattr(self, "labeled_path_modes", {})
        editing_key = getattr(self, "editing_path_key", None)
        for visible_row, local_index in enumerate(range(self._path_list_offset, stop)):
            key = path_keys[local_index]
            y = top - (visible_row + 0.5) * row_height
            is_selected = key == selected_key
            is_editing = key == editing_key
            if is_selected:
                ax.add_patch(Rectangle(
                    (0.025, y - row_height * 0.46), 0.925, row_height * 0.92,
                    transform=ax.transAxes,
                    facecolor="#ffe8d6" if is_editing else "#fff3cd",
                    edgecolor="#d35400" if is_editing else "#d99000",
                    linewidth=1.1, zorder=0,
                ))
            mode = modes.get(key, "Other")
            steps = metadata.get(key, {}).get(
                "steps", max(0, len(self.labeled_paths.get(key, ())) - 1),
            )
            marker_xs.append(0.11)
            marker_ys.append(y)
            marker_colors.append(LABELED_POINT_COLORS.get(
                mode, LABELED_POINT_COLORS["Other"],
            ))
            ax.text(
                0.20, y,
                f"{int(key[1]):02d}  {mode}  {int(steps)}步"
                f"{'  重绘中' if is_editing else ''}",
                ha="left", va="center", fontsize=7.0,
                fontweight="bold" if is_selected else "normal",
                color=(
                    "#b23a00" if is_editing
                    else "#8a5200" if is_selected else "#303030"
                ),
                transform=ax.transAxes,
            )
            self._path_list_hit_rows.append((
                y - row_height * 0.5, y + row_height * 0.5, key,
            ))

        if marker_xs:
            ax.scatter(
                marker_xs, marker_ys, s=28, marker="D",
                c=marker_colors, edgecolors="white", linewidths=0.7,
                transform=ax.transAxes, zorder=2,
            )
        if total > PATH_LIST_VISIBLE_ROWS:
            track_y, track_height = bottom, top - bottom
            thumb_height = track_height * PATH_LIST_VISIBLE_ROWS / total
            scroll_range = track_height - thumb_height
            thumb_y = top - thumb_height - (
                scroll_range * self._path_list_offset / max_offset if max_offset else 0
            )
            ax.add_patch(Rectangle(
                (0.965, track_y), 0.012, track_height,
                transform=ax.transAxes, facecolor="#e1e1e1", edgecolor="none",
            ))
            ax.add_patch(Rectangle(
                (0.965, thumb_y), 0.012, thumb_height,
                transform=ax.transAxes, facecolor="#7f7f7f", edgecolor="none",
            ))

    def _on_path_list_scroll(self, event):
        if event.inaxes is not self.ax_path_list:
            return
        total = len(self._uid_saved_path_keys())
        max_offset = max(0, total - PATH_LIST_VISIBLE_ROWS)
        if max_offset <= 0:
            return
        direction = getattr(event, "step", 0)
        if not direction:
            direction = 1 if event.button == "up" else -1
        delta = -PATH_LIST_SCROLL_STEP if direction > 0 else PATH_LIST_SCROLL_STEP
        next_offset = min(max(0, self._path_list_offset + delta), max_offset)
        if next_offset == self._path_list_offset:
            return
        self._path_list_offset = next_offset
        self._draw_uid_path_list(ensure_current=False)
        self.fig.canvas.draw_idle()

    def _saved_path_anchor_position(self, key):
        """Map a saved path back to the OD window in which it was recorded."""
        positions = self._uid_segment_positions.get(
            int(key[0]), np.empty(0, dtype=np.int32),
        )
        if not len(positions):
            return None
        anchor = getattr(self, "labeled_path_metadata", {}).get(
            key, {},
        ).get("anchor_idx_o")
        if anchor is None or "idx_o" not in self.traj_df.columns:
            return int(positions[0])
        idx_values = pd.to_numeric(
            self.traj_df.iloc[positions]["idx_o"], errors="coerce",
        ).to_numpy(dtype=float)
        exact = np.flatnonzero(idx_values == int(anchor))
        if len(exact):
            return int(positions[int(exact[0])])
        valid = np.flatnonzero(~np.isnan(idx_values))
        if not len(valid):
            return int(positions[0])
        nearest = valid[int(np.argmin(np.abs(idx_values[valid] - int(anchor))))]
        return int(positions[int(nearest)])

    def _on_path_list_click(self, event):
        if event.inaxes is not self.ax_path_list or event.ydata is None \
                or event.button != 1:
            return
        for y_min, y_max, key in self._path_list_hit_rows:
            if not y_min <= event.ydata <= y_max:
                continue
            self.editing_path_key = None
            self.selected_path_key = key
            target = self._saved_path_anchor_position(key)
            selected_mode = self.labeled_path_modes.get(key)
            if getattr(self, "annotation_mode", ANNOTATION_MODE) == ANNOTATION_PATH \
                    and selected_mode in MODE_LIST:
                self.set_active_path_mode(selected_mode)
            print(
                f"  [PATH VIEW] uid={key[0]} segment={key[1]} "
                f"mode={selected_mode or '-'}",
            )
            if target is not None and self.segment_select_callback is not None:
                self.segment_select_callback(target)
            else:
                self._draw_uid_path_list(ensure_current=True)
                self.fig.canvas.draw_idle()
            return

    def _draw_uid_segment_list(self, ensure_current=False):
        """Draw the scrollable OD list for the current UID."""
        ax = self.ax_uid_list
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        self._uid_list_hit_rows = []

        positions = self._uid_segment_positions.get(
            self.state.uid, np.empty(0, dtype=np.int32),
        )
        entries = self._uid_point_entries(self.state.uid)
        total = len(entries)
        if not total:
            ax.text(0.5, 0.5, "当前 UID\n没有可用 OD", ha="center", va="center", fontsize=9)
            return

        local_matches = np.flatnonzero(positions == self.current_idx)
        current_od_local = int(local_matches[0]) if len(local_matches) else 0
        current_local = next((
            index for index, entry in enumerate(entries)
            if not entry["destination"] and entry["position"] == self.current_idx
        ), 0)
        if self._uid_list_uid != self.state.uid:
            self._uid_list_uid = self.state.uid
            self._uid_list_offset = max(0, current_local - UID_LIST_VISIBLE_ROWS // 2)
        elif ensure_current:
            if current_local < self._uid_list_offset:
                self._uid_list_offset = current_local
            elif current_local >= self._uid_list_offset + UID_LIST_VISIBLE_ROWS:
                self._uid_list_offset = current_local - UID_LIST_VISIBLE_ROWS + 1

        max_offset = max(0, total - UID_LIST_VISIBLE_ROWS)
        self._uid_list_offset = min(max(0, self._uid_list_offset), max_offset)
        stop = min(total, self._uid_list_offset + UID_LIST_VISIBLE_ROWS)

        if getattr(self, "annotation_mode", ANNOTATION_MODE) == ANNOTATION_PATH:
            list_title = (
                f"UID {self.state.uid}\n"
                f"路径段已存 {self._uid_path_counts.get(self.state.uid, 0)} | "
                f"OD {current_od_local + 1}/{len(positions)} | 信令点 {total}"
            )
        else:
            list_title = (
                f"UID {self.state.uid}\n点 / OD列表  {current_local + 1} / {total}"
            )
        ax.text(
            0.5, 0.975, list_title,
            ha="center", va="top", fontsize=8.5, fontweight="bold",
            transform=ax.transAxes,
        )
        top, bottom = 0.89, 0.055
        row_height = (top - bottom) / UID_LIST_VISIBLE_ROWS
        dot_xs, dot_ys = [], []
        dot_sizes, dot_facecolors, dot_edgecolors = [], [], []
        ignored_xs, ignored_ys = [], []
        ignored_points = getattr(self, "ignored_points", set())
        for visible_row, local_index in enumerate(range(self._uid_list_offset, stop)):
            entry = entries[local_index]
            global_index = entry["position"]
            y = top - (visible_row + 0.5) * row_height
            is_current = (
                not entry["destination"] and global_index == self.current_idx
            )
            if is_current:
                ax.add_patch(Rectangle(
                    (0.025, y - row_height * 0.46), 0.925, row_height * 0.92,
                    transform=ax.transAxes, facecolor="#e7f2ff",
                    edgecolor="#4c91d9", linewidth=1.0, zorder=0,
                ))

            key = entry["key"]
            is_ignored = key in ignored_points
            mode = None if is_ignored or entry["destination"] \
                else self.labeled_modes.get(key)
            dot_color = LABELED_POINT_COLORS.get(mode)
            if is_ignored:
                ignored_xs.append(0.14)
                ignored_ys.append(y)
            else:
                dot_xs.append(0.14)
                dot_ys.append(y)
                dot_sizes.append(38 if dot_color is not None else 31)
                if dot_color is None:
                    dot_facecolors.append((1.0, 1.0, 1.0, 0.0))
                    dot_edgecolors.append("#a5a5a5")
                else:
                    dot_facecolors.append((*dot_color, 1.0))
                    dot_edgecolors.append("#303030")

            ax.text(
                0.25, y, f"{local_index + 1:>4d}",
                ha="left", va="center", fontsize=8,
                fontweight="bold" if is_current else "normal",
                color="#9a9a9a" if is_ignored else ("#005baa" if is_current else "#303030"),
                fontstyle="italic" if is_ignored else "normal",
                transform=ax.transAxes,
            )
            if is_ignored:
                ax.text(
                    0.88, y, "忽略", ha="right", va="center", fontsize=6.8,
                    color="#777777", fontweight="bold", transform=ax.transAxes,
                )
            elif mode is not None:
                ax.text(
                    0.88, y, mode,
                    ha="right", va="center", fontsize=6.8,
                    color=dot_color, fontweight="bold", transform=ax.transAxes,
                )
            self._uid_list_hit_rows.append((
                y - row_height * 0.5, y + row_height * 0.5, entry,
            ))

        if dot_xs:
            ax.scatter(
                dot_xs, dot_ys, s=dot_sizes, marker="o",
                facecolors=dot_facecolors, edgecolors=dot_edgecolors,
                linewidths=0.75, transform=ax.transAxes, zorder=2,
            )
        if ignored_xs:
            ax.scatter(
                ignored_xs, ignored_ys, s=38, marker="x", c="#777777",
                linewidths=1.25, transform=ax.transAxes, zorder=2,
            )

        if total > UID_LIST_VISIBLE_ROWS:
            track_y, track_height = bottom, top - bottom
            thumb_height = track_height * UID_LIST_VISIBLE_ROWS / total
            scroll_range = track_height - thumb_height
            thumb_y = top - thumb_height - (
                scroll_range * self._uid_list_offset / max_offset if max_offset else 0
            )
            ax.add_patch(Rectangle(
                (0.965, track_y), 0.012, track_height,
                transform=ax.transAxes, facecolor="#e1e1e1", edgecolor="none",
            ))
            ax.add_patch(Rectangle(
                (0.965, thumb_y), 0.012, thumb_height,
                transform=ax.transAxes, facecolor="#7f7f7f", edgecolor="none",
            ))

    def refresh_uid_segment_list(self):
        """Refresh UID completion and OD label colors without changing scroll."""
        self._draw_uid_navigation_list(ensure_current=False)
        self._draw_uid_path_list(ensure_current=True)
        self._draw_uid_segment_list(ensure_current=False)
        self.fig.canvas.draw_idle()

    def _on_uid_list_scroll(self, event):
        if event.inaxes is not self.ax_uid_list:
            return
        entries = self._uid_point_entries(self.state.uid)
        max_offset = max(0, len(entries) - UID_LIST_VISIBLE_ROWS)
        if max_offset <= 0:
            return

        direction = getattr(event, "step", 0)
        if not direction:
            direction = 1 if event.button == "up" else -1
        delta = -UID_LIST_SCROLL_STEP if direction > 0 else UID_LIST_SCROLL_STEP
        next_offset = min(max(0, self._uid_list_offset + delta), max_offset)
        if next_offset == self._uid_list_offset:
            return
        self._uid_list_offset = next_offset
        self._draw_uid_segment_list(ensure_current=False)
        self.fig.canvas.draw_idle()

    def _on_uid_list_click(self, event):
        if event.inaxes is not self.ax_uid_list or event.ydata is None:
            return
        if event.button not in (1, 3):
            return
        for y_min, y_max, entry in self._uid_list_hit_rows:
            if y_min <= event.ydata <= y_max:
                target = entry["position"]
                key = entry["key"]
                if event.button == 3:
                    self._toggle_ignored_point(
                        target, point_key=key,
                        destination=entry["destination"],
                    )
                elif not entry["destination"] \
                        and segment_is_reviewable(
                            self.traj_df, target,
                            getattr(self, "ignored_points", set()),
                        ) \
                        and target != self.current_idx \
                        and self.segment_select_callback is not None:
                    self.segment_select_callback(target)
                return

    def _init_road_toggle_buttons(self):
        """Create vertical road switches inside the right-side legend card."""
        info_bbox = self.ax_info.get_position()
        button_left = info_bbox.x0 + info_bbox.width * 0.055
        button_width = info_bbox.width * 0.89
        button_height = info_bbox.height * 0.022
        first_y = 0.250
        row_step = 0.028

        for index, mode_name in enumerate(DISPLAY_MODE_LIST):
            button_y = info_bbox.y0 + info_bbox.height * (first_y - index * row_step)
            button_ax = self.fig.add_axes([
                button_left,
                button_y,
                button_width,
                button_height,
            ])
            button = Button(button_ax, "")
            button.label.set_fontsize(6.9)
            button.label.set_fontweight("bold")
            button.on_clicked(
                lambda _event, selected_mode=mode_name: self._toggle_road_layer(selected_mode)
            )
            self._road_buttons[mode_name] = button
            self._update_road_button_style(mode_name)

    def _init_label_buttons(self):
        """Create direct-save mode buttons matching keyboard keys 1 through 6."""
        button_width = 0.106
        button_gap = 0.010
        button_height = 0.036
        left = 0.19
        self.fig.text(
            left - 0.015, 0.043, "标注", ha="right", va="center",
            fontsize=8, fontweight="bold", color="#444444",
        )

        for index, (key, mode_name) in enumerate(LABEL_OPTIONS.items()):
            button_ax = self.fig.add_axes([
                left + index * (button_width + button_gap),
                0.025,
                button_width,
                button_height,
            ])
            color = LABELED_POINT_COLORS.get(mode_name, (0.38, 0.38, 0.38))
            hover_color = tuple(min(channel + 0.16, 1.0) for channel in color)
            mode_label = MODE_LABELS.get(mode_name, "其他" if mode_name == "Other" else mode_name)
            button = Button(
                button_ax, f"{key}  {mode_name} {mode_label}",
                color=color, hovercolor=hover_color,
            )
            button.label.set_color("white")
            button.label.set_fontsize(8)
            button.label.set_fontweight("bold")
            button.ax.patch.set_edgecolor("#303030")
            button.ax.patch.set_linewidth(1.2)
            button.on_clicked(
                lambda _event, selected_mode=mode_name: self._request_label_mode(selected_mode)
            )
            self._label_buttons[mode_name] = button

        self._update_label_button_styles()

    def _init_annotation_toggle_button(self):
        """Create the single workflow switch at the lower-left edge."""
        button_ax = self.fig.add_axes([0.025, 0.018, 0.125, 0.050])
        self._annotation_toggle_button = Button(button_ax, "")
        self._annotation_toggle_button.label.set_fontsize(7.4)
        self._annotation_toggle_button.label.set_fontweight("bold")
        self._annotation_toggle_button.on_clicked(
            lambda _event: self._request_annotation_toggle()
        )
        self._update_annotation_toggle_style()

    def _request_annotation_toggle(self):
        if self.annotation_toggle_callback is not None:
            self.annotation_toggle_callback()

    def _update_annotation_toggle_style(self):
        button = getattr(self, "_annotation_toggle_button", None)
        if button is None:
            return
        is_path = getattr(self, "annotation_mode", ANNOTATION_MODE) == ANNOTATION_PATH
        if is_path:
            label = "当前：路径标注\n点击切到模式"
            color, hover = "#006d77", "#158c96"
        else:
            label = "当前：模式标注\n点击切到路径"
            color, hover = "#3d5a80", "#55749c"
        button.label.set_text(label)
        button.label.set_color("white")
        button.color = color
        button.hovercolor = hover
        button.ax.set_facecolor(color)
        button.ax.patch.set_edgecolor("#263238")
        button.ax.patch.set_linewidth(1.3)

    def _update_label_button_styles(self):
        """Disable mode-saving buttons visually while drawing a path."""
        enabled = getattr(self, "annotation_mode", ANNOTATION_MODE) == ANNOTATION_MODE
        for mode_name, button in getattr(self, "_label_buttons", {}).items():
            base_color = LABELED_POINT_COLORS.get(mode_name, (0.38, 0.38, 0.38))
            color = base_color if enabled else "#b8bec4"
            hover = (
                tuple(min(channel + 0.16, 1.0) for channel in base_color)
                if enabled else "#c8cdd2"
            )
            button.set_active(enabled)
            button.color = color
            button.hovercolor = hover
            button.ax.set_facecolor(color)
            button.label.set_color("white" if enabled else "#66717b")

    def _update_annotation_controls(self):
        self._update_annotation_toggle_style()
        self._update_label_button_styles()
        for mode_name in getattr(self, "_road_buttons", {}):
            self._update_road_button_style(mode_name)

    def _request_label_mode(self, mode_name):
        if getattr(self, "annotation_mode", ANNOTATION_MODE) == ANNOTATION_MODE \
                and self.label_select_callback is not None:
            self.label_select_callback(mode_name)

    def _toggle_road_layer(self, mode_name):
        """Toggle one road layer without redrawing or changing the viewport."""
        if getattr(self, "annotation_mode", ANNOTATION_MODE) == ANNOTATION_PATH:
            if mode_name not in MODE_LIST:
                print("  [PATH] L2 is display-only and cannot be used as a path mode")
                return
            self.set_active_path_mode(mode_name)
            return
        is_visible = not self.road_visibility.get(mode_name, True)
        self.road_visibility[mode_name] = is_visible
        artist = self._road_artists.get(mode_name)
        if is_visible and artist is None:
            self._prepare_road_display_mode(mode_name)
            artist = self._draw_projected_road_layer(mode_name)
        if artist is not None:
            artist.set_visible(is_visible)
        road_hover = getattr(self, "_road_hover", None)
        if road_hover is not None:
            road_hover.set_visible(False)
        self._update_road_button_style(mode_name)
        self.fig.canvas.draw_idle()

    def set_active_path_mode(self, mode_name):
        """Select exactly one road network; that network is the truth mode."""
        if mode_name not in MODE_LIST:
            return False
        previous_mode = self.active_path_mode
        self.active_path_mode = mode_name
        road_sets = getattr(self, "road_sets", None)
        if isinstance(road_sets, dict) and mode_name in road_sets:
            self.state.multi_mapdata = road_sets[mode_name]
        if previous_mode != mode_name and self.state.path_history:
            self.state.clear_path()
        for candidate in DISPLAY_MODE_LIST:
            is_visible = candidate == mode_name
            self.road_visibility[candidate] = is_visible
            artist = self._road_artists.get(candidate)
            if is_visible and artist is None:
                self._prepare_road_display_mode(candidate)
                artist = self._draw_projected_road_layer(candidate)
            if artist is not None:
                artist.set_visible(is_visible)
            self._update_road_button_style(candidate)
        road_hover = getattr(self, "_road_hover", None)
        if road_hover is not None:
            road_hover.set_visible(False)
        if hasattr(self, "path_line"):
            self.refresh()
        else:
            self.fig.canvas.draw_idle()
        print(f"  [PATH MODE] {mode_name} {MODE_LABELS.get(mode_name, mode_name)}")
        return True

    def enter_path_road_mode(self, preferred_mode=None):
        """Remember normal visibility and enter exclusive path-road selection."""
        if self._mode_workflow_road_visibility is None:
            self._mode_workflow_road_visibility = dict(self.road_visibility)
        if preferred_mode not in MODE_LIST:
            preferred_mode = next(
                (mode for mode in MODE_LIST if self.road_visibility.get(mode)),
                MODE_LIST[0],
            )
        return self.set_active_path_mode(preferred_mode)

    def leave_path_road_mode(self):
        """Restore the multi-layer visibility used during mode annotation."""
        saved_visibility = self._mode_workflow_road_visibility
        self._mode_workflow_road_visibility = None
        self.active_path_mode = None
        if saved_visibility is None:
            return
        self.road_visibility.update(saved_visibility)
        for candidate in DISPLAY_MODE_LIST:
            artist = self._road_artists.get(candidate)
            if artist is not None:
                artist.set_visible(self.road_visibility.get(candidate, False))
            self._update_road_button_style(candidate)
        self.fig.canvas.draw_idle()

    def _update_road_button_style(self, mode_name):
        button = self._road_buttons.get(mode_name)
        if button is None:
            return

        is_visible = self.road_visibility.get(mode_name, True)
        label = MODE_LABELS.get(mode_name, mode_name)
        focused = (
            getattr(self, "annotation_mode", ANNOTATION_MODE) == ANNOTATION_PATH
            and getattr(self, "active_path_mode", None) == mode_name
        )
        prefix = "> " if focused else ""
        button.label.set_text(f"{prefix}{mode_name} {label}  {'开' if is_visible else '关'}")
        if is_visible:
            color = ROAD_LAYER_COLORS.get(mode_name, (0.5, 0.5, 0.5))
            hover_color = tuple(min(channel + 0.16, 1.0) for channel in color)
            text_color = "white"
            edge_color = "#303030"
        else:
            color = "#d4d4d4"
            hover_color = "#e5e5e5"
            text_color = "#555555"
            edge_color = "#909090"

        button.color = color
        button.hovercolor = hover_color
        button.ax.set_facecolor(color)
        button.ax.patch.set_edgecolor(edge_color)
        button.ax.patch.set_linewidth(2.2 if focused else 1.2)
        button.label.set_color(text_color)

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

        # Saved manual paths for the current UID remain visible as reference
        # lines. They are clipped to this OD's fixed viewport and never rescale it.
        self._draw_saved_paths(state)

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
        start_sequence = self._point_sequence_number(
            state.uid, state.row.get("idx_o"), self.current_idx,
        )
        self._register_point_hover(
            self.start_handle, [start_mx], [start_my], [start_sequence],
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
        end_sequence = self._point_sequence_number(
            state.uid, state.row.get("idx_d"), self.current_idx,
            destination=True,
        )
        self._register_point_hover(
            self.end_handle, [end_mx], [end_my], [end_sequence],
        )
        if state.path_history:
            path_x, path_y = hex_to_mercator(
                [point[0] for point in state.path_history],
                [point[1] for point in state.path_history],
                [point[2] for point in state.path_history],
            )
            path_x, path_y = mercator_wgs84_to_gcj02(path_x, path_y)
        else:
            path_x, path_y = [], []
        (self.path_line,) = self.ax.plot(
            path_x, path_y, "-",
            color="crimson", linewidth=2.8, alpha=0.92, zorder=6.2, label="Path",
        )

        if state.cur is not None:
            cursor_mx, cursor_my = hex_to_mercator(*state.cur)
            cursor_mx, cursor_my = mercator_wgs84_to_gcj02(cursor_mx, cursor_my)
            cursor_offsets = [[cursor_mx, cursor_my]]
        else:
            cursor_offsets = np.empty((0, 2))
        self.cursor = self.ax.scatter(
            [], [],
            c="cyan", marker="o", s=48,
            edgecolors="darkblue", linewidths=1.8, zorder=8.3, label="Cursor",
        )
        self.cursor.set_offsets(cursor_offsets)
        self.path_start_handle = self.ax.scatter(
            [], [], c="#ffd60a", marker="o", s=34,
            edgecolors="#8a5a00", linewidths=1.2, zorder=8.5,
            label="Manual Path Start",
        )
        if state.path_history:
            self.path_start_handle.set_offsets([[path_x[0], path_y[0]]])

        # ---- 上下文参考点（前一段起点 / 后一段终点）----
        self._draw_context_points(state)
        self._point_hover = self.ax.annotate(
            "", xy=(0, 0), xytext=(8, 8), textcoords="offset points",
            fontsize=8.5, fontweight="bold", color="#202020", zorder=9,
            bbox=dict(
                boxstyle="round", facecolor="#fff7cc",
                edgecolor="#555555", alpha=0.96,
            ),
        )
        self._point_hover.set_visible(False)
        self._road_hover = self.ax.annotate(
            "", xy=(0, 0), xytext=(8, 8), textcoords="offset points",
            fontsize=8.2, fontweight="bold", color="#203040", zorder=8.9,
            bbox=dict(
                boxstyle="round", facecolor="#e8f4ff",
                edgecolor="#4b6f8f", alpha=0.96,
            ),
        )
        self._road_hover.set_visible(False)

        # ---- 右侧：速度分布直方图 ----
        self._draw_velocity_hist(state)

    def _draw_saved_paths(self, state):
        """Draw completed paths for the current UID without changing limits."""
        saved_paths = getattr(self, "labeled_paths", {})
        if not saved_paths:
            return
        lines_by_mode = {}
        highlighted_path = None
        uid_keys = [key for key in saved_paths if int(key[0]) == int(state.uid)]
        selected_key = getattr(self, "selected_path_key", None)
        editing_key = getattr(self, "editing_path_key", None)
        highlight_key = (
            selected_key if selected_key in uid_keys
            else max(uid_keys, key=lambda key: int(key[1])) if uid_keys else None
        )
        for key, path in saved_paths.items():
            if int(key[0]) != int(state.uid) or len(path) < 2:
                continue
            if key == editing_key:
                continue
            coordinates = np.asarray(path, dtype=np.int64)
            mx, my = hex_to_mercator(
                coordinates[:, 0], coordinates[:, 1], coordinates[:, 2],
            )
            gx, gy = mercator_wgs84_to_gcj02(mx, my)
            mode = getattr(self, "labeled_path_modes", {}).get(key, "Other")
            projected_path = np.column_stack((np.asarray(gx), np.asarray(gy)))
            if key == highlight_key:
                highlighted_path = mode, projected_path
                continue
            lines_by_mode.setdefault(mode, []).append(
                projected_path
            )
        for mode, lines in lines_by_mode.items():
            collection = LineCollection(
                lines,
                colors=[LABELED_POINT_COLORS.get(mode, LABELED_POINT_COLORS["Other"])],
                linewidths=1.7, alpha=0.42 if selected_key in uid_keys else 0.62,
                zorder=4.8,
            )
            self.ax.add_collection(collection, autolim=False)
        if highlighted_path is not None:
            mode, line = highlighted_path
            mode_color = LABELED_POINT_COLORS.get(
                mode, LABELED_POINT_COLORS["Other"],
            )
            halo = LineCollection(
                [line], colors=["white"], linewidths=7.0,
                alpha=0.92, zorder=5.35,
            )
            highlight = LineCollection(
                [line], colors=[mode_color], linewidths=4.0,
                alpha=1.0, zorder=5.45,
            )
            self.ax.add_collection(halo, autolim=False)
            self.ax.add_collection(highlight, autolim=False)
            x_limits = self.ax.get_xlim()
            y_limits = self.ax.get_ylim()
            self.ax.scatter(
                [line[-1, 0]], [line[-1, 1]], s=78, marker="o",
                facecolors="white", edgecolors=[mode_color], linewidths=2.5,
                alpha=1.0, zorder=8.4,
            )
            self.ax.set_xlim(x_limits)
            self.ax.set_ylim(y_limits)

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
        ignored_points = getattr(self, "ignored_points", set())
        if ignored_points:
            uid_positions = np.asarray([
                int(position) for position in uid_positions
                if segment_is_reviewable(traj_df, int(position), ignored_points)
            ], dtype=np.int32)
        previous_positions = uid_positions[uid_positions < idx]
        next_positions = uid_positions[uid_positions > idx]

        def _project_rows(positions, prefix):
            if not len(positions):
                return np.empty(0), np.empty(0), np.empty(0, dtype=object)
            if prefix == "d" and ignored_points:
                rows = pd.DataFrame([
                    effective_segment_row(traj_df, int(position), ignored_points)
                    for position in positions
                ])
            else:
                rows = traj_df.iloc[positions]
            mx, my = hex_to_mercator(
                rows[f"x_{prefix}"].to_numpy(dtype=np.int64),
                rows[f"y_{prefix}"].to_numpy(dtype=np.int64),
                rows[f"z_{prefix}"].to_numpy(dtype=np.int64),
            )
            gx, gy = mercator_wgs84_to_gcj02(mx, my)
            point_column = f"idx_{prefix}"
            sequence_numbers = np.asarray([
                self._point_sequence_number(
                    state.uid, row.get(point_column), int(position),
                    destination=prefix == "d",
                )
                for position, (_, row) in zip(positions, rows.iterrows())
            ], dtype=object)
            return np.asarray(gx), np.asarray(gy), sequence_numbers

        prev_x, prev_y, prev_sequences = _project_rows(previous_positions, "o")
        next_x, next_y, next_sequences = _project_rows(next_positions, "d")
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

        def _draw_visible(positions, xs, ys, sequence_numbers, fallback, edge,
                          near_position, near_size):
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
                artist = self.ax.scatter(
                    xs[far], ys[far],
                    c=[_saved_color(int(row_idx), fallback) for row_idx in far_positions],
                    marker="D", s=28, edgecolors=edge, linewidths=0.9,
                    alpha=CONTEXT_ALPHA, zorder=6.6,
                )
                self._register_point_hover(
                    artist, xs[far], ys[far], sequence_numbers[far],
                )
            if near.any():
                near_index = int(np.flatnonzero(near)[0])
                row_idx = int(positions[near_index])
                artist = self.ax.scatter(
                    [xs[near_index]], [ys[near_index]],
                    c=[_saved_color(row_idx, fallback)], marker="D",
                    s=near_size, edgecolors=edge, linewidths=1.5,
                    alpha=CONTEXT_ALPHA, zorder=7.25,
                )
                self._register_point_hover(
                    artist, [xs[near_index]], [ys[near_index]],
                    [sequence_numbers[near_index]],
                )

        nearest_previous = int(previous_positions[-1]) if len(previous_positions) else -1
        nearest_next = int(next_positions[0]) if len(next_positions) else -1
        _draw_visible(
            previous_positions, prev_x, prev_y, prev_sequences,
            "#d8a24a", "#7a4b00", nearest_previous, 42,
        )
        _draw_visible(
            next_positions, next_x, next_y, next_sequences,
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
        self.ax_hist.set_ylabel("")
        self.ax_hist.set_title(f"UID {state.uid}\nn={len(velocities)}", fontsize=10)
        self.ax_hist.tick_params(labelsize=8)

    def _draw_projected_road_layer(self, mode_name):
        """Draw one cached road layer inside the current viewport."""
        if not self.road_visibility.get(mode_name, True):
            return None
        projected = self._road_display.get(mode_name)
        if projected is None:
            return None
        mx, my, gx, gy = projected
        visible = (
            (mx >= self._mx_min) & (mx <= self._mx_max)
            & (my >= self._my_min) & (my <= self._my_max)
        )
        if not visible.any():
            return None
        color = ROAD_LAYER_COLORS.get(mode_name, (0.5, 0.5, 0.5))
        alpha = self._road_layer_alpha(mode_name)
        artist = self.ax.scatter(
            gx[visible], gy[visible],
            c=[color], s=6, alpha=alpha,
            marker="h", zorder=2, label=mode_name,
        )
        self._road_artists[mode_name] = artist
        return artist

    def _road_layer_alpha(self, mode_name):
        """Emphasize only the saved mode's network during path annotation."""
        if getattr(self, "annotation_mode", ANNOTATION_MODE) != ANNOTATION_PATH:
            return ROAD_OVERLAY_ALPHA
        selected_mode = getattr(self, "active_path_mode", None)
        if selected_mode not in MODE_LIST:
            return ROAD_OVERLAY_ALPHA
        return 0.46 if mode_name == selected_mode else 0.065

    def _build_hex_road_overlay(self, hex_grid):
        """六边形模式道路叠加层 —— 视口范围内的散点图（按 6 分组配色）"""
        if self._road_display:
            for mode_name in DISPLAY_MODE_LIST:
                self._draw_projected_road_layer(mode_name)
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

        for mode_name in DISPLAY_MODE_LIST:
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
                r, g, b = ROAD_LAYER_COLORS.get(
                    mode_name, (0.5, 0.5, 0.5),
                )
                gx, gy = mercator_wgs84_to_gcj02(mx_list, my_list)
                artist = self.ax.scatter(
                    gx, gy,
                    c=[(r, g, b)], s=6, alpha=self._road_layer_alpha(mode_name),
                    marker='h', zorder=2, label=mode_name,
                )
                artist.set_visible(self.road_visibility.get(mode_name, True))
                self._road_artists[mode_name] = artist

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

    def _visible_road_descriptions(self, hex_key):
        """Return enabled rendered road classes present in one hex cell."""
        try:
            cell = self.raw_mapdata[tuple(int(value) for value in hex_key)]
            code = int(cell.get("code", 0) if hasattr(cell, "get") else cell)
        except (KeyError, IndexError, TypeError, ValueError, OverflowError):
            code = 0
        descriptions = []
        seen = set()
        for key, label in decode_road_code(code):
            mode = ROAD_KEY_TO_MODE.get(key)
            if mode and self.road_visibility.get(mode, True):
                item = (mode, label)
                if item not in seen:
                    seen.add(item)
                    descriptions.append(item)
        if descriptions:
            return descriptions

        # Compatibility fallback for mapdata without a readable code field.
        normalized_key = tuple(int(value) for value in hex_key)
        for mode in DISPLAY_MODE_LIST:
            if not self.road_visibility.get(mode, True):
                continue
            road_sets = getattr(self, "road_sets", None)
            if road_sets and normalized_key in road_sets.get(mode, set()):
                descriptions.append((mode, MODE_LABELS.get(mode, mode)))
        return descriptions

    def _hovered_hex_from_event(self, event):
        """Convert a main-view GCJ Mercator mouse position to one WGS84 hex."""
        if event.xdata is None or event.ydata is None:
            return None
        try:
            from utils.geo_utils import _merc_to_wgs84
            gcj_lon, gcj_lat = _merc_to_wgs84.transform(
                float(event.xdata), float(event.ydata),
            )
            wgs_lon, wgs_lat = gcj02_to_wgs84(gcj_lon, gcj_lat)
            hex_x, hex_y, hex_z = wgs84_to_hex(wgs_lon, wgs_lat)
            return int(hex_x), int(hex_y), int(hex_z)
        except (TypeError, ValueError, OverflowError):
            return None

    def _on_main_view_click(self, event):
        """Choose a manual path start with a left click in path workflow."""
        if event.inaxes is not self.ax or event.button != 1:
            return
        if getattr(self, "annotation_mode", ANNOTATION_MODE) != ANNOTATION_PATH:
            return
        hex_key = self._hovered_hex_from_event(event)
        if hex_key is not None and self.path_start_select_callback is not None:
            self.path_start_select_callback(hex_key)

    def _on_poi_hover(self, event):
        """Show a point number, OSM POI name, or enabled road-cell classes."""
        point_hover = getattr(self, "_point_hover", None)
        poi_hover = getattr(self, "_poi_hover", None)
        road_hover = getattr(self, "_road_hover", None)
        if event.inaxes is self.ax:
            if point_hover is not None:
                for artist, records in reversed(
                        getattr(self, "_point_scatter_meta", [])):
                    contains, details = artist.contains(event)
                    indices = details.get("ind", []) if contains else []
                    if len(indices):
                        x, y, sequence = records[int(indices[0])]
                        if sequence is None:
                            continue
                        point_hover.xy = (x, y)
                        point_hover.set_text(f"序号: {sequence}")
                        point_hover.set_visible(True)
                        if poi_hover is not None:
                            poi_hover.set_visible(False)
                        if road_hover is not None:
                            road_hover.set_visible(False)
                        self.fig.canvas.draw_idle()
                        return
            if point_hover is not None and point_hover.get_visible():
                point_hover.set_visible(False)

            if poi_hover is not None:
                for artist, records in reversed(
                        getattr(self, "_poi_scatter_meta", [])):
                    contains, details = artist.contains(event)
                    indices = details.get("ind", []) if contains else []
                    if len(indices):
                        record = records[int(indices[0])]
                        poi_hover.xy = (record["_display_x"], record["_display_y"])
                        category = POI_CATEGORY_LABELS.get(
                            record["category"], record["category"],
                        )
                        poi_hover.set_text(
                            f"{category}: {record.get('name', '未命名')}"
                        )
                        poi_hover.set_visible(True)
                        if road_hover is not None:
                            road_hover.set_visible(False)
                        self.fig.canvas.draw_idle()
                        return
            if poi_hover is not None and poi_hover.get_visible():
                poi_hover.set_visible(False)

            if road_hover is not None:
                hex_key = self._hovered_hex_from_event(event)
                if hex_key is not None:
                    descriptions = self._visible_road_descriptions(hex_key)
                    if descriptions:
                        road_hover.xy = (float(event.xdata), float(event.ydata))
                        road_hover.set_text(
                            "路网: " + "  ".join(
                                f"[{road_mode}]{label}"
                                for road_mode, label in descriptions
                            )
                        )
                        road_hover.set_visible(True)
                        self.fig.canvas.draw_idle()
                        return
        changed = False
        if point_hover is not None and point_hover.get_visible():
            point_hover.set_visible(False)
            changed = True
        if poi_hover is not None and poi_hover.get_visible():
            poi_hover.set_visible(False)
            changed = True
        if road_hover is not None and road_hover.get_visible():
            road_hover.set_visible(False)
            changed = True
        if changed:
            self.fig.canvas.draw_idle()

    def _update_title(self):
        state = self.state
        key = _annotation_key(state.row)
        saved_mode = getattr(self, "labeled_modes", {}).get(key)
        if getattr(self, "annotation_mode", ANNOTATION_MODE) == ANNOTATION_PATH:
            saved_mode = getattr(self, "active_path_mode", None)
            if state.cur is None:
                title = (
                    f"路径标注 | 当前路网/模式: {saved_mode or '-'} | "
                    "左键点击主视图选择路径起点"
                )
                self.ax.set_title(title, fontsize=11, fontfamily="sans-serif")
                return
            if len(state.path_history) == 1:
                title = (
                    f"路径标注 | 当前路网/模式: {saved_mode or '-'} | "
                    f"起点: {state.path_history[0]} | 再点击终点自动规划"
                )
                self.ax.set_title(title, fontsize=11, fontfamily="sans-serif")
                return
            match = state.current_match_rate()
            road_label = saved_mode or "-"
            title = (
                f"路径标注 | 当前路网/模式: {road_label} | Steps: {state.step_count} | "
                f"Match: {match:.2%} | READY | Enter确认"
            )
        else:
            title = f"模式标注 | 当前已有模式: {saved_mode or '-'} | 数字键/按钮直接保存"
        self.ax.set_title(title, fontsize=11, fontfamily="sans-serif")

    def _init_info_panel(self):
        """Create four equally sized and consistently styled information cards."""
        card_x, card_width = 0.025, 0.95
        card_specs = (
            ("keys", 0.76, 0.22, "Keys"),
            ("segment", 0.55, 0.19, "Segment Info"),
            ("cell", 0.34, 0.19, "当前栅格"),
            ("legend", 0.02, 0.30, "路网开关 / 地点图例"),
        )
        self._info_cards = {}
        self._info_titles = {}
        for name, y, height, title in card_specs:
            card = FancyBboxPatch(
                (card_x, y), card_width, height,
                boxstyle="round,pad=0.006,rounding_size=0.012",
                transform=self.ax_info.transAxes,
                facecolor="#f7f9fb", edgecolor="#7d8994",
                linewidth=0.9, zorder=0,
            )
            self.ax_info.add_patch(card)
            self._info_cards[name] = card
            self._info_titles[name] = self.ax_info.text(
                card_x + 0.03, y + height - 0.018, title,
                transform=self.ax_info.transAxes,
                ha="left", va="top", fontsize=7.8, fontweight="bold",
                color="#34495e", zorder=2,
            )

    def _draw_legend_box(self):
        if getattr(self, "annotation_mode", ANNOTATION_MODE) == ANNOTATION_PATH:
            current_mode = getattr(self, "active_path_mode", None) or "-"
            text = (
                "Click 1/2     start / auto end\n"
                "Click again   replan end\n"
                "Arrow/QWEASD  manual adjust\n"
                "Backspace     clear route\n"
                "R             clear all\n"
                "Enter         confirm & save\n"
                f"Road mode:    {current_mode}"
            )
        else:
            text = (
                "1 TG  2 TS  3 DT\n"
                "4 GG  5 GSD 6 Other\n"
                "Number/button saves mode\n"
                "Backspace     previous OD"
            )
        self.ax_info.text(
            0.055, 0.928, text,
            transform=self.ax_info.transAxes,
            fontsize=7.2, fontfamily="monospace",
            verticalalignment="top",
            color="#27313a", zorder=2,
        )

    def _draw_segment_info(self):
        """在右侧信息栏显示当前段的 dist / time / velocity。"""
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
        if getattr(self, "annotation_mode", ANNOTATION_MODE) == ANNOTATION_PATH:
            active_mode = getattr(self, "active_path_mode", None) or "-"
            text = (
                f"Path seg: {self._uid_path_counts.get(state.uid, 0)} saved\n"
                f"Road mode:{active_mode}\n"
                f"UID OD:   {uid_od_str}\n"
                f"dist:     {dist_str} m\n"
                f"time:     {time_str} s"
            )
        else:
            text = (
                f"UID OD:   {uid_od_str}\n"
                f"dist:     {dist_str} m\n"
                f"time:     {time_str} s\n"
                f"velocity: {vel_str} km/h"
            )
        self.ax_info.text(
            0.055, 0.688, text,
            transform=self.ax_info.transAxes,
            fontsize=7.2, fontfamily="monospace",
            verticalalignment="top", horizontalalignment="left",
            color="#27313a", zorder=2,
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
        """右侧信息栏：当前光标所在栅格的完整道路属性。"""
        # 不用 monospace：Windows 等宽字体无中文字形，会显示为方框，
        # 改用默认 sans-serif（已配置微软雅黑/SimHei）以正常显示中文。
        self.cell_info_text = self.ax_info.text(
            0.055, 0.478, "",
            transform=self.ax_info.transAxes,
            fontsize=7.2, color="#27313a",
            verticalalignment="top", horizontalalignment="left",
            zorder=7,
        )
        self._update_cell_info()

    def _update_cell_info(self):
        """刷新当前光标所在栅格的道路属性（细类 + 所属可视化分组）。

        一个栅格的 code 可能同时命中多种道路（如 高速+国道），叠加层只能
        显示一种颜色，这里把完整属性列出来辅助判断标注标签。
        """
        cur = self.state.cur
        if cur is None:
            self._info_titles["cell"].set_text("当前栅格：未选择路径起点")
            self.cell_info_text.set_text("左键点击主视图中的栅格\n作为手工路径起点")
            return
        try:
            code = self.state.hex_grid[cur]["code"]
        except (KeyError, IndexError, TypeError):
            code = 0

        hits = decode_road_code(code)
        if not hits:
            body = "无道路"
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
                lines.append("  ".join(rendered))
            if others:
                lines.append("另含: " + " ".join(others))
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
            body += "\n地点: " + "  ".join(poi_text)

        self._info_titles["cell"].set_text(f"当前栅格 {cur}")
        self.cell_info_text.set_text(body)

    def refresh(self):
        """Incremental update: path line + cursor position."""
        state = self.state
        if state.path_history:
            xs = [p[0] for p in state.path_history]
            ys = [p[1] for p in state.path_history]
            zs = [p[2] for p in state.path_history]
            merc_x, merc_y = hex_to_mercator(xs, ys, zs)
            merc_x, merc_y = mercator_wgs84_to_gcj02(merc_x, merc_y)
            self.path_line.set_data(merc_x, merc_y)
            cursor_mx, cursor_my = hex_to_mercator(*state.cur)
            cursor_mx, cursor_my = mercator_wgs84_to_gcj02(cursor_mx, cursor_my)
            self.cursor.set_offsets([[cursor_mx, cursor_my]])
            start_mx, start_my = hex_to_mercator(*state.path_history[0])
            start_mx, start_my = mercator_wgs84_to_gcj02(start_mx, start_my)
            self.path_start_handle.set_offsets([[start_mx, start_my]])
        else:
            self.path_line.set_data([], [])
            self.cursor.set_offsets(np.empty((0, 2)))
            self.path_start_handle.set_offsets(np.empty((0, 2)))
        self._sync_path_artists()
        self._update_title()
        self._update_cell_info()
        self.fig.canvas.draw_idle()

    def _sync_path_artists(self):
        """Show drawing aids only while the path workflow is active."""
        is_path = getattr(self, "annotation_mode", ANNOTATION_MODE) == ANNOTATION_PATH
        path_line = getattr(self, "path_line", None)
        cursor = getattr(self, "cursor", None)
        path_start = getattr(self, "path_start_handle", None)
        if path_line is not None:
            path_line.set_visible(is_path)
        if cursor is not None:
            cursor.set_visible(is_path and self.state.cur is not None)
        if path_start is not None:
            path_start.set_visible(is_path and bool(self.state.path_history))


# ========================== Controller ==========================

class LabelController:
    """Coordinates state, renderer, and keyboard input."""

    def __init__(self, state: LabelState, renderer: PathRenderer,
                 output_dir: str, batch_mode: bool, current_idx: int,
                 navigate_callback=None):
        self.state = state
        self.renderer = renderer
        self.output_dir = output_dir
        self.batch_mode = batch_mode
        self.current_idx = current_idx
        self.saved = False
        self.next_requested = False
        self.go_back_requested = False
        self.navigate_callback = navigate_callback
        self.annotation_mode = ANNOTATION_MODE
        self._all_road_mapdata = state.multi_mapdata
        self.renderer.annotation_mode = ANNOTATION_MODE
        if not isinstance(getattr(self.renderer, "labeled_paths", None), dict):
            self.renderer.labeled_paths = {}

    def set_segment(self, state, current_idx):
        """Point the existing controller at a newly rendered OD."""
        if getattr(self.renderer, "editing_path_key", None) is not None \
                and (int(state.uid) != int(self.state.uid)
                     or int(current_idx) != int(self.current_idx)):
            self.renderer.editing_path_key = None
        self.state = state
        self.current_idx = current_idx
        self.saved = False
        self.next_requested = False
        self.go_back_requested = False
        self._prepare_state_for_workflow()

    def _prepare_state_for_workflow(self):
        """Prepare a clean state for the selected annotation workflow."""
        state = self.state
        if self.annotation_mode != ANNOTATION_PATH:
            state.multi_mapdata = self._all_road_mapdata
            state.reset()
            return

        active_mode = getattr(self.renderer, "active_path_mode", None)
        road_sets = getattr(self.renderer, "road_sets", None)
        if isinstance(road_sets, dict) and active_mode in road_sets:
            state.multi_mapdata = road_sets[active_mode]
        else:
            state.multi_mapdata = self._all_road_mapdata
        state.clear_path()

    def toggle_annotation_mode(self):
        """Switch between direct mode labels and manual path drawing."""
        target_mode = (
            ANNOTATION_PATH if self.annotation_mode == ANNOTATION_MODE
            else ANNOTATION_MODE
        )
        self.set_annotation_mode(target_mode)

    def set_annotation_mode(self, annotation_mode):
        if annotation_mode not in (ANNOTATION_MODE, ANNOTATION_PATH):
            return False
        if annotation_mode == self.annotation_mode:
            return True

        target_index = self.current_idx
        if annotation_mode == ANNOTATION_PATH:
            ignored = getattr(self.renderer, "ignored_points", set())
            excluded = getattr(self.renderer, "excluded_uids", set())
            current_key = _annotation_key(self.state.row)
            current_eligible = (
                current_key not in ignored
                and int(self.state.uid) not in excluded
            )
            if not current_eligible:
                traj_df = getattr(self.renderer, "traj_df", None)
                target_index = next_nonignored_index(
                    traj_df, ignored, self.current_idx, 1, excluded,
                )
                if target_index >= len(traj_df):
                    target_index = next_nonignored_index(
                        traj_df, ignored, self.current_idx, -1, excluded,
                    )
                if not 0 <= target_index < len(traj_df):
                    print("  [PATH] No reviewable UID is available for path annotation")
                    return False

            selected_key = getattr(self.renderer, "selected_path_key", None)
            selected_path_mode = getattr(
                self.renderer, "labeled_path_modes", {},
            ).get(selected_key)
            current_mode = self.renderer.labeled_modes.get(current_key)
            preferred_mode = (
                selected_path_mode if selected_path_mode in MODE_LIST
                else current_mode if current_mode in MODE_LIST
                else getattr(self.renderer, "active_path_mode", None)
            )

        self.annotation_mode = annotation_mode
        self.renderer.annotation_mode = annotation_mode
        if annotation_mode == ANNOTATION_PATH:
            self.renderer.enter_path_road_mode(preferred_mode)
        else:
            self.renderer.editing_path_key = None
            self.renderer.leave_path_road_mode()
        if self.renderer.segment_select_callback is not None:
            self.renderer.segment_select_callback(target_index)
        else:
            self._prepare_state_for_workflow()
            self.renderer.show_segment(self.state, self.current_idx)
        print(f"  [WORKFLOW] {'路径标注' if annotation_mode == ANNOTATION_PATH else '模式标注'}")
        return True

    def _current_is_available(self):
        current_key = _annotation_key(self.state.row)
        ignored_points = getattr(self.renderer, "ignored_points", set())
        if isinstance(ignored_points, (set, frozenset)) \
                and current_key in ignored_points:
            print("  [IGNORED] Restore this point before labeling its OD")
            return False
        traj_df = getattr(self.renderer, "traj_df", None)
        if isinstance(traj_df, pd.DataFrame) \
                and not segment_is_reviewable(
                    traj_df, self.current_idx, ignored_points,
                ):
            print("  [IGNORED] Restore the OD endpoint before labeling this OD")
            return False
        excluded_uids = getattr(self.renderer, "excluded_uids", set())
        if isinstance(excluded_uids, (set, frozenset)) \
                and int(self.state.uid) in excluded_uids:
            print("  [EXCLUDED] Restore this UID before labeling it")
            return False
        return True

    def save_mode(self, label):
        """Save one mode from either a number key or a clickable button."""
        if self.annotation_mode != ANNOTATION_MODE:
            print("  [PATH] Number keys and mode buttons are disabled during path annotation")
            return
        canonical_mode = _canonical_label_mode(label)
        if canonical_mode is None or not self._current_is_available():
            return
        if not self._finalize(canonical_mode):
            return
        self.saved = True
        print(f"  [LABELED] #{self.current_idx} -> {canonical_mode}")
        self._advance_after_save(canonical_mode)

    def save_path(self):
        """Save one user-defined truth segment under the active road mode."""
        if self.annotation_mode != ANNOTATION_PATH or not self._current_is_available():
            return
        active_mode = getattr(self.renderer, "active_path_mode", None)
        if active_mode not in MODE_LIST:
            print("  [PATH] Select one road network before saving")
            return
        if len(self.state.path_history) < 2:
            print("  [PATH] Click a start and an end to generate a route first")
            return
        editing_key = getattr(self.renderer, "editing_path_key", None)
        segment_key = self._finalize_path(active_mode)
        if not segment_key:
            return
        self.saved = True
        print(
            f"  [{'PATH UPDATED' if editing_key else 'PATH SAVED'}] "
            f"uid={segment_key[0]} "
            f"segment={segment_key[1]} {active_mode} "
            f"({self.state.step_count} steps)"
        )
        self.state.clear_path()
        self.renderer.show_segment(self.state, self.current_idx)

    def select_path_start(self, hex_key):
        """Use first/next map clicks as route start/end and preview A* output."""
        if self.annotation_mode != ANNOTATION_PATH:
            return False
        road_cells = self.state.multi_mapdata
        snapped = nearest_road_hex(hex_key, road_cells)
        if snapped is None:
            print("  [PATH] No active-mode road cell was found near the click")
            return False
        if not self.state.path_history:
            if not self.state.set_path_start(snapped):
                print("  [PATH] The selected hex is outside the valid map")
                return False
            self.renderer.refresh()
            suffix = f" (snapped from {tuple(hex_key)})" if snapped != tuple(hex_key) else ""
            print(f"  [PATH START] {self.state.cur}{suffix}")
            return True

        start = self.state.path_history[0]
        route = find_road_route(start, snapped, road_cells)
        if route is None:
            print("  [PATH] No connected route was found on this mode's road network")
            return False
        if not self.state.restore_path(route):
            print("  [PATH] The selected hex is outside the valid map")
            return False
        self.renderer.refresh()
        suffix = f" (snapped from {tuple(hex_key)})" if snapped != tuple(hex_key) else ""
        print(f"  [ROUTE] {start} -> {snapped}, {len(route) - 1} steps{suffix}")
        return True

    def _selected_saved_path_key(self):
        """Return the selected path only when it still belongs to this UID."""
        key = getattr(self.renderer, "selected_path_key", None)
        if not isinstance(key, tuple) or len(key) != 2:
            return None
        key = (int(key[0]), int(key[1]))
        if key[0] != int(self.state.uid):
            return None
        labeled_paths = getattr(self.renderer, "labeled_paths", {})
        return key if isinstance(labeled_paths, dict) and key in labeled_paths else None

    def reset_selected_path(self):
        """Start redrawing the selected saved path without deleting it first."""
        if self.annotation_mode != ANNOTATION_PATH:
            return False
        key = self._selected_saved_path_key()
        if key is None:
            print("  [PATH] Select one saved path before resetting it")
            return False
        self.renderer.editing_path_key = key
        self.state.clear_path()
        self.renderer.show_segment(self.state, self.current_idx)
        print(
            f"  [PATH RESET] uid={key[0]} segment={key[1]}; "
            "choose a new start/end, then press Enter"
        )
        return True

    def delete_selected_path(self):
        """Delete the selected saved path and refresh its dated export copy."""
        if self.annotation_mode != ANNOTATION_PATH:
            return False
        key = self._selected_saved_path_key()
        if key is None:
            print("  [PATH] Select one saved path before deleting it")
            return False
        try:
            csv_path, deleted_count = delete_truth_path_segment(
                self.output_dir, key,
            )
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            print(f"  [WARN] Failed to delete selected path: {exc}")
            return False
        if not deleted_count:
            print(
                f"  [PATH] uid={key[0]} segment={key[1]} "
                "was already absent from CSV"
            )
            return False
        self.renderer.clear_labeled_path(key)
        self.renderer.renumber_labeled_paths(key[0])
        self.state.clear_path()
        self._refresh_path_copy()
        self.renderer.refresh_uid_segment_list()
        self.renderer.show_segment(self.state, self.current_idx)
        self.saved = True
        print(f"  [PATH DELETED] uid={key[0]} segment={key[1]}")
        print(f"  -> CSV: {csv_path}")
        return True

    def _advance_after_save(self, label):
        if self.batch_mode:
            self.next_requested = True
            if self.navigate_callback is not None:
                self.navigate_callback(1)
            else:
                plt.close(self.renderer.fig)
        else:
            self.renderer.ax.set_title(
                self.renderer.ax.get_title() + f" [{label}]",
                fontsize=11, fontfamily="sans-serif",
            )
            self.renderer.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key is None:
            return

        uid_search_box = getattr(self.renderer, "_uid_search_box", None)
        if uid_search_box is not None \
                and getattr(uid_search_box, "capturekeystrokes", False) is True:
            return

        key = event.key.lower()

        # Number keys save the mode immediately; Enter is no longer required.
        if key in LABEL_OPTIONS:
            self.save_mode(LABEL_OPTIONS[key])
            return

        if key in ("enter", "return"):
            if self.annotation_mode == ANNOTATION_PATH:
                self.save_path()
            return

        if key not in HEX_KEY_MAP:
            return

        action, arg = HEX_KEY_MAP[key]

        if action == "move":
            if self.annotation_mode != ANNOTATION_PATH:
                return
            if self.state.cur is None:
                print("  [PATH] Left-click the main view to choose a path start first")
                return
            if self.state.can_move(*HEX_DIRS[arg]):
                self.state.apply_move(arg)
                self.renderer.refresh()

        elif action == "undo":
            if self.annotation_mode == ANNOTATION_PATH \
                    and self.state.path_history:
                if len(self.state.path_history) > 1:
                    self.state.set_path_start(self.state.path_history[0])
                    print("  [PATH] Route cleared; click a new end")
                else:
                    self.state.clear_path()
                self.renderer.refresh()
            elif self.annotation_mode == ANNOTATION_PATH \
                    and self._selected_saved_path_key() is not None:
                self.delete_selected_path()
            elif len(self.state.path_history) <= 1:
                # No steps taken — go back to previous trajectory's label
                if self.current_idx > 0:
                    self.go_back_requested = True
                    print(f"  Going back to re-label previous trajectory #{self.current_idx - 1}")
                    if self.navigate_callback is not None:
                        self.navigate_callback(-1)
                    else:
                        plt.close(self.renderer.fig)
                else:
                    print(f"  Already at first trajectory, cannot go back")
            elif self.state.undo():
                self.renderer.refresh()

        elif action == "reset":
            if self.annotation_mode != ANNOTATION_PATH:
                return
            if self.state.path_history:
                self.state.clear_path()
                self.renderer.refresh()
            else:
                self.reset_selected_path()

    def _finalize(self, label):
        """Upsert one OD mode label without changing independent path truth."""
        state = self.state
        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = os.path.join(self.output_dir, "traj_labeled.csv")
        row = state.row
        skip_cols = {
            "order", "mode", "path", "traj", "success", "match", "steps",
        }
        source_record = {}
        for col in row.index:
            if col not in skip_cols:
                source_record[col] = row[col]

        existing = self._read_active_labels(csv_path)
        key = _annotation_key(source_record)
        mask = self._matching_record_mask(existing, key)
        if mask.any():
            record = existing.loc[mask].iloc[-1].to_dict()
            record.update(source_record)
        else:
            record = source_record
        record["mode"] = label
        self._write_active_record(csv_path, existing, mask, record)

        if key is not None:
            self.renderer.set_labeled_mode(key, label)
        self._after_csv_write(csv_path)
        return True

    def _finalize_path(self, active_mode):
        """Append a route or replace the path currently being redrawn."""
        anchor_key = _annotation_key(self.state.row)
        anchor_idx_o = anchor_key[1] if anchor_key is not None else -1
        editing_key = getattr(self.renderer, "editing_path_key", None)
        if not isinstance(editing_key, tuple) or len(editing_key) != 2:
            editing_key = None
        csv_path, segment_key = write_truth_path_segment(
            self.output_dir,
            self.state.uid,
            active_mode,
            anchor_idx_o,
            self.state.path_history,
            self.state.current_match_rate(),
            self.state.step_count,
            segment_key=editing_key,
        )
        self.renderer.set_labeled_path(
            segment_key, self.state.path_history, active_mode,
            metadata={
                "anchor_idx_o": anchor_idx_o,
                "steps": self.state.step_count,
                "match": self.state.current_match_rate(),
            },
        )
        self.renderer.editing_path_key = None
        self._refresh_path_copy()
        self.renderer.refresh_uid_segment_list()
        print(f"  -> CSV: {csv_path}")
        return segment_key

    @staticmethod
    def _read_active_labels(csv_path):
        if not os.path.exists(csv_path):
            return pd.DataFrame()
        try:
            return pd.read_csv(csv_path, encoding="utf-8")
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            print(f"  [WARN] Failed to read active labels: {exc}")
            return pd.DataFrame()

    @staticmethod
    def _matching_record_mask(existing, key):
        if existing.empty or key is None \
                or not {"uid", "idx_o"}.issubset(existing.columns):
            return np.zeros(len(existing), dtype=bool)
        uid_values = pd.to_numeric(existing["uid"], errors="coerce")
        idx_values = pd.to_numeric(existing["idx_o"], errors="coerce")
        return ((uid_values == int(key[0])) & (idx_values == int(key[1]))).to_numpy()

    @staticmethod
    def _write_active_record(csv_path, existing, mask, record):
        path_columns = ["path", "traj", "success", "match", "steps"]
        existing = existing.drop(columns=path_columns, errors="ignore")
        record = dict(record)
        for column in path_columns:
            record.pop(column, None)
        remaining = existing.loc[~mask] if len(existing) else existing
        df_new = pd.DataFrame([record])
        out_df = (
            pd.concat([remaining, df_new], ignore_index=True)
            if not remaining.empty else df_new
        )
        front_cols = [column for column in ("uid", "idx_o", "idx_d")
                      if column in out_df.columns]
        other_cols = [column for column in out_df.columns if column not in front_cols]
        out_df = sort_labeled_records(out_df[front_cols + other_cols])
        csv_temp = csv_path + ".tmp"
        out_df.to_csv(csv_temp, index=False, encoding="utf-8")
        os.replace(csv_temp, csv_path)

    def _after_csv_write(self, csv_path):
        export_date = getattr(self.renderer, "export_date", None)
        if export_date:
            try:
                write_labeled_data_copy(
                    self.output_dir, export_date,
                    getattr(self.renderer, "excluded_uids", set()),
                )
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                print(f"  [WARN] Failed to update labeled-data copy: {exc}")
        self.renderer.refresh_uid_segment_list()
        print(f"  -> CSV: {csv_path}")

    def _refresh_path_copy(self):
        export_date = getattr(self.renderer, "export_date", None)
        if not export_date:
            return
        try:
            write_path_labeled_copy(
                self.output_dir, export_date,
                getattr(self.renderer, "excluded_uids", set()),
            )
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            print(f"  [WARN] Failed to update path-label copy: {exc}")


# ========================== Main Loop ==========================

def run_single(state, raw_mapdata, output_dir, batch_mode, idx,
               road_sets=None, traj_df=None, pois=None,
               labeled_modes=None, labeled_paths=None, labeled_path_modes=None,
               labeled_path_metadata=None,
               ignored_points=None, excluded_uids=None, export_date=None):
    """Run labeling for one trajectory. Returns (next_idx, keep_going)."""
    renderer = PathRenderer(state, raw_mapdata, road_sets=road_sets,
                             traj_df=traj_df, current_idx=idx,
                             output_dir=output_dir, pois=pois,
                             labeled_modes=labeled_modes,
                             labeled_paths=labeled_paths,
                             labeled_path_modes=labeled_path_modes,
                             labeled_path_metadata=labeled_path_metadata,
                             ignored_points=ignored_points,
                             excluded_uids=excluded_uids,
                             export_date=export_date)
    controller = LabelController(
        state, renderer, output_dir, batch_mode, idx,
    )

    def select_segment(target):
        if traj_df is None or not 0 <= int(target) < len(traj_df):
            return
        target = int(target)
        display_row = effective_segment_row(
            traj_df, target, renderer.ignored_points,
        )
        next_state = LabelState(
            display_row, controller._all_road_mapdata, hex_grid=raw_mapdata,
        )
        controller.set_segment(next_state, int(target))
        renderer.show_segment(next_state, int(target))
        _print_segment_status(int(target), next_state, controller.annotation_mode)

    renderer.set_segment_select_callback(select_segment)
    renderer.set_label_select_callback(controller.save_mode)
    renderer.set_annotation_toggle_callback(controller.toggle_annotation_mode)
    renderer.set_path_start_select_callback(controller.select_path_start)

    renderer.fig.canvas.mpl_connect("key_press_event", controller.on_key)

    def on_close(event):
        if not controller.saved:
            print(f"  [WARN] window closed, #{controller.current_idx} not saved")

    renderer.fig.canvas.mpl_connect("close_event", on_close)

    print(f"\n{'='*60}")
    print(f"#{idx}  order={state.order}  mode={state.mode}")
    print(f"Start: {state.start}  ->  End: {state.end}")
    print("Mode annotation: 1-6/button=save mode; Path annotation: "
          "W/A/S/D/Q/E=draw, Backspace=undo, R=reset, Enter=save path")
    print(f"{'='*60}")

    plt.show(block=True)

    if controller.next_requested:
        return idx + 1, True
    elif controller.go_back_requested:
        return max(0, idx - 1), True
    else:
        return idx, False


def _print_segment_status(idx, state, annotation_mode=ANNOTATION_MODE):
    print(f"\n{'='*60}")
    print(f"#{idx}  order={state.order}  mode={state.mode}")
    print(f"Start: {state.start}  ->  End: {state.end}")
    if annotation_mode == ANNOTATION_PATH:
        print("Path annotation: W/A/S/D/Q/E=draw  Backspace=undo  "
              "R=reset  Enter=save path")
    else:
        print("Mode annotation: 1-6 or click a mode button to save")
    print(f"{'='*60}")


def run_continuous(start_idx, make_state, raw_mapdata, output_dir,
                   road_sets, traj_df, pois, labeled_modes, labeled_paths,
                   labeled_path_modes, labeled_path_metadata, ignored_points,
                   excluded_uids, export_date):
    """Run a batch in one persistent Matplotlib window.

    Static data and offline map tiles stay cached in the same process. Moving
    between ODs clears and redraws the two axes without destroying the GUI
    window or reconnecting keyboard/mouse handlers.
    """
    state = make_state(traj_df.iloc[start_idx])
    renderer = PathRenderer(
        state, raw_mapdata, road_sets=road_sets, traj_df=traj_df,
        current_idx=start_idx, output_dir=output_dir, pois=pois,
        labeled_modes=labeled_modes, ignored_points=ignored_points,
        labeled_paths=labeled_paths,
        labeled_path_modes=labeled_path_modes,
        labeled_path_metadata=labeled_path_metadata,
        excluded_uids=excluded_uids, export_date=export_date,
    )
    controller = LabelController(
        state, renderer, output_dir, batch_mode=True, current_idx=start_idx,
    )

    def show_target(target):
        if target is None:
            return
        target = int(target)
        if target >= len(traj_df):
            controller.saved = True
            print(f"\nAll {len(traj_df)} trajectories labeled!")
            plt.close(renderer.fig)
            return
        if target < 0:
            print("  Already at first trajectory, cannot go back")
            return

        next_state = make_state(traj_df.iloc[target])
        controller.set_segment(next_state, target)
        renderer.show_segment(next_state, target)
        _print_segment_status(target, next_state, controller.annotation_mode)

    def navigate(delta):
        target = next_nonignored_index(
            traj_df, renderer.ignored_points, controller.current_idx, delta,
            renderer.excluded_uids,
        )
        show_target(target)

    controller.navigate_callback = navigate
    renderer.set_segment_select_callback(show_target)
    renderer.set_label_select_callback(controller.save_mode)
    renderer.set_annotation_toggle_callback(controller.toggle_annotation_mode)
    renderer.set_path_start_select_callback(controller.select_path_start)
    renderer.fig.canvas.mpl_connect("key_press_event", controller.on_key)

    def on_close(event):
        if not controller.saved:
            print(f"  [WARN] window closed, #{controller.current_idx} not saved")

    renderer.fig.canvas.mpl_connect("close_event", on_close)
    _print_segment_status(start_idx, state, controller.annotation_mode)
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
    for m in DISPLAY_MODE_LIST:
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
    multi_mapdata = (
        set().union(*(road_sets.get(mode, set()) for mode in MODE_LIST))
        if road_sets else set()
    )

    output_dir = args.output
    try:
        reordered = normalize_label_storage(output_dir)
        if reordered:
            print("Normalized OD mode labels (path fields moved to separate storage)")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"  [WARN] Failed to normalize label CSV storage: {exc}")
    labeled_modes = load_labeled_modes(output_dir)
    if labeled_modes:
        print(f"Loaded existing labels for map feedback: {len(labeled_modes):,}")
    ignored_points = load_ignored_points(output_dir)
    if ignored_points:
        print(f"Loaded ignored sampled points: {len(ignored_points):,}")
    try:
        removed_paths = normalize_path_label_storage(output_dir)
        if removed_paths:
            print(f"Normalized user-defined path storage: {removed_paths:,}")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"  [WARN] Failed to normalize path truth: {exc}")
    labeled_paths, labeled_path_modes = load_labeled_path_data(output_dir)
    labeled_path_metadata = load_labeled_path_metadata(output_dir)
    if labeled_paths:
        print(f"Loaded user-defined truth paths: {len(labeled_paths):,}")
    excluded_uids = load_excluded_uids(output_dir)
    if excluded_uids:
        print(f"Loaded excluded UID trajectories: {len(excluded_uids):,}")
    export_date = derive_labeled_data_date(args.csv, traj_df)
    try:
        export_path, export_rows = write_labeled_data_copy(
            output_dir, export_date, excluded_uids,
        )
        if os.path.exists(export_path):
            print(f"Labeled-data copy: {export_path} ({export_rows:,} rows)")
        path_export, path_export_rows = write_path_labeled_copy(
            output_dir, export_date, excluded_uids,
        )
        if os.path.exists(path_export):
            print(f"Path-label copy: {path_export} ({path_export_rows:,} rows)")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"  [WARN] Failed to build labeled-data copy: {exc}")

    def make_state(row):
        position = int(traj_df.index.get_indexer([row.name])[0])
        effective_row = effective_segment_row(traj_df, position, ignored_points)
        return LabelState(effective_row, multi_mapdata, hex_grid=raw_mapdata)

    if args.index is not None:
        if args.index < 0 or args.index >= len(traj_df):
            print(f"Error: index out of range [0, {len(traj_df)-1}]")
            sys.exit(1)
        state = make_state(traj_df.iloc[args.index])
        run_single(state, raw_mapdata, output_dir, batch_mode=False,
                   idx=args.index, road_sets=road_sets, traj_df=traj_df, pois=pois,
                   labeled_modes=labeled_modes, labeled_paths=labeled_paths,
                   labeled_path_modes=labeled_path_modes,
                   labeled_path_metadata=labeled_path_metadata,
                   ignored_points=ignored_points, excluded_uids=excluded_uids,
                   export_date=export_date)

    else:
        start_idx = first_unlabeled_index(
            traj_df, labeled_modes, ignored_points, excluded_uids,
        )
        print(f"Opening annotation view at first unfinished OD: #{start_idx}")

        run_continuous(
            start_idx, make_state, raw_mapdata, output_dir,
            road_sets, traj_df, pois, labeled_modes, labeled_paths,
            labeled_path_modes, labeled_path_metadata, ignored_points,
            excluded_uids, export_date,
        )


if __name__ == "__main__":
    main()
