import numpy as np

# ============================================================
# 13 种道路类型 → code 位掩码（hex_grid_2025.pkl）
# ------------------------------------------------------------
# bit 顺序按用户指定（注意：与 *Name 字段顺序相反）：
#   bit 0  水路       SL
#   bit 1  在建       ZJ
#   bit 2  其他道路   QT
#   bit 3  四级道路   L4
#   bit 4  三级道路   L3
#   bit 5  二级道路   L2
#   bit 6  环路       HL
#   bit 7  省道       SD
#   bit 8  国道       GD
#   bit 9  高速       GS
#   bit 10 地铁       DT
#   bit 11 铁路       TL
#   bit 12 高铁       GT
# 路网判断一律用 code 的 bit，**禁止用 *Name 字段**。
# ============================================================

# 单类型 bit 掩码（按位）
ROAD_BITS = {
    "SL":  1 << 0,   # 水路
    "ZJ":  1 << 1,   # 在建
    "QT":  1 << 2,   # 其他道路
    "L4":  1 << 3,   # 四级道路
    "L3":  1 << 4,   # 三级道路
    "L2":  1 << 5,   # 二级道路
    "HL":  1 << 6,   # 环路
    "SD":  1 << 7,   # 省道
    "GD":  1 << 8,   # 国道
    "GS":  1 << 9,   # 高速
    "DT":  1 << 10,  # 地铁
    "TL":  1 << 11,  # 铁路
    "GT":  1 << 12,  # 高铁
}

# 中文标签（供图例 / 提示使用）
ROAD_LABELS = {
    "GT": "高铁", "TL": "铁路", "DT": "地铁", "GS": "高速",
    "GD": "国道", "SD": "省道", "HL": "环路",
    "L2": "二级道路", "L3": "三级道路", "L4": "四级道路",
    "QT": "其他道路", "ZJ": "在建", "SL": "水路",
}

# ============================================================
# 可视化分组：将细类合并为展示用的大类
# ------------------------------------------------------------
#   GT   = 高铁
#   TL   = 铁路
#   DT   = 地铁
#   GS   = 高速
#   GSD  = 国道 + 省道 + 环路（合并）
#   EJ   = 二级道路（L2）
# 三级道路(L3) / 四级道路(L4) / 其他(QT) / 在建(ZJ) / 水路(SL)
# 默认不渲染、不参与匹配。
# ============================================================
MODE_LIST = ["GT", "TL", "DT", "GS", "GSD", "EJ"]

# 每个 MODE 由哪些细类 bit 合并而成
MODE_BITS = {
    "GT":  ROAD_BITS["GT"],
    "TL":  ROAD_BITS["TL"],
    "DT":  ROAD_BITS["DT"],
    "GS":  ROAD_BITS["GS"],
    "GSD": ROAD_BITS["GD"] | ROAD_BITS["SD"] | ROAD_BITS["HL"],
    "EJ":  ROAD_BITS["L2"],
}

# MODE 的中文标签（合并类合并命名）
MODE_LABELS = {
    "GT":  "高铁",
    "TL":  "铁路",
    "DT":  "地铁",
    "GS":  "高速",
    "GSD": "国道/省道/环路",
    "EJ":  "二级道路",
}

# 默认不参与路网渲染与匹配的细类
EXCLUDED_BITS = (ROAD_BITS["L3"] | ROAD_BITS["L4"] | ROAD_BITS["QT"]
                 | ROAD_BITS["ZJ"] | ROAD_BITS["SL"])


def hex_mapdata_to_road_sets(hex_grid):
    """将 hex_grid 按 MODE 拆分为各分组的六边形坐标集合。

    仅依据每格的 code 位掩码判断，不读取 *Name 字段。
    每个格子的 code 可能同时命中多个分组。

    若 hex_grid 是 geo_utils.HexGridProxy（密集数组缓存），用向量化
    快速构建；否则回退到逐格遍历（兼容旧式 dict）。

    Returns:
        dict: {mode: set((x,y,z), ...)}，键为 MODE_LIST 中的 6 个分组
    """
    road_sets = {mode: set() for mode in MODE_LIST}

    # 快速路径：HexGridProxy 背后的密集 code 数组（向量化，~1s）
    try:
        from utils import geo_utils
        if isinstance(hex_grid, geo_utils.HexGridProxy):
            code_arr = geo_utils._HEX_CODE
            x_off = geo_utils._HEX_X_OFFSET
            z_off = geo_utils._HEX_Z_OFFSET
            valid_mask = ~np.isnan(geo_utils._HEX_LON)
            # 每个有效 cell 的 (x, z)
            ixs, izs = np.nonzero(valid_mask)
            codes = code_arr[ixs, izs]
            xs_all = ixs + x_off
            zs_all = izs + z_off
            ys_all = -xs_all - zs_all
            for mode in MODE_LIST:
                hit = (codes & MODE_BITS[mode]) != 0
                if hit.any():
                    road_sets[mode] = set(zip(xs_all[hit].tolist(),
                                              ys_all[hit].tolist(),
                                              zs_all[hit].tolist()))
            return road_sets
    except Exception:
        pass  # 回退到通用遍历

    # 通用路径：逐格遍历（兼容 dict）
    for (x, y, z), val in hex_grid.items():
        code = int(val.get("code", 0) or 0)
        if not code:
            continue
        for mode in MODE_LIST:
            if code & MODE_BITS[mode]:
                road_sets[mode].add((x, y, z))
    return road_sets


def build_multi_mapdata_hex(road_sets, selected_mode):
    """根据选中分组取 road_set 并集

    Args:
        road_sets: hex_mapdata_to_road_sets() 的返回值
        selected_mode: 字符串 "GT" 或列表 ["GT", "GS"]

    Returns:
        set: 包含选中分组所有道路的六边形坐标集合
    """
    if isinstance(selected_mode, str):
        selected_mode = [selected_mode]
    result = set()
    for m in selected_mode:
        if m in road_sets:
            result |= road_sets[m]
    return result


def calculate_match_rate_hex(traj_points, road_set):
    """计算六边形轨迹点在道路上的匹配率

    Args:
        traj_points: [(x,y,z), ...] 路径历史
        road_set: build_multi_mapdata_hex() 返回的 set

    Returns:
        float: 在路上的点数占比
    """
    if not traj_points:
        return 0.0
    on_road = 0
    for p in traj_points:
        if p is None or len(p) < 3:
            continue
        key = (int(p[0]), int(p[1]), int(p[2]))
        if key in road_set:
            on_road += 1
    return on_road / len(traj_points)
