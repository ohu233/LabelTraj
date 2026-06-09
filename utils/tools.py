import numpy as np

MODE_LIST = ["GSD", "GG", "TS", "TG"]

# ============================================================
# 200m 六边形网格地图处理 (hex_grid.pkl)
# ============================================================

HEX_MODE_BITS = {
    "TG":  0b01000000,   # bit 6: has_gt（高铁）
    "GG":  0b00011000,   # bit 3+4: has_gs_sfz + has_gs（高速）
    "GSD": 0b10100100,   # bit 2+5+7: has_sd + has_gd + has_xd（省道+国道+县道）
    "TS":  0b00000010,   # bit 1: has_pt（普铁）
}


def hex_mapdata_to_road_sets(hex_grid):
    """将 hex_grid.pkl 字典转换为各模式的六边形坐标集合

    Returns:
        dict: {"TG": set(), "GG": set(), "GSD": set(), "TS": set()}
              每个 set 包含该模式对应的 (x, y, z) 六边形坐标
    """
    road_sets = {mode: set() for mode in MODE_LIST}
    for (x, y, z), val in hex_grid.items():
        code = val["code"]
        for mode in MODE_LIST:
            if code & HEX_MODE_BITS[mode]:
                road_sets[mode].add((x, y, z))
    return road_sets


def build_multi_mapdata_hex(road_sets, selected_mode):
    """根据选中模式取 road_set 并集

    Args:
        road_sets: hex_mapdata_to_road_sets() 的返回值
        selected_mode: 字符串 "TG" 或列表 ["TG", "GG"]

    Returns:
        set: 包含选中模式所有道路的六边形坐标集合
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
