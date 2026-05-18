import pickle
import numpy as np
from pyproj import Transformer

# 加载 pkl
with open("hex_grid.pkl", "rb") as f:
    grid = pickle.load(f)

# 获取原点 (0,0,0) 的经纬度，用于推算投影坐标原点
origin_lon = grid[(0, 0, 0)]["lon"]
origin_lat = grid[(0, 0, 0)]["lat"]

# WGS84 → Beijing 1954 GK CM 111E
to_bj = Transformer.from_crs("EPSG:4326", "EPSG:2434", always_xy=True)
origin_x, origin_y = to_bj.transform(origin_lon, origin_lat)
SIDE = 200.0
SQRT3 = np.sqrt(3)


def lonlat_to_hex(lon, lat):
    """输入 WGS84 经纬度，返回最近的六边形栅格坐标 (x, y, z)"""
    px, py = to_bj.transform(lon, lat)
    cx = px - origin_x
    cy = py - origin_y

    # flat-top 六边形：世界坐标 → 轴向坐标 (分数)
    q = (cx * 2.0 / 3.0) / SIDE
    r = (-cx / 3.0 + SQRT3 / 3.0 * cy) / SIDE

    # 立方体坐标取整
    s = -q - r
    qi = round(q)
    ri = round(r)
    si = round(s)

    qd = abs(qi - q)
    rd = abs(ri - r)
    sd = abs(si - s)

    if qd > rd and qd > sd:
        qi = -ri - si
    elif rd > sd:
        ri = -qi - si
    else:
        si = -qi - ri

    return (qi, si, ri)  # (x, y, z)


def query(lon, lat):
    """查询经纬度对应的六边形栅格信息"""
    key = lonlat_to_hex(lon, lat)
    val = grid.get(key)
    return key, val


# CLI
if __name__ == "__main__":
    # 自检：用 pkl 里几个格子的经纬度反查，验证正确性
    print("自检 (用格子中心点反查)...")
    test_keys = [(0, 0, 0), (100, -50, -50), (500, -250, -250)]
    ok = 0
    for k in test_keys:
        if k in grid:
            v = grid[k]
            qk, qv = query(v["lon"], v["lat"])
            match = "OK" if qk == k else f"FAIL (got {qk})"
            if match == "OK":
                ok += 1
            print(f"  {k} → lon={v['lon']:.6f} lat={v['lat']:.6f} → {qk}  {match}")
    print(f"  {ok}/{len(test_keys)} 通过")

    print()
    import sys

    # 命令行参数模式
    if len(sys.argv) == 3:
        lon = float(sys.argv[1])
        lat = float(sys.argv[2])
        key, val = query(lon, lat)
        print(f"查询: ({lon}, {lat})")
        print(f"栅格坐标: {key}")
        if val:
            print(f"中心经纬度: ({val['lon']:.6f}, {val['lat']:.6f})")
            print(f"道路编码: {val['code']} (0b{val['code']:08b})")
        else:
            print("该坐标不在数据范围内")
        sys.exit(0)

    # 交互模式
    print("=" * 50)
    print("六边形栅格坐标查询")
    print("输入经纬度，格式: 经度 纬度  (如 120.0 30.0)")
    print("输入 q 退出")
    print("=" * 50)

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() == "q":
            break
        parts = line.replace(",", " ").split()
        if len(parts) != 2:
            print("格式错误，请输入: 经度 纬度")
            continue
        try:
            lon = float(parts[0])
            lat = float(parts[1])
        except ValueError:
            print("请输入有效的数字")
            continue

        key, val = query(lon, lat)
        print(f"栅格坐标: {key}")
        if val:
            print(f"中心经纬度: ({val['lon']:.6f}, {val['lat']:.6f})")
            print(f"道路编码: {val['code']} (0b{val['code']:08b})")
        else:
            print("该坐标不在数据范围内")
