"""
网格坐标 ↔ Beijing 1954 GK CM 111E ↔ WGS84 ↔ Web Mercator 转换。

坐标系说明:
  网格 (grid_x, grid_y): 0-563 列, 0-528 行, 每格 1000m
  Beijing 1954 3 Degree GK CM 111E (EPSG:2434): 投影米制坐标
  WGS84 (EPSG:4326): 经纬度
  Web Mercator (EPSG:3857): contextily 使用的坐标系
"""

import os
import sys

# 修复 PROJ 数据库版本冲突：pip 安装的 rasterio 自带的 PROJ DLL
# 需要匹配版本的 proj.db。必须先于 rasterio 导入前设置 PROJ_DATA。
# 查找 rasterio 自带的 proj_data 目录。
def _find_rasterio_proj_data():
    """Locate the proj_data directory bundled with rasterio."""
    for p in sys.path:
        candidate = os.path.join(p, "rasterio", "proj_data")
        if os.path.isdir(candidate):
            return candidate
    return None

_proj_data = _find_rasterio_proj_data()
if _proj_data:
    os.environ.setdefault("PROJ_DATA", _proj_data)

import numpy as np
import pyproj

# ============================================================
# Beijing 1954 GK CM 111E 网格参数 (来自 wgs84tobj1954.py)
# ============================================================
X_MIN = 988144.781509014
X_MAX = 1552144.781509014
Y_MIN = 3417504.294282121
Y_MAX = 3946504.294282121
GRID_SIZE = 1000  # 每格 1000 米

GRID_COLS = int((X_MAX - X_MIN) // GRID_SIZE)  # 564
GRID_ROWS = int((Y_MAX - Y_MIN) // GRID_SIZE)  # 529

# ============================================================
# WKT 定义 (来自 wgs84tobj1954.py)
# ============================================================
WGS84_WKT = (
    'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],'
    'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433],AUTHORITY["EPSG",4326]]'
)

BEIJING1954_WKT = (
    'PROJCS["Beijing_1954_3_Degree_GK_CM_111E",'
    'GEOGCS["GCS_Beijing_1954",DATUM["D_Beijing_1954",SPHEROID["Krasovsky_1940",6378245.0,298.3]],'
    'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],'
    'PROJECTION["Gauss_Kruger"],PARAMETER["False_Easting",500000.0],'
    'PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",111.0],'
    'PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",0.0],'
    'UNIT["Meter",1.0],AUTHORITY["EPSG",2434]]'
)

# ============================================================
# pyproj 转换器 (缓存，避免重复创建)
# ============================================================
_wgs84_crs = pyproj.CRS.from_wkt(WGS84_WKT)
_beijing1954_crs = pyproj.CRS.from_wkt(BEIJING1954_WKT)
_web_mercator_crs = pyproj.CRS.from_epsg(3857)

_bj_to_wgs84 = pyproj.Transformer.from_crs(_beijing1954_crs, _wgs84_crs, always_xy=True)
_wgs84_to_bj = pyproj.Transformer.from_crs(_wgs84_crs, _beijing1954_crs, always_xy=True)
_wgs84_to_merc = pyproj.Transformer.from_crs(_wgs84_crs, _web_mercator_crs, always_xy=True)
_merc_to_wgs84 = pyproj.Transformer.from_crs(_web_mercator_crs, _wgs84_crs, always_xy=True)


def grid_to_beijing(grid_x, grid_y):
    """网格坐标 → Beijing 1954 米制坐标 (cell 中心)"""
    bx = X_MIN + (np.asarray(grid_x) + 0.5) * GRID_SIZE
    by = Y_MIN + (np.asarray(grid_y) + 0.5) * GRID_SIZE
    return bx, by


def beijing_to_wgs84(bx, by):
    """Beijing 1954 米制 → WGS84 (lon, lat)"""
    lon, lat = _bj_to_wgs84.transform(np.asarray(bx), np.asarray(by))
    return lon, lat


def grid_to_wgs84(grid_x, grid_y):
    """网格坐标 → WGS84 (lon, lat)"""
    bx, by = grid_to_beijing(grid_x, grid_y)
    return beijing_to_wgs84(bx, by)


def grid_to_mercator(grid_x, grid_y):
    """网格坐标 → Web Mercator (EPSG:3857)"""
    lon, lat = grid_to_wgs84(grid_x, grid_y)
    mx, my = _wgs84_to_merc.transform(np.asarray(lon), np.asarray(lat))
    return mx, my


def mercator_to_grid(mx, my):
    """Web Mercator → 网格坐标"""
    lon, lat = _merc_to_wgs84.transform(np.asarray(mx), np.asarray(my))
    bx, by = _wgs84_to_bj.transform(np.asarray(lon), np.asarray(lat))
    grid_x = (np.asarray(bx) - X_MIN) / GRID_SIZE
    grid_y = (np.asarray(by) - Y_MIN) / GRID_SIZE
    return grid_x, grid_y


def grid_bounds_to_mercator(x_min, x_max, y_min, y_max):
    """网格范围 → Web Mercator 范围"""
    corners_x = np.array([x_min, x_max, x_min, x_max])
    corners_y = np.array([y_min, y_min, y_max, y_max])
    mx, my = grid_to_mercator(corners_x, corners_y)
    return mx.min(), mx.max(), my.min(), my.max()


def full_grid_bounds_mercator():
    """全图网格范围 → Web Mercator 范围"""
    return grid_bounds_to_mercator(0, GRID_COLS, 0, GRID_ROWS)


# ============================================================
# 200m 平顶六边形网格坐标转换 (hex_grid.pkl)
# ============================================================
import pickle as _pickle

_HEX_ORIGIN = None          # (origin_x, origin_y) in EPSG:2434
_HEX_SIDE = 200.0
_SQRT3 = np.sqrt(3)


def _init_hex_origin(pkl_path=None):
    """懒加载六边形网格原点 (0,0,0) 对应的 EPSG:2434 投影坐标"""
    global _HEX_ORIGIN
    if _HEX_ORIGIN is not None:
        return _HEX_ORIGIN
    if pkl_path is None:
        pkl_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "data", "hex_grid.pkl")
    with open(pkl_path, "rb") as f:
        hex_grid = _pickle.load(f)
    origin_entry = hex_grid[(0, 0, 0)]
    origin_x, origin_y = _wgs84_to_bj.transform(origin_entry["lon"], origin_entry["lat"])
    _HEX_ORIGIN = (origin_x, origin_y)
    return _HEX_ORIGIN


def wgs84_to_hex(lon, lat):
    """WGS84 经纬度 → 平顶六边形立方体坐标 (x, y, z)"""
    origin_x, origin_y = _init_hex_origin()
    px, py = _wgs84_to_bj.transform(np.asarray(lon), np.asarray(lat))
    cx = px - origin_x
    cy = py - origin_y

    q = (cx * 2.0 / 3.0) / _HEX_SIDE
    r = (-cx / 3.0 + _SQRT3 / 3.0 * cy) / _HEX_SIDE

    s = -q - r
    qi = np.round(q).astype(int)
    ri = np.round(r).astype(int)
    si = np.round(s).astype(int)

    qd = np.abs(qi - q)
    rd = np.abs(ri - r)
    sd = np.abs(si - s)

    # 修正舍入误差最大的分量，确保 x+y+z=0
    fix_q = (qd > rd) & (qd > sd)
    fix_r = (rd > sd) & ~fix_q
    fix_s = ~(fix_q | fix_r)

    if np.ndim(lon) == 0:
        if fix_q:
            qi = -ri - si
        elif fix_r:
            ri = -qi - si
        else:
            si = -qi - ri
        return (int(qi), int(si), int(ri))
    else:
        qi = np.where(fix_q, -ri - si, qi)
        ri = np.where(fix_r, -qi - si, ri)
        si = np.where(fix_s, -qi - ri, si)
        return (qi, si, ri)


def hex_to_wgs84(x, y, z):
    """六边形立方体坐标 (x, y, z) → WGS84 (lon, lat)"""
    origin_x, origin_y = _init_hex_origin()
    x_arr = np.asarray(x, dtype=np.float64)
    z_arr = np.asarray(z, dtype=np.float64)

    q = x_arr
    r = z_arr
    cx = q * _HEX_SIDE * 1.5
    cy = _SQRT3 * _HEX_SIDE * (r + q / 2.0)
    px = cx + origin_x
    py = cy + origin_y
    lon, lat = _bj_to_wgs84.transform(px, py)
    return lon, lat


def hex_to_mercator(x, y, z):
    """六边形立方体坐标 → Web Mercator (EPSG:3857)"""
    lon, lat = hex_to_wgs84(x, y, z)
    mx, my = _wgs84_to_merc.transform(np.asarray(lon), np.asarray(lat))
    return mx, my


def hex_neighbors(x, y, z):
    """返回平顶六边形 (x,y,z) 的 6 个邻居坐标列表"""
    offsets = [
        (1, -1, 0),   # 0: 东/右
        (1, 0, -1),   # 1: 东北
        (0, 1, -1),   # 2: 西北/上
        (-1, 1, 0),   # 3: 西/左
        (-1, 0, 1),   # 4: 西南
        (0, -1, 1),   # 5: 东南/下
    ]
    return [(x + dx, y + dy, z + dz) for dx, dy, dz in offsets]


def hex_distance(a, b):
    """六边形立方体坐标切比雪夫距离"""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))


def hex_in_map(x, y, z, hex_grid):
    """检查 (x,y,z) 是否在 hex_grid 字典范围内"""
    return (int(x), int(y), int(z)) in hex_grid


# ============================================================
# GCJ-02 (火星坐标系) 转换
# 高德/腾讯地图使用 GCJ-02，与 WGS84 有 100-700m 非线性偏移
# ============================================================
import math as _math

_GCJ_A = 6378245.0
_GCJ_EE = 0.00669342162296594323


def _gcj_delta(lon, lat):
    """计算 GCJ-02 相对于 WGS84 的偏移量 (度)"""
    x = np.asarray(lon, dtype=np.float64) - 105.0
    y = np.asarray(lat, dtype=np.float64) - 35.0

    dlat = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * np.sqrt(np.abs(x))
    dlat += (20.0 * np.sin(6.0 * x * _math.pi) + 20.0 * np.sin(2.0 * x * _math.pi)) * 2.0 / 3.0
    dlat += (20.0 * np.sin(y * _math.pi) + 40.0 * np.sin(y / 3.0 * _math.pi)) * 2.0 / 3.0
    dlat += (160.0 * np.sin(y / 12.0 * _math.pi) + 320.0 * np.sin(y * _math.pi / 30.0)) * 2.0 / 3.0

    dlon = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * np.sqrt(np.abs(x))
    dlon += (20.0 * np.sin(6.0 * x * _math.pi) + 20.0 * np.sin(2.0 * x * _math.pi)) * 2.0 / 3.0
    dlon += (20.0 * np.sin(x * _math.pi) + 40.0 * np.sin(x / 3.0 * _math.pi)) * 2.0 / 3.0
    dlon += (150.0 * np.sin(x / 12.0 * _math.pi) + 300.0 * np.sin(x / 30.0 * _math.pi)) * 2.0 / 3.0

    radlat = lat / 180.0 * _math.pi
    magic = np.sin(radlat)
    magic = 1.0 - _GCJ_EE * magic * magic
    sqrtmagic = np.sqrt(magic)
    dlat = (dlat * 180.0) / ((_GCJ_A * (1.0 - _GCJ_EE)) / (magic * sqrtmagic) * _math.pi)
    dlon = (dlon * 180.0) / (_GCJ_A / sqrtmagic * np.cos(radlat) * _math.pi)
    return dlon, dlat


def wgs84_to_gcj02(lon, lat):
    """WGS84 (EPSG:4326) → GCJ-02 火星坐标系"""
    dlon, dlat = _gcj_delta(lon, lat)
    return np.asarray(lon, dtype=np.float64) + dlon, np.asarray(lat, dtype=np.float64) + dlat


def gcj02_to_wgs84(lon, lat):
    """GCJ-02 → WGS84 (迭代逆变换)"""
    lon_arr = np.asarray(lon, dtype=np.float64)
    lat_arr = np.asarray(lat, dtype=np.float64)
    wgs_lon, wgs_lat = lon_arr.copy(), lat_arr.copy()
    for _ in range(10):
        gcj_lon, gcj_lat = wgs84_to_gcj02(wgs_lon, wgs_lat)
        wgs_lon -= gcj_lon - lon_arr
        wgs_lat -= gcj_lat - lat_arr
    return wgs_lon, wgs_lat


def mercator_wgs84_to_gcj02(mx, my):
    """Web Mercator WGS84 坐标 → GCJ-02 偏移后的 Mercator 坐标"""
    lon, lat = _merc_to_wgs84.transform(np.asarray(mx, dtype=np.float64),
                                          np.asarray(my, dtype=np.float64))
    gcj_lon, gcj_lat = wgs84_to_gcj02(lon, lat)
    gx, gy = _wgs84_to_merc.transform(gcj_lon, gcj_lat)
    return gx, gy
