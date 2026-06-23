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


# ============================================================
# 网格 ↔ WGS84 ↔ Mercator（basemap.py 通过 full_grid_bounds_mercator 依赖此链）
# ============================================================

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
# 200m 平顶六边形网格坐标转换 (hex_grid_2025.pkl)
# ------------------------------------------------------------
# 仿照 metro 版实现，并针对全国 pkl（640 万 cell, 2.2GB）做缓存优化：
#
# 启动开销主要在 pickle.load(2.2GB) ≈ 287s。优化策略：
#  首次加载后把不变量序列化为密集 numpy 数组存到 data/hex_cache.npz：
#    - lon/lat 密集 2D 数组 (shape [NX, NZ], float32, NaN 标记缺失)
#    - code 密集 2D 数组 (int32，供 road_sets 构建)
#    - 仿射矩阵 M/T
#  之后启动直接 np.load(npz)（~1s），不再碰 2.2GB pkl。
#
# 查表用密集数组下标 (x, z+Z_OFFSET)，O(1)，无需重建 dict。
# ============================================================
import pickle as _pickle

_HEX_ORIGIN = None          # 兼容旧引用（= _HEX_AFFINE_T）
_HEX_SIDE = 200.0
_SQRT3 = np.sqrt(3)
_HEX_GRID = None           # HexGridProxy（供 load_hex_mapdata 返回）
# 密集数组（懒加载）：用 (x, z + Z_OFFSET) 下标取值，NaN 表示无 cell
_HEX_LON = None            # 2D float32 [NX, NZ]
_HEX_LAT = None            # 2D float32 [NX, NZ]
_HEX_CODE = None           # 2D int32  [NX, NZ]（道路位掩码，供 road_sets）
_HEX_X_OFFSET = 0          # x - _HEX_X_OFFSET 即数组第一维下标
_HEX_Z_OFFSET = 0          # z - _HEX_Z_OFFSET 即数组第二维下标
_HEX_N_CELLS = 0           # 有效 cell 数（len）
_HEX_AFFINE_M = None        # 2x2: (x, z) -> (px, py) in EPSG:2434
_HEX_AFFINE_T = None        # (tx, ty) 平移
_HEX_AFFINE_MINV = None     # M 的逆
_HEX_PKL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "data", "hex_grid_2025.pkl")
_HEX_CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               "data", "hex_cache.npz")


class HexGridProxy:
    """轻量网格代理，模拟 dict 的 __contains__/__len__/__getitem__。

    背后是密集数组（_HEX_LON/_HEX_LAT/_HEX_CODE），不存 640 万 dict。
    用于 LabelPath 的 hex_in_map / 道路叠加层存在性判断。
    """

    def __contains__(self, key):
        try:
            x, y, z = int(key[0]), int(key[1]), int(key[2])
        except (TypeError, IndexError):
            return False
        ix = x - _HEX_X_OFFSET
        iz = z - _HEX_Z_OFFSET
        if not (0 <= ix < _HEX_LON.shape[0] and 0 <= iz < _HEX_LON.shape[1]):
            return False
        return not np.isnan(_HEX_LON[ix, iz])

    def __len__(self):
        return _HEX_N_CELLS

    def __getitem__(self, key):
        ix = int(key[0]) - _HEX_X_OFFSET
        iz = int(key[2]) - _HEX_Z_OFFSET
        if not (0 <= ix < _HEX_LON.shape[0] and 0 <= iz < _HEX_LON.shape[1]):
            raise KeyError(key)
        lon = _HEX_LON[ix, iz]
        if np.isnan(lon):
            raise KeyError(key)
        return {"lon": float(lon), "lat": float(_HEX_LAT[ix, iz]),
                "code": int(_HEX_CODE[ix, iz])}

    def items(self):
        """遍历所有有效 cell（供 hex_mapdata_to_road_sets 等使用）。

        仅在缓存缺失、需重建 road_sets 时调用；正常路径用缓存避免此遍历。
        """
        NX, NZ = _HEX_LON.shape
        for ix in range(NX):
            lon_row = _HEX_LON[ix]
            lat_row = _HEX_LAT[ix]
            code_row = _HEX_CODE[ix]
            x = ix + _HEX_X_OFFSET
            for iz in range(NZ):
                lon = lon_row[iz]
                if np.isnan(lon):
                    continue
                z = iz + _HEX_Z_OFFSET
                yield ((x, -x - z, z),
                       {"lon": float(lon), "lat": float(lat_row[iz]),
                        "code": int(code_row[iz])})


def _fit_affine_from_arrays():
    """从已加载的密集数组采样拟合仿射 (x,z)→EPSG:2434。"""
    global _HEX_AFFINE_M, _HEX_AFFINE_T, _HEX_AFFINE_MINV
    NX, NZ = _HEX_LON.shape
    # 等距采样 ~20000 个有效点
    valid_mask = ~np.isnan(_HEX_LON)
    total_valid = int(valid_mask.sum())
    step = max(1, total_valid // 20000)
    xs_idx, zs_idx = np.nonzero(valid_mask)
    xs_idx = xs_idx[::step]
    zs_idx = zs_idx[::step]
    lons = _HEX_LON[xs_idx, zs_idx]
    lats = _HEX_LAT[xs_idx, zs_idx]
    xs = xs_idx.astype(float) + _HEX_X_OFFSET
    zs = zs_idx.astype(float) + _HEX_Z_OFFSET
    px, py = _wgs84_to_bj.transform(lons, lats)
    A = np.column_stack([xs, zs, np.ones_like(xs)])
    cx, *_ = np.linalg.lstsq(A, px, rcond=None)
    cy, *_ = np.linalg.lstsq(A, py, rcond=None)
    _HEX_AFFINE_M = np.array([[cx[0], cx[1]], [cy[0], cy[1]]])
    _HEX_AFFINE_T = np.array([cx[2], cy[2]])
    _HEX_AFFINE_MINV = np.linalg.inv(_HEX_AFFINE_M)


def _build_cache_from_pkl(pkl_path):
    """从 2.2GB pkl 构建密集数组缓存并保存到 data/hex_cache.npz。"""
    global _HEX_ORIGIN, _HEX_GRID, _HEX_LON, _HEX_LAT, _HEX_CODE
    global _HEX_X_OFFSET, _HEX_Z_OFFSET, _HEX_N_CELLS
    with open(pkl_path, "rb") as f:
        hex_grid = _pickle.load(f)
    if not hex_grid:
        raise ValueError(f"hex_grid is empty in {pkl_path}")

    # 求坐标范围 → 密集数组尺寸
    xs = np.fromiter((k[0] for k in hex_grid.keys()), dtype=np.int32, count=len(hex_grid))
    zs = np.fromiter((k[2] for k in hex_grid.keys()), dtype=np.int32, count=len(hex_grid))
    x_min, x_max = int(xs.min()), int(xs.max())
    z_min, z_max = int(zs.min()), int(zs.max())
    NX = x_max - x_min + 1
    NZ = z_max - z_min + 1
    _HEX_X_OFFSET = x_min
    _HEX_Z_OFFSET = z_min
    _HEX_N_CELLS = len(hex_grid)

    lon = np.full((NX, NZ), np.nan, dtype=np.float32)
    lat = np.full((NX, NZ), np.nan, dtype=np.float32)
    code = np.zeros((NX, NZ), dtype=np.int32)
    for (x, y, z), v in hex_grid.items():
        ix = x - x_min
        iz = z - z_min
        lon[ix, iz] = v["lon"]
        lat[ix, iz] = v["lat"]
        code[ix, iz] = int(v.get("code", 0) or 0)
    _HEX_LON, _HEX_LAT, _HEX_CODE = lon, lat, code

    _fit_affine_from_arrays()
    _HEX_ORIGIN = (float(_HEX_AFFINE_T[0]), float(_HEX_AFFINE_T[1]))

    np.savez(_HEX_CACHE_PATH,
             lon=lon, lat=lat, code=code,
             x_offset=np.int32(x_min), z_offset=np.int32(z_min),
             n_cells=np.int64(_HEX_N_CELLS),
             affine_M=_HEX_AFFINE_M, affine_T=_HEX_AFFINE_T)
    _HEX_GRID = HexGridProxy()
    del hex_grid  # 释放原始 dict 内存
    return _HEX_ORIGIN


def _load_from_cache():
    """从 data/hex_cache.npz 加载密集数组（快路径，~1s）。"""
    global _HEX_ORIGIN, _HEX_GRID, _HEX_LON, _HEX_LAT, _HEX_CODE
    global _HEX_X_OFFSET, _HEX_Z_OFFSET, _HEX_N_CELLS
    global _HEX_AFFINE_M, _HEX_AFFINE_T, _HEX_AFFINE_MINV
    d = np.load(_HEX_CACHE_PATH)
    _HEX_LON = d["lon"]
    _HEX_LAT = d["lat"]
    _HEX_CODE = d["code"]
    _HEX_X_OFFSET = int(d["x_offset"])
    _HEX_Z_OFFSET = int(d["z_offset"])
    _HEX_N_CELLS = int(d["n_cells"])
    _HEX_AFFINE_M = d["affine_M"]
    _HEX_AFFINE_T = d["affine_T"]
    _HEX_AFFINE_MINV = np.linalg.inv(_HEX_AFFINE_M)
    _HEX_ORIGIN = (float(_HEX_AFFINE_T[0]), float(_HEX_AFFINE_T[1]))
    _HEX_GRID = HexGridProxy()


def _init_hex_origin(pkl_path=None):
    """初始化网格：优先读缓存(npz)，无缓存则读 pkl 并构建缓存。"""
    global _HEX_ORIGIN
    if _HEX_LON is not None:
        return _HEX_ORIGIN
    if os.path.exists(_HEX_CACHE_PATH):
        _load_from_cache()
        return _HEX_ORIGIN
    if pkl_path is None:
        pkl_path = _HEX_PKL_PATH
    return _build_cache_from_pkl(pkl_path)


def get_hex_grid(pkl_path=None):
    """触发网格加载（优先缓存）并返回 HexGridProxy（避免重复读取 2.2GB）。"""
    if _HEX_LON is None:
        _init_hex_origin(pkl_path)
    return _HEX_GRID


def rebuild_hex_cache(pkl_path=None):
    """强制从 pkl 重建 data/hex_cache.npz（数据更新后调用）。"""
    if pkl_path is None:
        pkl_path = _HEX_PKL_PATH
    # 重置全局状态，强制重建
    import sys as _sys
    mod = _sys.modules[__name__]
    for attr in ("_HEX_LON", "_HEX_LAT", "_HEX_CODE", "_HEX_GRID",
                 "_HEX_AFFINE_M", "_HEX_AFFINE_T", "_HEX_AFFINE_MINV",
                 "_HEX_ORIGIN"):
        setattr(mod, attr, None)
    return _build_cache_from_pkl(pkl_path)


def _affine_xyz_to_lonlat(x, y, z):
    """仿射回退：(x, y, z) -> (lon, lat)。y 不参与（= -x-z）。"""
    xa = np.asarray(x, dtype=float)
    za = np.asarray(z, dtype=float)
    px = _HEX_AFFINE_M[0, 0] * xa + _HEX_AFFINE_M[0, 1] * za + _HEX_AFFINE_T[0]
    py = _HEX_AFFINE_M[1, 0] * xa + _HEX_AFFINE_M[1, 1] * za + _HEX_AFFINE_T[1]
    return _bj_to_wgs84.transform(px, py)


def wgs84_to_hex(lon, lat):
    """WGS84 经纬度 → 平顶六边形立方体坐标 (x, y, z)。

    仿射反算初值 + 邻域搜索最近 cell。支持标量与数组。
    """
    if _HEX_LON is None:
        _init_hex_origin()
    la = np.asarray(lon)
    if la.ndim == 0:
        return _nearest_hex_key(float(lon), float(lat))
    la2 = np.asarray(lat)
    n = la.shape[0]
    xs, ys, zs = [0] * n, [0] * n, [0] * n
    for i in range(n):
        k = _nearest_hex_key(float(la[i]), float(la2[i]))
        xs[i], ys[i], zs[i] = k[0], k[1], k[2]
    return xs, ys, zs


def _nearest_hex_key(lon, lat):
    """标量 (lon, lat) -> 最近实际 cell key (x, y, z)。仿射初值 + 邻域搜索。

    注：全国 pkl 范围大，纯仿射初值最大偏差约 9.5 cell，故邻域搜索半径
    取 ±12 以保证覆盖。
    """
    pxb, pyb = _wgs84_to_bj.transform(np.asarray(lon, dtype=float),
                                      np.asarray(lat, dtype=float))
    d = np.array([float(pxb) - _HEX_AFFINE_T[0], float(pyb) - _HEX_AFFINE_T[1]])
    xz = _HEX_AFFINE_MINV @ d
    xi = int(round(xz[0]))
    zi = int(round(xz[1]))
    best, bestd = None, float("inf")
    for ddx in range(-12, 13):
        for ddz in range(-12, 13):
            kx, kz = xi + ddx, zi + ddz
            ix = kx - _HEX_X_OFFSET
            iz = kz - _HEX_Z_OFFSET
            if not (0 <= ix < _HEX_LON.shape[0] and 0 <= iz < _HEX_LON.shape[1]):
                continue
            clo = _HEX_LON[ix, iz]
            if np.isnan(clo):
                continue
            cla = _HEX_LAT[ix, iz]
            dd = (clo - lon) ** 2 + (cla - lat) ** 2
            if dd < bestd:
                bestd = dd
                best = (kx, -kx - kz, kz)
    if best is not None:
        return best
    return (xi, -xi - zi, zi)


def hex_to_wgs84(x, y, z):
    """六边形立方体坐标 (x, y, z) → WGS84 (lon, lat)。

    优先查密集数组（精确）；越界缺失用仿射回退。支持标量与数组。
    """
    if _HEX_LON is None:
        _init_hex_origin()
    xa = np.asarray(x)
    if xa.ndim == 0:
        ix = int(x) - _HEX_X_OFFSET
        iz = int(z) - _HEX_Z_OFFSET
        if 0 <= ix < _HEX_LON.shape[0] and 0 <= iz < _HEX_LON.shape[1]:
            clo = _HEX_LON[ix, iz]
            if not np.isnan(clo):
                return float(clo), float(_HEX_LAT[ix, iz])
        lon, lat = _affine_xyz_to_lonlat(x, y, z)
        return float(lon), float(lat)
    n = xa.shape[0]
    yi = np.asarray(y)
    zi = np.asarray(z)
    lon = np.empty(n, dtype=float)
    lat = np.empty(n, dtype=float)
    for i in range(n):
        ix = int(xa[i]) - _HEX_X_OFFSET
        iz = int(zi[i]) - _HEX_Z_OFFSET
        if 0 <= ix < _HEX_LON.shape[0] and 0 <= iz < _HEX_LON.shape[1]:
            clo = _HEX_LON[ix, iz]
            if not np.isnan(clo):
                lon[i] = clo
                lat[i] = _HEX_LAT[ix, iz]
                continue
        clo, cla = _affine_xyz_to_lonlat(xa[i], yi[i], zi[i])
        lon[i] = clo
        lat[i] = cla
    return lon, lat


def hex_to_mercator(x, y, z):
    """六边形立方体坐标 → Web Mercator (EPSG:3857)"""
    lon, lat = hex_to_wgs84(x, y, z)
    mx, my = _wgs84_to_merc.transform(np.asarray(lon), np.asarray(lat))
    return mx, my


def hex_distance(a, b):
    """六边形立方体坐标切比雪夫距离"""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))


def hex_in_map(x, y, z, hex_grid):
    """检查 (x,y,z) 是否在网格范围内。支持 HexGridProxy 与旧式 dict。"""
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
