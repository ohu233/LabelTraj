"""
底图渲染：高德道路瓦片底图 (GCJ-02 坐标系)。
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import contextily as ctx
import xyzservices

from utils.geo_utils import full_grid_bounds_mercator

USE_BASEMAP = True

# 高德瓦片 (GCJ-02 火星坐标系)
GAODE_PROVIDER = xyzservices.TileProvider(
    url="https://webrd04.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}",
    max_zoom=18,
    min_zoom=0,
    attribution="(C) AutoNavi",
    name="AutoNavi.Normal",
)

# 备用: 高德卫星图
GAODE_SATELLITE = xyzservices.TileProvider(
    url="https://webst04.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=6&x={x}&y={y}&z={z}",
    max_zoom=18,
    min_zoom=0,
    attribution="(C) AutoNavi",
    name="AutoNavi.Satellite",
)


def add_basemap(ax, alpha=1.0, zoom=None):
    """在给定 Axes 上叠加高德道路瓦片底图 (GCJ-02)。
    失败时尝试高德卫星图作为备用。
    """
    providers = [GAODE_PROVIDER, GAODE_SATELLITE]
    last_err = None
    for provider in providers:
        try:
            ctx.add_basemap(
                ax,
                crs="EPSG:3857",
                source=provider,
                zoom=zoom or "auto",
                alpha=alpha,
                reset_extent=False,
            )
            return
        except Exception as e:
            last_err = e
    print(f"  [WARN] All tile providers failed: {last_err}")


# ---- 兼容旧 API ----
add_osm_basemap = add_basemap
USE_OSM_BASEMAP = USE_BASEMAP


def set_ax_extent(ax, xmin, xmax, ymin, ymax):
    """设置 Axes 范围并添加底图。坐标需为 Web Mercator (EPSG:3857)。"""
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if USE_BASEMAP:
        add_basemap(ax)
    else:
        _add_jpeg_fallback(ax)
    ax.set_aspect("equal")


def set_full_extent(ax):
    """设置全图范围并添加底图。"""
    xmin, xmax, ymin, ymax = full_grid_bounds_mercator()
    set_ax_extent(ax, xmin, xmax, ymin, ymax)


def _add_jpeg_fallback(ax):
    """旧 JPEG 底图回退方案。"""
    bg_img = mpimg.imread(r"figur\jiangsu\js.jpg")
    ax.imshow(bg_img, extent=[0, 564, 0, 529], aspect="equal", alpha=1)
