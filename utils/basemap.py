"""
统一底图渲染：OSM 道路瓦片或旧 JPEG 底图。
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import contextily as ctx
import xyzservices

from utils.geo_utils import full_grid_bounds_mercator

# 设为 False 可回退到旧 JPEG 底图
USE_OSM_BASEMAP = True

OSM_PROVIDER = xyzservices.TileProvider(
    url="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    max_zoom=19,
    min_zoom=0,
    attribution="(C) OpenStreetMap contributors",
    name="OpenStreetMap.Mapnik",
)

# 备用 tile 源（CartoDB，在国内通常比 OSM 快）
FALLBACK_PROVIDER = xyzservices.TileProvider(
    url="https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
    max_zoom=19,
    min_zoom=0,
    attribution="(C) CartoDB",
    name="CartoDB.Positron",
)


def add_osm_basemap(ax, alpha=1.0, zoom=None):
    """在给定 Axes 上叠加 OSM 道路瓦片底图。
    先尝试 OSM，失败则尝试 CartoDB 备用源。
    """
    providers = [OSM_PROVIDER, FALLBACK_PROVIDER]
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
            return  # 成功
        except Exception as e:
            last_err = e
    # 全部失败时仅打印警告，不做额外处理
    print(f"  [WARN] All tile providers failed: {last_err}")


def set_ax_extent(ax, xmin, xmax, ymin, ymax):
    """设置 Axes 范围并添加底图。坐标需为 Web Mercator (EPSG:3857)。"""
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if USE_OSM_BASEMAP:
        add_osm_basemap(ax)
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
