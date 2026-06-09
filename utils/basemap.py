"""
底图渲染：高德道路瓦片底图 (GCJ-02 坐标系)。
"""

import contextily as ctx
import xyzservices

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
