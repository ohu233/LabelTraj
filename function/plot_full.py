import geopandas as gpd
import matplotlib.pyplot as plt

base = "江苏安徽浙江上海——三省一市地理数据/基础数据"

hex_gdf = gpd.read_file(f"{base}/xyz_all_road_flag.shp")
basemap = gpd.read_file(f"{base}/三省一市区划图.shp").to_crs(hex_gdf.crs)

road_files = {
    "xd": f"{base}/县道.shp",
    "pt": f"{base}/2024普铁.shp",
    "sd": f"{base}/省道.shp",
    "gs_sfz": f"{base}/2024高速收费站.shp",
    "gs": f"{base}/2024高速.shp",
    "gd": f"{base}/国道.shp",
    "gt": f"{base}/2024高铁.shp",
    "hcz": f"{base}/2024火车站.shp",
}

for name, road_path in road_files.items():
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)

    hex_gdf.plot(ax=ax, facecolor="whitesmoke", edgecolor="#e0e0e0", linewidth=0.1)
    basemap.plot(ax=ax, facecolor="#f5f5dc", edgecolor="#999999", linewidth=0.5, alpha=0.6)

    road_gdf = gpd.read_file(road_path).to_crs(hex_gdf.crs)
    geom_type = road_gdf.geometry.type.iloc[0]
    if "Point" in geom_type or "MultiPoint" in geom_type:
        road_gdf.plot(ax=ax, color="#d62728", markersize=3, alpha=0.9)
    else:
        road_gdf.plot(ax=ax, color="#d62728", linewidth=0.5, alpha=0.9)

    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(f"hex_{name}.png", dpi=200, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"Done: hex_{name}.png")
