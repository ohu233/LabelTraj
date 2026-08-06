# LabelTraj

交互式 hex 路径标注程序。程序将轨迹、hex 路网和离线 OSM 交通地点缓存叠加在同一地图中。

## OSM 地点数据

先下载长三角 hex 有效范围内的地铁站、火车站和高速收费站：

```powershell
python download_osm_pois.py
```

下载器从 `data/hex_cache.npz` 自动取得范围，分块查询 Overpass API，并将结果再次映射到实际有效 hex；外接矩形内但 hex 范围外的地点会被排除。结果保存为 `data/osm_transport_pois.geojson`。标注程序只读取本地缓存，不会在标注过程中访问 Overpass。

若只调整了本项目的分类/去重规则，不需要重新访问 OSM，可对已有缓存重算：

```powershell
python download_osm_pois.py --normalize-existing
```

运行标注程序：

```powershell
python LabelPath.py
```

默认显示三类地点；可用 `--no-pois` 临时关闭，或用 `--poi-data <path>` 指定另一份缓存。地图上的形状/颜色图例区分三类地点，视口地点较少时显示名称；地点密集时将鼠标悬停在标记上可查看名称。光标进入地点所在 hex 后，左下角栅格信息也会列出地点。

数据来源为 OpenStreetMap contributors，采用 ODbL 1.0 许可。建议在需要更新 OSM 数据时重新运行下载命令。
