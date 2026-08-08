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

显示时完整保留当前段前后窗口内的参考点和连接线；hex 路网和交通地点作为辅助背景淡化显示。已经写入当前输出目录
`traj_labeled.csv` 的参考段，会按 TG/TS/DT/GG/GSD 的路网颜色回显。

程序在连续标注时复用同一个窗口；切换 OD 只重绘当前视图，不会关闭窗口或重新加载全量数据。

## 离线底图

完整下载并构建当前 hex 范围的离线路网底图：

```powershell
python download_offline_basemap.py
```

源数据保存为 `data/offline_basemap/overture_segments.parquet`，运行时使用
`data/offline_basemap/tiles/` 下的 50 km 空间分块。只要
`manifest.json` 存在，标注程序默认优先使用本地底图，不访问网络。可通过环境变量
`LABELTRAJ_BASEMAP_MODE=offline|auto|online` 显式选择模式。

当前构建覆盖现有 hex 外接范围，并已核对默认轨迹数据的每一个点均有对应本地
分块。数据来源为 Overture Maps transportation，其中包含 OpenStreetMap 贡献，
运行界面会保留来源署名。

默认显示三类地点；可用 `--no-pois` 临时关闭，或用 `--poi-data <path>` 指定另一份缓存。地图上的形状/颜色图例区分三类地点，视口地点较少时显示名称；地点密集时将鼠标悬停在标记上可查看名称。光标进入地点所在 hex 后，左下角栅格信息也会列出地点。

数据来源为 OpenStreetMap contributors，采用 ODbL 1.0 许可。建议在需要更新 OSM 数据时重新运行下载命令。
