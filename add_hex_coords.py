# -*- coding: utf-8 -*-
"""从原始轨迹 CSV 的 lon/lat 计算六边形坐标 hex_x/hex_y/hex_z/hex_code。

使用与标注系统同一套坐标转换（geo_utils.wgs84_to_hex，基于 hex_grid_2025.pkl
的仿射拟合 + 查表），保证轨迹与路网渲染坐标系一致，不会飘移。

输入 CSV 需包含 lon/lat 列；输出新增 hex_x/hex_y/hex_z/hex_code 列。

用法:
  python add_hex_coords.py
  python add_hex_coords.py --csv <in.csv> --out <out.csv>
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd

# 确保能 import 项目根的 utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import geo_utils

DEFAULT_CSV = r"data\dataset_20230917_nanjing_to_gaochun_lishui.csv"
DEFAULT_OUT = r"data\dataset_20230917_nanjing_to_gaochun_lishui_with_hex.csv"


def add_hex_coords(csv_path, out_path):
    print(f"读取 {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df):,} 行")

    if "lon" not in df.columns or "lat" not in df.columns:
        raise ValueError("CSV 缺少 lon/lat 列，无法计算 hex 坐标")

    print("加载 hex 网格（优先读 data/hex_cache.npz 缓存）...")
    grid = geo_utils.get_hex_grid()
    print(f"  网格 {len(grid):,} cell")

    print("计算 hex_x/hex_y/hex_z（2025 仿射坐标系统）...")
    lons = df["lon"].to_numpy(dtype=float)
    lats = df["lat"].to_numpy(dtype=float)
    xs, ys, zs = geo_utils.wgs84_to_hex(lons, lats)
    xs = np.asarray(xs, dtype=int)
    ys = np.asarray(ys, dtype=int)
    zs = np.asarray(zs, dtype=int)

    print("查 hex_code ...")
    codes = np.full(len(df), -1, dtype=np.int64)
    hit = 0
    for i in range(len(df)):
        key = (int(xs[i]), int(ys[i]), int(zs[i]))
        if key in grid:
            codes[i] = int(grid[key]["code"])
            hit += 1

    df["hex_x"] = xs
    df["hex_y"] = ys
    df["hex_z"] = zs
    df["hex_code"] = codes

    print(f"  命中网格: {hit:,}/{len(df):,} ({hit/len(df)*100:.1f}%)")
    print(f"  hex_x range: [{df.hex_x.min()}, {df.hex_x.max()}]")
    print(f"  hex_z range: [{df.hex_z.min()}, {df.hex_z.max()}]")

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"已保存 -> {out_path}")
    print(f"新增列: hex_x, hex_y, hex_z, hex_code")


def main():
    parser = argparse.ArgumentParser(description="从 lon/lat 计算 hex 坐标（2025 坐标系）")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV, help="输入轨迹 CSV")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT, help="输出 CSV")
    args = parser.parse_args()
    add_hex_coords(args.csv, args.out)


if __name__ == "__main__":
    main()
