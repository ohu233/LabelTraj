import pickle

with open("hex_grid.pkl", "rb") as f:
    data = pickle.load(f)

# 判断格式：DataFrame 还是 dict
if hasattr(data, "head"):
    # 旧格式：DataFrame
    print("格式: DataFrame")
    print(f"行数: {len(data):,}  列数: {len(data.columns)}")
    print(f"列名: {data.columns.tolist()}")
    print(data.head(20).to_string())
else:
    # 新格式：dict
    print(f"格式: dict, 条目数: {len(data):,}")
    # 取前 20 个 key
    keys = sorted(data.keys())[:20]
    print("\n前 20 条 (按坐标排序):")
    print(f"{'坐标':<20} {'lon':>12} {'lat':>12} {'code':>6}")
    print("-" * 54)
    for k in keys:
        v = data[k]
        print(f"{str(k):<20} {v['lon']:>12.6f} {v['lat']:>12.6f} {v['code']:>6}")

    # code 分布
    from collections import Counter
    code_counts = Counter(v["code"] for v in data.values())
    print(f"\ncode 分布 (前 20 种):")
    for code, cnt in code_counts.most_common(20):
        bin_str = f"{code:08b}"
        print(f"  code={code:>3}  0b{bin_str}  : {cnt:>8,} 个格子 ({cnt/len(data)*100:.1f}%)")

    # 坐标范围
    xs = [k[0] for k in data]
    ys = [k[1] for k in data]
    zs = [k[2] for k in data]
    lons = [v["lon"] for v in data.values()]
    lats = [v["lat"] for v in data.values()]
    print(f"\n坐标范围:")
    print(f"  x: {min(xs)} ~ {max(xs)}")
    print(f"  y: {min(ys)} ~ {max(ys)}")
    print(f"  z: {min(zs)} ~ {max(zs)}")
    print(f"  lon: {min(lons):.4f} ~ {max(lons):.4f}")
    print(f"  lat: {min(lats):.4f} ~ {max(lats):.4f}")
