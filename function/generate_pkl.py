import pandas as pd
import numpy as np

src = "江苏安徽浙江上海——三省一市地理数据/基础数据/xyz_all_road_flag.csv"
df = pd.read_csv(src)
print(f"读取 {len(df):,} 条记录")

# 8 列按位权求和：has_xd(128) has_pt(64) has_sd(32) has_gs_sfz(16) has_gs(8) has_gd(4) has_gt(2) has_hcz(1)
bits = [128, 64, 32, 16, 8, 4, 2, 1]
has_cols = ["has_xd", "has_pt", "has_sd", "has_gs_sfz", "has_gs", "has_gd", "has_gt", "has_hcz"]

code = np.zeros(len(df), dtype=np.int16)
for col, bit in zip(has_cols, bits):
    code += df[col].values * bit

# 构建 {(x,y,z): {"lon": ..., "lat": ..., "code": ...}}
print("构建字典...")
coords = list(zip(df["x"].values, df["y"].values, df["z"].values))
result = {
    (int(x), int(y), int(z)): {
        "lon": float(lon),
        "lat": float(lat),
        "code": int(c),
    }
    for (x, y, z), lon, lat, c in zip(coords, df["lon"].values, df["lat"].values, code)
}

# 按原坐标类型转换，确保 key 和 value 都是 Python 原生类型
import gc
del df, code, coords
gc.collect()

print(f"字典条目数: {len(result):,}")
print(f"内存占用约: {len(result) * 300 / 1e9:.1f} GB")

pd.to_pickle(result, "hex_grid.pkl")
print("已保存到 hex_grid.pkl")
