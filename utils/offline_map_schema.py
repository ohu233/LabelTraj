"""Shared schema for the prebuilt offline vector basemap."""

OFFLINE_MAP_VERSION = 1

CLASS_CODES = {
    "motorway": 1,
    "trunk": 2,
    "primary": 3,
    "secondary": 4,
    "tertiary": 5,
    "residential": 6,
    "unclassified": 7,
    "living_street": 8,
    "service": 9,
    "pedestrian": 10,
    "footway": 11,
    "path": 12,
    "track": 13,
    "steps": 14,
    "rail": 20,
}

DEFAULT_ROAD_CODE = CLASS_CODES["unclassified"]

# Draw minor roads first and important roads last. Colors are intentionally
# neutral so the colored hex evidence remains visually dominant.
CLASS_STYLES = {
    14: {"color": "#d8d3cb", "linewidth": 0.25, "zorder": 0.20, "linestyle": ":"},
    12: {"color": "#d8d3cb", "linewidth": 0.28, "zorder": 0.21, "linestyle": ":"},
    13: {"color": "#d2cdc5", "linewidth": 0.30, "zorder": 0.22, "linestyle": ":"},
    11: {"color": "#d4cfc7", "linewidth": 0.30, "zorder": 0.23, "linestyle": ":"},
    10: {"color": "#d4cfc7", "linewidth": 0.32, "zorder": 0.24, "linestyle": "-"},
    9: {"color": "#ddd9d2", "linewidth": 0.38, "zorder": 0.25, "linestyle": "-"},
    8: {"color": "#d7d3cc", "linewidth": 0.42, "zorder": 0.26, "linestyle": "-"},
    7: {"color": "#d3d0ca", "linewidth": 0.48, "zorder": 0.27, "linestyle": "-"},
    6: {"color": "#ffffff", "linewidth": 0.58, "zorder": 0.28, "linestyle": "-"},
    5: {"color": "#fffdf7", "linewidth": 0.75, "zorder": 0.30, "linestyle": "-"},
    4: {"color": "#f6e8b1", "linewidth": 0.95, "zorder": 0.32, "linestyle": "-"},
    3: {"color": "#f3cf78", "linewidth": 1.15, "zorder": 0.34, "linestyle": "-"},
    2: {"color": "#efb65f", "linewidth": 1.35, "zorder": 0.36, "linestyle": "-"},
    1: {"color": "#e58b45", "linewidth": 1.60, "zorder": 0.38, "linestyle": "-"},
    20: {"color": "#777777", "linewidth": 0.75, "zorder": 0.40, "linestyle": "--"},
}

