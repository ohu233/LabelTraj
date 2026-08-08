"""Download and load OSM transport POIs used by the labeling interface.

The download is deliberately kept separate from the interactive labeling run:
annotation stays fast and works offline once ``data/osm_transport_pois.geojson``
has been generated.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

from utils.geo_utils import (
    _wgs84_to_merc,
    get_hex_grid,
    hex_to_wgs84,
    mercator_wgs84_to_gcj02,
    wgs84_to_hex,
)


DEFAULT_POI_PATH = os.path.join("data", "osm_transport_pois.geojson")
DEFAULT_OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)

CATEGORY_SUBWAY = "subway_station"
CATEGORY_TRAIN = "train_station"
CATEGORY_TOLL = "toll_station"
CATEGORY_LABELS = {
    CATEGORY_SUBWAY: "地铁站",
    CATEGORY_TRAIN: "火车站",
    CATEGORY_TOLL: "高速收费站",
}
CATEGORY_ORDER = {
    CATEGORY_SUBWAY: 0,
    CATEGORY_TRAIN: 1,
    CATEGORY_TOLL: 2,
}

# The point must actually fall in/very near its nearest 200 m hex.  This keeps
# points in the bounding rectangle but outside the irregular hex coverage out.
MAX_HEX_CENTER_DISTANCE_METERS = 350.0

# ``railway=station`` also covers freight and operating facilities in OSM.
# The annotation layer is intended to show places where a passenger may board
# a train, so obvious non-passenger facilities are excluded even when their
# mode tags are noisy.
NON_PASSENGER_RAIL_NAME_RE = re.compile(
    r"油库|卸油|煤矿|铜矿|(?<!大学)矿$|车辆基地|车辆段|动车所|机务段|"
    r"编组场|货运|货场|线路所|信号场|会让站|越行站|工业站$",
    re.IGNORECASE,
)
INACTIVE_RAIL_NAME_RE = re.compile(
    r"在建|待建|规划(?:中)?|废弃|停运|关闭|已拆|建设中",
    re.IGNORECASE,
)


def load_hex_bounds(cache_path=os.path.join("data", "hex_cache.npz")):
    """Return the WGS84 outer bounds of all valid cells in the hex cache."""
    with np.load(cache_path) as data:
        lon = data["lon"]
        lat = data["lat"]
        return (
            float(np.nanmin(lat)),
            float(np.nanmin(lon)),
            float(np.nanmax(lat)),
            float(np.nanmax(lon)),
        )


def iter_query_tiles(bounds, tile_degrees=4.5):
    """Split a ``(south, west, north, east)`` bbox into Overpass-sized tiles."""
    south, west, north, east = map(float, bounds)
    if tile_degrees <= 0:
        raise ValueError("tile_degrees must be positive")
    lat = south
    while lat < north:
        tile_north = min(lat + tile_degrees, north)
        lon = west
        while lon < east:
            tile_east = min(lon + tile_degrees, east)
            yield (lat, lon, tile_north, tile_east)
            lon = tile_east
        lat = tile_north


def build_overpass_query(bounds, timeout_seconds=180):
    """Build the broad OSM query; semantic classification happens locally."""
    south, west, north, east = bounds
    bbox = f"{south:.7f},{west:.7f},{north:.7f},{east:.7f}"
    return f"""[out:json][timeout:{int(timeout_seconds)}];
(
  nwr[\"railway\"~\"^(station|halt)$\"]({bbox});
  nwr[\"public_transport\"=\"station\"][\"subway\"=\"yes\"]({bbox});
  nwr[\"barrier\"=\"toll_booth\"]({bbox});
  nwr[\"highway\"=\"toll_gantry\"]({bbox});
  nwr[\"amenity\"=\"toll_booth\"]({bbox});
);
out center tags;"""


def classify_tags(tags):
    """Return one or more requested categories represented by an OSM element."""
    tags = tags or {}
    categories = []
    railway = str(tags.get("railway", "")).lower()
    station = str(tags.get("station", "")).lower()
    subway = str(tags.get("subway", "")).lower()
    train = str(tags.get("train", "")).lower()
    public_transport = str(tags.get("public_transport", "")).lower()
    construction = str(tags.get("construction", "")).lower()
    proposed = str(tags.get("proposed", "")).lower()
    landuse = str(tags.get("landuse", "")).lower()
    name = str(tags.get("name:zh") or tags.get("name") or "")
    inactive_values = {"construction", "proposed", "disused", "abandoned", "razed"}
    rail_is_active = (
        railway not in inactive_values
        and station not in inactive_values
        and construction in {"", "no"}
        and proposed in {"", "no"}
        and landuse != "construction"
        and str(tags.get("disused", "")).lower() != "yes"
        and str(tags.get("abandoned", "")).lower() != "yes"
        and not INACTIVE_RAIL_NAME_RE.search(name)
    )

    is_rail_stop = railway in {"station", "halt"}
    is_subway = rail_is_active and ((
        is_rail_stop and (station == "subway" or subway == "yes")
    ) or (public_transport == "station" and subway == "yes"))
    if is_subway:
        categories.append(CATEGORY_SUBWAY)

    non_train_station_types = {"light_rail", "monorail", "tram"}
    explicit_train = train == "yes"
    passenger_station = public_transport == "station" and train != "no"
    obvious_non_passenger = bool(NON_PASSENGER_RAIL_NAME_RE.search(name))
    is_train = (
        rail_is_active
        and is_rail_stop
        and station not in non_train_station_types
        and not obvious_non_passenger
        and (
            explicit_train
            # A metro-only station is not also a train station.  An interchange
            # remains dual-classified when OSM explicitly says train=yes.
            or (passenger_station and station != "subway" and subway != "yes")
        )
    )
    # A real train/metro interchange may intentionally appear in both layers.
    if is_train:
        categories.append(CATEGORY_TRAIN)

    if (
        tags.get("barrier") == "toll_booth"
        or tags.get("highway") == "toll_gantry"
        or tags.get("amenity") == "toll_booth"
    ):
        categories.append(CATEGORY_TOLL)
    return categories


def element_lon_lat(element):
    """Extract point coordinates from an Overpass node or way/relation center."""
    if "lon" in element and "lat" in element:
        return float(element["lon"]), float(element["lat"])
    center = element.get("center") or {}
    if "lon" in center and "lat" in center:
        return float(center["lon"]), float(center["lat"])
    return None


def preferred_name(tags):
    """Choose a useful Chinese/display name without inventing a station name."""
    tags = tags or {}
    return str(
        tags.get("name:zh")
        or tags.get("name")
        or tags.get("official_name")
        or tags.get("short_name")
        or tags.get("ref")
        or "未命名"
    ).strip()


def _haversine_m(lon1, lat1, lon2, lat2):
    radius = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _normalize_name(name):
    return "".join(str(name).lower().split())


def _normalize_station_name(name, category):
    """Normalize harmless station suffix differences for spatial deduplication."""
    normalized = _normalize_name(name)
    if category in {CATEGORY_SUBWAY, CATEGORY_TRAIN}:
        normalized = re.sub(
            r"(?:铁路车站|火车站|railwaystation|trainstation|station|站)$",
            "",
            normalized,
        )
    return normalized


def _collapse_nearby_named_records(records):
    """Collapse duplicate node/area representations of the same named place."""
    unnamed = _normalize_name(preferred_name({}))
    groups = defaultdict(list)
    passthrough = []
    for record in records:
        normalized = _normalize_station_name(record["name"], record["category"])
        if normalized == unnamed:
            passthrough.append(record)
        else:
            groups[(record["category"], normalized)].append(record)

    element_rank = {"relation": 3, "way": 2, "node": 1}
    result = list(passthrough)
    for (category, _), group in groups.items():
        distance_limit = 1000.0 if category == CATEGORY_TOLL else 500.0
        clusters = []
        for record in group:
            for cluster in clusters:
                anchor = cluster[0]
                if _haversine_m(
                    record["lon"], record["lat"], anchor["lon"], anchor["lat"],
                ) <= distance_limit:
                    cluster.append(record)
                    break
            else:
                clusters.append([record])

        for cluster in clusters:
            representative = max(
                cluster,
                key=lambda r: (element_rank.get(r["osm_type"], 0), len(r.get("tags", {}))),
            )
            representative = dict(representative)
            representative["osm_member_count"] = sum(
                int(record.get("osm_member_count", 1)) for record in cluster
            )
            result.append(representative)
    return result


def elements_to_records(elements, hex_grid=None, max_center_distance=MAX_HEX_CENTER_DISTANCE_METERS):
    """Classify, deduplicate and retain only POIs covered by a valid hex cell."""
    if hex_grid is None:
        hex_grid = get_hex_grid()

    # First remove duplicates caused by overlapping query tiles.
    unique_elements = {}
    for element in elements:
        key = (str(element.get("type", "")), int(element.get("id", 0)))
        if key[0] and key[1]:
            unique_elements[key] = element

    records = []
    for element in unique_elements.values():
        coordinates = element_lon_lat(element)
        if coordinates is None:
            continue
        lon, lat = coordinates
        tags = dict(element.get("tags") or {})
        categories = classify_tags(tags)
        if not categories:
            continue

        x, y, z = map(int, wgs84_to_hex(lon, lat))
        hex_key = (x, y, z)
        if hex_key not in hex_grid:
            continue
        center_lon, center_lat = hex_to_wgs84(x, y, z)
        if _haversine_m(lon, lat, center_lon, center_lat) > max_center_distance:
            continue

        name = preferred_name(tags)
        for category in categories:
            records.append({
                "category": category,
                "name": name,
                "lon": lon,
                "lat": lat,
                "hex_x": x,
                "hex_y": y,
                "hex_z": z,
                "osm_type": str(element.get("type")),
                "osm_id": int(element.get("id")),
                "tags": tags,
            })

    # An OSM station can occasionally exist both as a node and an area/relation.
    # Collapse same-name, same-category objects landing in the same 200 m hex.
    element_rank = {"relation": 3, "way": 2, "node": 1}
    deduped = {}
    for record in records:
        key = (
            record["category"], record["hex_x"], record["hex_y"], record["hex_z"],
            _normalize_station_name(record["name"], record["category"]),
        )
        previous = deduped.get(key)
        if previous is None or element_rank.get(record["osm_type"], 0) > element_rank.get(previous["osm_type"], 0):
            deduped[key] = record

    spatially_deduped = _collapse_nearby_named_records(list(deduped.values()))
    return sorted(
        spatially_deduped,
        key=lambda r: (CATEGORY_ORDER[r["category"]], r["name"], r["osm_type"], r["osm_id"]),
    )


def _request_tile(session, query, endpoints, request_timeout, retries):
    last_error = None
    for attempt in range(retries):
        endpoint = endpoints[attempt % len(endpoints)]
        try:
            response = session.post(
                endpoint,
                data={"data": query},
                timeout=request_timeout,
            )
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = response.headers.get("Retry-After")
                delay = min(float(retry_after), 60.0) if retry_after else min(2 ** attempt, 30)
                raise requests.HTTPError(
                    f"Overpass returned HTTP {response.status_code}; retry in {delay:g}s",
                    response=response,
                )
            response.raise_for_status()
            payload = response.json()
            return payload.get("elements", []), endpoint
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"All Overpass attempts failed: {last_error}")


def download_osm_pois(
    output_path=DEFAULT_POI_PATH,
    hex_cache_path=os.path.join("data", "hex_cache.npz"),
    endpoints=DEFAULT_OVERPASS_ENDPOINTS,
    tile_degrees=4.5,
    overpass_timeout=180,
    request_timeout=120,
    retries=6,
    progress=print,
):
    """Download requested POIs, filter to the actual hex coverage and save GeoJSON."""
    bounds = load_hex_bounds(hex_cache_path)
    tiles = list(iter_query_tiles(bounds, tile_degrees=tile_degrees))
    session = requests.Session()
    session.headers.update({
        "User-Agent": "LabelTraj-OSM-POI-Downloader/1.0 (offline research annotation cache)",
        "Accept": "application/json",
    })

    elements = []
    endpoints_used = []
    for index, tile in enumerate(tiles, start=1):
        progress(f"  OSM tile {index}/{len(tiles)}: {tuple(round(v, 4) for v in tile)}")
        query = build_overpass_query(tile, timeout_seconds=overpass_timeout)
        tile_elements, endpoint = _request_tile(
            session, query, tuple(endpoints), request_timeout=request_timeout, retries=retries,
        )
        progress(f"    received {len(tile_elements):,} elements from {endpoint}")
        elements.extend(tile_elements)
        endpoints_used.append(endpoint)

    progress(f"  Filtering {len(elements):,} elements to valid hex cells...")
    records = elements_to_records(elements)
    counts = Counter(record["category"] for record in records)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    features = _records_to_features(records)

    geojson = {
        "type": "FeatureCollection",
        "name": "LabelTraj OSM transport POIs",
        "source": "OpenStreetMap contributors via Overpass API",
        "license": "Open Database License (ODbL) 1.0",
        "generated_at_utc": generated_at,
        "hex_bounds_wgs84": {
            "south": bounds[0], "west": bounds[1], "north": bounds[2], "east": bounds[3],
        },
        "feature_counts": {key: int(counts.get(key, 0)) for key in CATEGORY_ORDER},
        "overpass_endpoints": sorted(set(endpoints_used)),
        "features": features,
    }

    output = _atomic_write_geojson(output_path, geojson)
    progress(f"  Saved {len(features):,} POIs to {output}")
    for category in CATEGORY_ORDER:
        progress(f"    {CATEGORY_LABELS[category]}: {counts.get(category, 0):,}")
    return geojson


def _records_to_features(records):
    features = []
    for record in records:
        properties = {key: value for key, value in record.items() if key not in {"lon", "lat"}}
        properties["category_label"] = CATEGORY_LABELS[record["category"]]
        properties["osm_url"] = f"https://www.openstreetmap.org/{record['osm_type']}/{record['osm_id']}"
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [record["lon"], record["lat"]]},
            "properties": properties,
        })
    return features


def _atomic_write_geojson(output_path, geojson):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(geojson, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, output)
    return output


def normalize_existing_poi_cache(path=DEFAULT_POI_PATH, progress=print):
    """Reapply current classification/dedup rules without another OSM download."""
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8") as handle:
        geojson = json.load(handle)
    if geojson.get("type") != "FeatureCollection":
        raise ValueError(f"Invalid POI GeoJSON FeatureCollection: {source_path}")

    # Reconstruct unique Overpass-like elements. Interchange records may be in
    # two categories, but should only enter classification once.
    elements = {}
    for feature in geojson.get("features", []):
        geometry = feature.get("geometry") or {}
        properties = feature.get("properties") or {}
        coordinates = geometry.get("coordinates") or []
        if geometry.get("type") != "Point" or len(coordinates) < 2:
            continue
        osm_type = properties.get("osm_type")
        osm_id = properties.get("osm_id")
        if not osm_type or not osm_id:
            continue
        elements[(str(osm_type), int(osm_id))] = {
            "type": str(osm_type),
            "id": int(osm_id),
            "lon": float(coordinates[0]),
            "lat": float(coordinates[1]),
            "tags": dict(properties.get("tags") or {}),
        }

    progress(f"  Reclassifying {len(elements):,} cached OSM elements...")
    records = elements_to_records(elements.values())
    counts = Counter(record["category"] for record in records)
    geojson["features"] = _records_to_features(records)
    geojson["feature_counts"] = {
        key: int(counts.get(key, 0)) for key in CATEGORY_ORDER
    }
    geojson["normalized_at_utc"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    output = _atomic_write_geojson(source_path, geojson)
    progress(f"  Saved {len(records):,} normalized POIs to {output}")
    for category in CATEGORY_ORDER:
        progress(f"    {CATEGORY_LABELS[category]}: {counts.get(category, 0):,}")
    return geojson


def load_osm_pois(path=DEFAULT_POI_PATH, hex_grid=None):
    """Load the offline GeoJSON cache and validate records used by the UI."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("type") != "FeatureCollection":
        raise ValueError(f"Invalid POI GeoJSON FeatureCollection: {path}")

    records = []
    for feature in data.get("features", []):
        geometry = feature.get("geometry") or {}
        properties = dict(feature.get("properties") or {})
        coordinates = geometry.get("coordinates") or []
        if geometry.get("type") != "Point" or len(coordinates) < 2:
            continue
        category = properties.get("category")
        if category not in CATEGORY_ORDER:
            continue
        try:
            record = {
                **properties,
                "lon": float(coordinates[0]),
                "lat": float(coordinates[1]),
                "hex_x": int(properties["hex_x"]),
                "hex_y": int(properties["hex_y"]),
                "hex_z": int(properties["hex_z"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
        if hex_grid is not None and (
            record["hex_x"], record["hex_y"], record["hex_z"]
        ) not in hex_grid:
            continue
        records.append(record)

    # Project once at load time. Batch labeling creates many renderers, so doing
    # this in every window would repeat the same work unnecessarily.
    if records:
        lons = np.asarray([record["lon"] for record in records], dtype=float)
        lats = np.asarray([record["lat"] for record in records], dtype=float)
        mercator_x, mercator_y = _wgs84_to_merc.transform(lons, lats)
        display_x, display_y = mercator_wgs84_to_gcj02(mercator_x, mercator_y)
        for index, record in enumerate(records):
            record["_mercator_x"] = float(mercator_x[index])
            record["_mercator_y"] = float(mercator_y[index])
            record["_display_x"] = float(display_x[index])
            record["_display_y"] = float(display_y[index])
    return records


def group_pois_by_hex(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["hex_x"], record["hex_y"], record["hex_z"])].append(record)
    return dict(grouped)
