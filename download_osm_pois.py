"""Download Yangtze River Delta transport POIs from OpenStreetMap.

Usage:
    python download_osm_pois.py
    python download_osm_pois.py --output data/osm_transport_pois.geojson
"""

import argparse

from utils.osm_pois import (
    DEFAULT_OVERPASS_ENDPOINTS,
    DEFAULT_POI_PATH,
    download_osm_pois,
    normalize_existing_poi_cache,
)


def main():
    parser = argparse.ArgumentParser(
        description="Download subway/train/toll stations inside the LabelTraj hex coverage",
    )
    parser.add_argument("--output", default=DEFAULT_POI_PATH, help="output GeoJSON cache")
    parser.add_argument(
        "--endpoint", action="append", dest="endpoints",
        help="Overpass interpreter URL; repeat to provide fallbacks",
    )
    parser.add_argument(
        "--tile-degrees", type=float, default=4.5,
        help="query tile side in degrees (default: 4.5)",
    )
    parser.add_argument("--overpass-timeout", type=int, default=180)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument(
        "--normalize-existing", action="store_true",
        help="reapply current filters/dedup rules to --output without downloading",
    )
    args = parser.parse_args()

    if args.normalize_existing:
        print("Normalizing the existing OSM POI cache...")
        normalize_existing_poi_cache(args.output)
        return

    print("Downloading OSM transport POIs for the valid hex coverage...")
    download_osm_pois(
        output_path=args.output,
        endpoints=tuple(args.endpoints or DEFAULT_OVERPASS_ENDPOINTS),
        tile_degrees=args.tile_degrees,
        overpass_timeout=args.overpass_timeout,
        request_timeout=args.request_timeout,
        retries=args.retries,
    )


if __name__ == "__main__":
    main()
