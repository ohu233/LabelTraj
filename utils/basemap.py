"""Offline-first basemap rendering with an optional online fallback."""

from __future__ import annotations

import importlib
import os
import tempfile

import xyzservices

from utils.offline_basemap import (
    DEFAULT_OFFLINE_MAP_DIR,
    OfflineBasemap,
    offline_basemap_available,
)


USE_BASEMAP = True

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SESSION_TMP_ROOT = os.path.join(_PROJECT_ROOT, "data", ".contextily_tmp")
_SESSION_CACHE_DIR = os.path.join(_SESSION_TMP_ROOT, "session")
_OFFLINE_INSTANCE = None
_CTX = None


GAODE_PROVIDER = xyzservices.TileProvider(
    url="https://webrd04.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}",
    max_zoom=18,
    min_zoom=0,
    attribution="(C) AutoNavi",
    name="AutoNavi.Normal",
)

GAODE_SATELLITE = xyzservices.TileProvider(
    url="https://webst04.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=6&x={x}&y={y}&z={z}",
    max_zoom=18,
    min_zoom=0,
    attribution="(C) AutoNavi",
    name="AutoNavi.Satellite",
)


def _get_contextily():
    """Import contextily lazily; offline labeling never needs this dependency."""
    global _CTX
    if _CTX is not None:
        return _CTX
    os.makedirs(_SESSION_CACHE_DIR, exist_ok=True)
    original_mkdtemp = tempfile.mkdtemp
    try:
        # contextily initializes joblib during import. A regular mkdtemp ACL can
        # be inaccessible to joblib in restricted Windows sessions.
        tempfile.mkdtemp = lambda *args, **kwargs: _SESSION_CACHE_DIR
        _CTX = importlib.import_module("contextily")
    finally:
        tempfile.mkdtemp = original_mkdtemp
    return _CTX


def _requested_zoom(zoom):
    if zoom is not None:
        return zoom
    value = os.environ.get("LABELTRAJ_BASEMAP_ZOOM", "").strip()
    if not value:
        return "auto"
    try:
        return int(value)
    except ValueError:
        print(f"  [WARN] Invalid LABELTRAJ_BASEMAP_ZOOM={value!r}; using auto")
        return "auto"


def _add_offline(ax, alpha):
    global _OFFLINE_INSTANCE
    if not offline_basemap_available(DEFAULT_OFFLINE_MAP_DIR):
        return False
    if _OFFLINE_INSTANCE is None:
        _OFFLINE_INSTANCE = OfflineBasemap(DEFAULT_OFFLINE_MAP_DIR)
        print(
            "  Offline basemap: "
            f"{_OFFLINE_INSTANCE.manifest.get('tile_count', '?')} tiles, "
            f"{_OFFLINE_INSTANCE.manifest.get('source_rows', '?')} source segments"
        )
    return _OFFLINE_INSTANCE.draw(ax, alpha=alpha)


def _add_online_provider(ax, provider, alpha, zoom):
    """Fetch one current viewport without persisting provider tiles."""
    ctx = _get_contextily()
    xmin, xmax, ymin, ymax = ax.axis()
    image, extent = ctx.bounds2img(
        xmin,
        ymin,
        xmax,
        ymax,
        zoom=zoom,
        source=provider,
        ll=False,
        use_cache=False,
        n_connections=1,
    )
    ax.imshow(
        image,
        extent=extent,
        interpolation="bilinear",
        aspect=ax.get_aspect(),
        alpha=alpha,
        zorder=0,
    )
    ax.axis((xmin, xmax, ymin, ymax))
    ctx.add_attribution(ax, provider.get("attribution", ""), font_size=6)


def add_basemap(ax, alpha=1.0, zoom=None):
    """Render the local vector map first, optionally falling back to online.

    ``LABELTRAJ_BASEMAP_MODE`` accepts ``auto`` (default), ``offline``, or
    ``online``. Once the local manifest exists, ``auto`` performs no network
    request and the annotation interface is fully offline.
    """
    mode = os.environ.get("LABELTRAJ_BASEMAP_MODE", "auto").strip().lower()
    if mode not in {"auto", "offline", "online"}:
        print(f"  [WARN] Invalid LABELTRAJ_BASEMAP_MODE={mode!r}; using auto")
        mode = "auto"

    if mode in {"auto", "offline"}:
        try:
            if _add_offline(ax, alpha=alpha):
                return True
        except Exception as exc:
            print(f"  [WARN] Offline basemap failed: {exc}")
        if mode == "offline":
            print(f"  [WARN] Offline basemap is unavailable: {DEFAULT_OFFLINE_MAP_DIR}")
            return False

    requested_zoom = _requested_zoom(zoom)
    last_error = None
    for provider in (GAODE_PROVIDER, GAODE_SATELLITE):
        try:
            _add_online_provider(ax, provider, alpha=alpha, zoom=requested_zoom)
            return True
        except Exception as exc:
            last_error = exc
            print(f"  [WARN] {provider.name} basemap failed: {exc}")
    print(f"  [WARN] All basemap providers failed: {last_error}")
    return False
