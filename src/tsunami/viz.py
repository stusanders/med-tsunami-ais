"""Plotting helpers and consistent styling for all figures (T07, Stream C).

Contract (docs/interfaces.md section 15):
    viz.map_axes(cfg, ax=None) -> (fig, ax)
    viz.save_figure(fig, path, title, caption)
    viz.COLORS
"""
from functools import lru_cache
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import shapefile
from matplotlib.collections import LineCollection, PolyCollection

from .config import repo_path
from .units import deg2rad

# Consistent, colourblind-safe palette (Okabe-Ito / Tol inspired) shared across all figures.
COLORS = {
    # Receivers & sources
    "ships": "#0072B2",
    "gauges": "#D55E00",
    "buoys": "#CC79A7",
    "sources": "#E69F00",
    "truth": "#000000",
    # Source regions
    "hellenic": "#0072B2",
    "calabrian": "#D55E00",
    "cyprus": "#009E73",
    "aegean": "#CC79A7",
    # Operators
    "ais": "#0072B2",
    "gps": "#009E73",
    # Weather regimes
    "calm": "#009E73",
    "moderate": "#E69F00",
    "rough": "#D55E00",
    # Depth bands (Map 1: <50 m, 50-200 m, >200 m)
    "depth_shallow": "#E69F00",
    "depth_mid": "#56B4E9",
    "depth_deep": "#0072B2",
    # Detectors
    "network": "#0072B2",
    "single_literal": "#D55E00",
    "single_calibrated": "#E69F00",
    # Map base
    "land": "#E8E4DC",
    "ocean": "#F5F9FC",
    "coast": "#4A4A4A",
    "grid": "#CCCCCC",
    # Generic semantic
    "positive": "#009E73",
    "negative": "#D55E00",
    "neutral": "#7F7F7F",
}


def apply_style():
    """Apply clean, readable matplotlib rcParams without changing backend."""
    matplotlib.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


def domain_extent(cfg, padded=False):
    """Return (lon_min, lon_max, lat_min, lat_max) in degrees.

    Defaults to `domain.area_of_interest` (contract update 3962d9a); set
    `padded=True` for the full solver domain including sponge bands.
    """
    dom = cfg["domain"]
    if not padded and "area_of_interest" in dom:
        aoi = dom["area_of_interest"]
        return (
            float(aoi["lon_deg"][0]),
            float(aoi["lon_deg"][1]),
            float(aoi["lat_deg"][0]),
            float(aoi["lat_deg"][1]),
        )
    return (
        float(dom["lon_min_deg"]),
        float(dom["lon_max_deg"]),
        float(dom["lat_min_deg"]),
        float(dom["lat_max_deg"]),
    )


@lru_cache(maxsize=8)
def _load_coastline_rings(shp_path_str, lon_min, lon_max, lat_min, lat_max):
    """Read Natural Earth land polygon rings intersecting the padded bounding box."""
    margin = 1.0
    x0, x1 = lon_min - margin, lon_max + margin
    y0, y1 = lat_min - margin, lat_max + margin
    sf = shapefile.Reader(shp_path_str)
    rings = []
    for shp in sf.iterShapes():
        pts = np.asarray(shp.points, dtype=np.float64)
        if len(pts) == 0:
            continue
        parts = list(shp.parts) + [len(pts)]
        for a, b in zip(parts[:-1], parts[1:]):
            ring = pts[a:b]
            if (
                ring[:, 0].max() >= x0
                and ring[:, 0].min() <= x1
                and ring[:, 1].max() >= y0
                and ring[:, 1].min() <= y1
            ):
                rings.append(ring)
    return tuple(rings)


def add_coastline(
    ax,
    cfg,
    *,
    extent=None,
    padded=False,
    fill_land=True,
    land_color=None,
    coast_color=None,
    coast_lw=0.6,
    land_zorder=0.5,
    coast_zorder=2.5,
):
    """Overlay Natural Earth land polygons and coastline outlines on `ax`.

    Land fill sits at `land_zorder=0.5` so transparent-land pcolormesh layers
    (default zorder=1) and scatter overlays remain visible while coastlines sit
    crisply on top at `coast_zorder=2.5`.
    """
    lon_min, lon_max, lat_min, lat_max = extent if extent is not None else domain_extent(cfg, padded=padded)
    shp_path = repo_path(cfg["paths"]["coastline"])
    rings = _load_coastline_rings(str(shp_path), lon_min, lon_max, lat_min, lat_max)
    if not rings:
        return ax
    if fill_land:
        polys = PolyCollection(
            rings,
            facecolors=land_color or COLORS["land"],
            edgecolors="none",
            zorder=land_zorder,
        )
        ax.add_collection(polys)
    lines = LineCollection(
        rings,
        colors=coast_color or COLORS["coast"],
        linewidths=coast_lw,
        zorder=coast_zorder,
    )
    ax.add_collection(lines)
    return ax


def map_axes(
    cfg,
    ax=None,
    *,
    extent=None,
    padded=False,
    figsize=(10.5, 5.5),
    fill_land=True,
    ocean_color=None,
    land_color=None,
    coast_color=None,
    coast_lw=0.6,
    land_zorder=0.5,
    coast_zorder=2.5,
    grid=True,
):
    """Create or configure a Mediterranean map axes with coastline and lon/lat labels.

    Defaults to `domain.area_of_interest` (12 to 37 E, 30 to 42 N).
    Returns `(fig, ax)`.
    """
    apply_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    lon_min, lon_max, lat_min, lat_max = extent if extent is not None else domain_extent(cfg, padded=padded)
    ax.set_facecolor(ocean_color or COLORS["ocean"])
    add_coastline(
        ax,
        cfg,
        extent=(lon_min, lon_max, lat_min, lat_max),
        fill_land=fill_land,
        land_color=land_color,
        coast_color=coast_color,
        coast_lw=coast_lw,
        land_zorder=land_zorder,
        coast_zorder=coast_zorder,
    )

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    mid_lat = 0.5 * (lat_min + lat_max)
    ax.set_aspect(1.0 / np.cos(deg2rad(mid_lat)))
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    if grid:
        ax.grid(True, linestyle=":", linewidth=0.5, color=COLORS["grid"], alpha=0.8, zorder=0.2)
    return fig, ax


def save_figure(fig, path, title, caption, *, dpi=150, close=True):
    """Add required title and one-line caption, lay out cleanly, and save to `path`."""
    if not isinstance(title, str) or not title.strip():
        raise ValueError("Figure title must be a non-empty string.")
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("Figure caption must be a non-empty string.")

    clean_title = title.strip()
    clean_caption = " ".join(caption.strip().splitlines())

    fig.suptitle(clean_title, fontsize=12, fontweight="bold", y=0.98)
    fig.text(
        0.01,
        0.01,
        clean_caption,
        fontsize=8.5,
        style="italic",
        color="#333333",
        ha="left",
        va="bottom",
    )

    if not fig.get_constrained_layout():
        try:
            fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.94))
        except Exception:
            fig.subplots_adjust(bottom=0.12, top=0.90)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    if close:
        plt.close(fig)
    return out_path
