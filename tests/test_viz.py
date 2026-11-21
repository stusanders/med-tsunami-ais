"""Tests for T07 plotting helpers (src/tsunami/viz.py)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from tsunami import viz
from tsunami.config import load_config


def test_colors_has_expected_keys():
    required = {
        "ships",
        "gauges",
        "buoys",
        "sources",
        "hellenic",
        "calabrian",
        "cyprus",
        "aegean",
        "ais",
        "gps",
        "calm",
        "moderate",
        "rough",
        "depth_shallow",
        "depth_mid",
        "depth_deep",
        "land",
        "ocean",
        "coast",
    }
    assert required.issubset(viz.COLORS.keys())
    for k in required:
        assert isinstance(viz.COLORS[k], str) and viz.COLORS[k].startswith("#")


def test_map_axes_defaults_to_area_of_interest():
    cfg = load_config()
    aoi = cfg["domain"]["area_of_interest"]
    fig, ax = viz.map_axes(cfg)
    try:
        assert ax.get_xlim() == (float(aoi["lon_deg"][0]), float(aoi["lon_deg"][1]))
        assert ax.get_ylim() == (float(aoi["lat_deg"][0]), float(aoi["lat_deg"][1]))
        assert "°E" in ax.get_xlabel() or "deg" in ax.get_xlabel().lower()
        assert "°N" in ax.get_ylabel() or "deg" in ax.get_ylabel().lower()
        # Natural Earth polygons + coastline lines added
        assert len(ax.collections) >= 2
        expected_aspect = 1.0 / np.cos(np.deg2rad(0.5 * sum(aoi["lat_deg"])))
        assert np.isclose(ax.get_aspect(), expected_aspect, rtol=1e-6)
    finally:
        plt.close(fig)


def test_map_axes_padded_and_existing_ax():
    cfg = load_config()
    fig0, ax0 = plt.subplots(figsize=(8, 4))
    fig, ax = viz.map_axes(cfg, ax=ax0, padded=True)
    try:
        assert fig is fig0
        assert ax is ax0
        assert ax.get_xlim() == (cfg["domain"]["lon_min_deg"], cfg["domain"]["lon_max_deg"])
        assert ax.get_ylim() == (cfg["domain"]["lat_min_deg"], cfg["domain"]["lat_max_deg"])
    finally:
        plt.close(fig)


def test_save_figure_requires_title_and_caption(tmp_path):
    cfg = load_config()
    fig, ax = viz.map_axes(cfg)
    with pytest.raises(ValueError):
        viz.save_figure(fig, tmp_path / "bad.png", "", "Valid caption.", close=False)
    with pytest.raises(ValueError):
        viz.save_figure(fig, tmp_path / "bad.png", "Valid title", "   ", close=False)

    out = viz.save_figure(
        fig,
        tmp_path / "sub" / "map_test.png",
        "Eastern Mediterranean Base Map",
        "Natural Earth 1:10m coastline over the area of interest (12 to 37 E, 30 to 42 N).",
    )
    assert out.exists()
    assert out.stat().st_size > 1000
