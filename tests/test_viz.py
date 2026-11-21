"""Tests for T07 plotting helpers (src/tsunami/viz.py) and T22 animation helpers."""
import importlib.util

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from tsunami import viz
from tsunami.config import load_config, repo_path


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
        "Natural Earth 1:10m coastline over the area of interest (12 to 37 E, 30 to 42 N) with a long enough caption to verify automatic line wrapping works cleanly.",
    )
    assert out.exists()
    assert out.stat().st_size > 1000


def test_animation_eta_rgba_masks_dry_and_small_waves():
    spec = importlib.util.spec_from_file_location("anim_script", repo_path("scripts/fig_animation.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    eta = np.array([[0.0, 0.2], [-0.2, 0.001]], dtype=np.float32)
    wet = np.array([[True, True], [False, True]])
    rgba = mod._eta_rgba(eta, wet, vmax=0.25)
    assert rgba.shape == (2, 2, 4)
    assert rgba[0, 0, 3] == 0.0
    assert rgba[1, 0, 3] == 0.0  # dry cell transparent
    assert rgba[0, 1, 3] > 0.5   # wet strong wave visible
