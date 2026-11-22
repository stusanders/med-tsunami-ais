"""T27 demo animation: shared style, scene table and render framework.

Every scene script (scripts/demo/scene_XX.py) defines

    SCENE = 3                     # scene number (row in SCENES below)
    def build(cut, ctx) -> Scene  # cut is "full" or "3min"; ctx is a Ctx (cfg, grid, paths)

and ends with

    if __name__ == "__main__":
        common.main(__file__)

`build` returns a `Scene(fig, update, still_t)` where `update(t)` redraws the figure for
time `t` seconds into the scene (0 <= t < duration). The framework owns the duration
(from SCENES, per cut), the frame loop, ffmpeg encoding and the still export, so scene
code only ever thinks in seconds.

Rules (docs/DEMO.md Part 3):
  * Every number on screen comes from a file in outputs/ (or data/) or a cited entry in
    scripts/demo/cards.yaml. Never type a result into a scene.
  * If an input file is missing, draw `placeholder(...)` ("RESULT PENDING") in its place,
    so the whole video always renders.
  * Use the palette and font sizes below; never pick ad hoc colours for a meaning that
    already has one.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import FancyBboxPatch

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
DEMO_DIR = Path(__file__).resolve().parent
REPO = DEMO_DIR.parents[1]
sys.path.insert(0, str(REPO / "src"))

from tsunami import viz  # noqa: E402  (Stream C's shared style; we only read COLORS / coastline)
from tsunami.config import load_config, repo_path  # noqa: E402

OUTPUTS = REPO / "outputs"
OUT_DEMO = OUTPUTS / "demo"            # scene_XX.mp4 / .png, demo_full.mp4, demo_3min.mp4
DEMO_DATA = OUTPUTS / "demo_data"      # data generated for the demo (showcase snapshots etc.)
CARDS = DEMO_DIR / "cards.yaml"        # cited text for scenes 2, 3 (and any other fact on screen)

# --------------------------------------------------------------------------------------
# Video format
# --------------------------------------------------------------------------------------
W_PX, H_PX, DPI, FPS = 1920, 1080, 100, 30
FIGSIZE = (W_PX / DPI, H_PX / DPI)

# --------------------------------------------------------------------------------------
# Scene table: id -> (slug, section, title, full_s, cut3_s or None if dropped, label).
# Titles from docs/DEMO_SCRIPT.md (Stream A, main eb3f9d4). Durations are the script's (284 s) except
# 1, 7, 9b, 9c, 11, 14 and 15, lengthened so each narration line fits at 2.5 words/s (total 299 s; at the
# script's 2 words/s it would need 319 s). See make_narration.py. Scenes 9b and 9c
# have ids 16 and 17 (files scene_09b.py, scene_09c.py); ORDER is the running order.
# The 3-minute cut (217 s) is ours, not in the script: it drops 3, 5, 9c and 14 and trims 2, 6 and 13.
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SceneSpec:
    num: int
    slug: str
    section: str
    title: str
    full_s: float
    cut3_s: float | None
    label: str = ""

    @property
    def tag(self) -> str:          # "08", "09b": file names and the corner number
        return self.label or f"{self.num:02d}"

    @property
    def stem(self) -> str:
        return f"scene_{self.tag}"


SCENES = {s.num: s for s in [
    SceneSpec(1, "title", "Intro", "Ships as a Tsunami Network", 9, 9),
    SceneSpec(2, "warning_gap", "Problem", "Tsunami warnings today are mostly inferred from seismic sensors", 22, 15),
    SceneSpec(3, "previous_work", "Problem & Opportunity", "The Problem & The Opportunity: Can ships fill the offshore warning gap?", 18, None),
    SceneSpec(4, "ships_as_sensors", "Solution", "We measure a tsunami from the drift between a ship's heading and its actual course", 20, 20),
    SceneSpec(5, "data", "Data", "Real inputs, simulated events", 15, None),
    SceneSpec(6, "method", "Method", "We simulate each quake, add realistic ship noise, and test if the fleet detects it", 25, 15),
    SceneSpec(7, "reshaping", "Parallelism", "GPU reshaping: 3 mechanisms turn a 3-year experiment into 1 hour on a GPU", 16, 16),
    SceneSpec(8, "tsunami_crosses", "Simulation", "Hellenic Arc, Mw 8.6: only ships on shallow shelves feel the wave above their noise", 20, 20),
    SceneSpec(9, "network_confirms", "Simulation", "The network confirms the tsunami, but after the tide gauges",
              20, 20),
    SceneSpec(16, "precise_gps", "Results", "Precise GPS confirms tsunamis in minutes, beating coastal tide gauges", 13, 13, "09b"),
    SceneSpec(17, "not_always", "Simulation", "With no ships on the shallow Cretan shelf, a Mw 8.1 tsunami goes unconfirmed", 13, None, "09c"),
    SceneSpec(10, "is_it_right", "Results", "How we verified the results: 7 physics checks and planted tsunami tests", 20, 20),
    SceneSpec(11, "one_vs_network", "Results", "Networking the fleet cuts false alarms from every quiet day to about 1 in 100", 11, 11),
    SceneSpec(12, "speed", "Parallelism", "Meaningful speedup: reshaping turns a 3-year experiment into one hour on a GPU", 18, 18),
    SceneSpec(13, "where_ships_help", "Results", "Today's ship drift confirms the largest quakes; precise GPS confirms from Mw 7", 23, 20),
    SceneSpec(14, "real_earthquake", "Verification", "Samos 2020 (Mw 7.0) verification: first 2 hours match Kos gauge (6.4 vs 6.7 cm)", 16, None),
    SceneSpec(15, "conclusion", "Conclusion", "Policy: run fleet detection on live AIS today; add live precise GPS on key corridors", 20, 20),
    SceneSpec(18, "closing", "Open Source", "Ships as a Tsunami Network", 6, 6, "end"),
]}
ORDER = [1, 2, 3, 4, 5, 6, 7, 12, 10, 14, 8, 9, 11, 13, 16, 15, 18]
SCRIPT = Path.home() / "projects" / "med-tsunami-ais" / "docs" / "DEMO_SCRIPT.md"   # on-screen text source


def scene_id(x) -> int:
    """'9b', '09b', 'scene_09b', 16 or '16' -> 16."""
    x = str(x).strip().replace("scene_", "")
    for s in SCENES.values():
        if x in (s.tag, s.tag.lstrip("0"), str(s.num)):
            return s.num
    raise KeyError(f"unknown scene {x!r}")


def duration(num: int, cut: str) -> float | None:
    s = SCENES[num]
    return s.full_s if cut == "full" else s.cut3_s


def out_paths(num: int, cut: str):
    """(mp4, png) paths. The 3-minute variant only gets its own mp4 if its length differs."""
    s = SCENES[num]
    suffix = "" if (cut == "full" or s.cut3_s == s.full_s) else "_3min"
    return OUT_DEMO / f"{s.stem}{suffix}.mp4", OUT_DEMO / f"{s.stem}.png"


# --------------------------------------------------------------------------------------
# Palette: dark background, one colour per meaning (bright variants for a dark room)
# --------------------------------------------------------------------------------------
BG = "#0A0F1E"            # figure background
PANEL = "#111A2E"         # card / panel background
PANEL_EDGE = "#2A3756"
TEXT = "#EEF2F8"
MUTED = "#8C97AD"
FAINT = "#3A4660"
LAND = "#2A3142"
COAST = "#7A86A3"

C = {
    # meanings (keep these consistent across all scenes)
    "wave_pos": "#FF6B4A",     # sea surface up
    "wave_neg": "#3FA7FF",     # sea surface down
    "ship": "#7FE3FF",         # ship dots (no signal)
    "signal": "#FFE066",       # ship signal / SNR highlight
    "alarm": "#FF3B5C",        # alarms, threshold crossings, false alarms
    "gauge": "#FFA630",        # tide gauges (from viz.COLORS gauges, brightened)
    "buoy": "#E07BFF",         # buoys
    "source": "#FFC24B",       # earthquake sources
    "network": "#4CC9F0",      # network detector
    "single": "#FF8C42",       # single-ship rule
    "gps": "#52D88A",          # precise GPS operator
    "good": "#52D88A",         # ticks / "ships add value"
    "bad": "#FF5C7A",          # crosses / "needs buoys"
    "cpu": "#9AA5BD",          # CPU in speed comparisons
    "gpu": "#76E04B",          # GPU in speed comparisons
    "calm": "#52D88A", "moderate": "#FFC24B", "rough": "#FF5C7A",
    "depth_shallow": "#FFC24B", "depth_mid": "#4CC9F0", "depth_deep": "#2B5BA8",
}

# Colormaps
WAVE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "demo_wave", ["#0B3D91", "#3FA7FF", "#0A0F1E", "#FF6B4A", "#FFE3A3"], N=256)
SNR_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "demo_snr", ["#2A3756", "#7FE3FF", "#FFE066", "#FF3B5C"], N=256)
DEPTH_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "demo_depth", ["#2F7FBF", "#1B4F86", "#0F2F5C", "#0A1B3A"], N=256)

# Font sizes (readable from the back of a room)
FS_HERO = 64
FS_TITLE = 40
FS_H2 = 30
FS_BODY = 26
FS_SMALL = 20
FS_CAPTION = 19
FS_LABEL = 17


def apply_style():
    matplotlib.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "text.color": TEXT, "axes.labelcolor": TEXT, "axes.edgecolor": FAINT,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "font.size": FS_BODY, "axes.titlesize": FS_H2, "axes.labelsize": FS_SMALL,
        "xtick.labelsize": FS_LABEL, "ytick.labelsize": FS_LABEL, "legend.fontsize": FS_SMALL,
        "legend.facecolor": PANEL, "legend.edgecolor": PANEL_EDGE, "legend.labelcolor": TEXT,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "lines.linewidth": 2.5,
    })


apply_style()


# --------------------------------------------------------------------------------------
# Context shared by scenes
# --------------------------------------------------------------------------------------
@dataclass
class Ctx:
    cfg: dict
    cut: str = "full"

    @property
    def grid(self):
        return load_grid_cached()

    def pool(self):
        return load_pool_cached()


@lru_cache(maxsize=1)
def load_cfg():
    return load_config(REPO / "config.yaml")


@lru_cache(maxsize=1)
def load_grid_cached():
    from tsunami.grid import load_grid
    return load_grid(load_cfg())


@lru_cache(maxsize=1)
def load_pool_cached():
    from tsunami import ships
    return ships.load_pool(load_cfg())


# --------------------------------------------------------------------------------------
# Scene object and figure chrome
# --------------------------------------------------------------------------------------
@dataclass
class Scene:
    fig: plt.Figure
    update: Callable[[float], None]
    still_t: float | None = None     # time (s) for the still; default = 85% of duration
    meta: dict = field(default_factory=dict)


def new_figure(num: int, caption: str | None = None, show_title: bool = True):
    """Full-frame figure with section label (top-left), scene title, caption line (bottom).

    Returns (fig, chrome) where chrome is a dict of the text artists, so a scene can fade
    them or change the caption over time (chrome["caption"].set_text(...)).
    Content area (figure coords): x 0.04..0.96, y 0.09..0.86 when show_title.
    """
    spec = SCENES[num]
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(BG)
    chrome = {}
    chrome["section"] = fig.text(0.025, 0.955, spec.section.upper(), fontsize=FS_LABEL,
                                 color=MUTED, weight="bold", va="center", ha="left")
    display_num = f"{ORDER.index(num) + 1:02d}" if num in ORDER else spec.tag
    chrome["num"] = fig.text(0.975, 0.955, display_num, fontsize=FS_LABEL, color=FAINT,
                             weight="bold", va="center", ha="right")
    if show_title:
        chrome["title"] = fig.text(0.025, 0.905, spec.title, fontsize=FS_TITLE, color=TEXT,
                                   weight="bold", va="center", ha="left")
        fit_text_width(fig, chrome["title"], 0.95)
    chrome["caption"] = fig.text(0.5, 0.03, caption or "", fontsize=FS_CAPTION, color=MUTED,
                                 va="center", ha="center")
    return fig, chrome


def fit_text_width(fig, txt, max_w_frac, min_fs=20):
    """Shrink a text artist's font until it spans at most max_w_frac of the figure width."""
    r = fig.canvas.get_renderer()
    while txt.get_fontsize() > min_fs and txt.get_window_extent(r).width > max_w_frac * fig.bbox.width:
        txt.set_fontsize(txt.get_fontsize() - 1)
    return txt


def source_note(fig, text, x=0.975, y=0.03):
    """Provenance belongs in the scene's docstring, not on screen: DEMO_SCRIPT.md says no file paths or
    working labels appear in the video. Kept as an invisible artist so older scenes still run."""
    return fig.text(x, y, "", fontsize=13, color=FAINT, ha="right", va="center", visible=False)


def placeholder(ax_or_fig, what: str, needs: str = "", rect=None):
    """Draw a clearly labelled 'RESULT PENDING' box.

    ax_or_fig: an Axes (fills it) or a Figure (then pass rect=(x, y, w, h) in fig coords).
    what:  what will appear here, e.g. "Map 2: smallest confirmable Mw by source".
    needs: the missing input, e.g. "outputs/sweep/*.parquet (T17)".
    Returns the list of artists.
    """
    if isinstance(ax_or_fig, plt.Figure):
        ax = ax_or_fig.add_axes(rect or (0.1, 0.15, 0.8, 0.65))
    else:
        ax = ax_or_fig
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    arts = [ax.add_patch(FancyBboxPatch((0.02, 0.04), 0.96, 0.92, boxstyle="round,pad=0.0,rounding_size=0.03",
                                        facecolor=PANEL, edgecolor=C["alarm"], lw=2.5, ls=(0, (6, 4)),
                                        transform=ax.transAxes))]
    arts.append(ax.text(0.5, 0.62, "RESULT PENDING", ha="center", va="center", fontsize=FS_H2,
                        color=C["alarm"], weight="bold", transform=ax.transAxes))
    arts.append(ax.text(0.5, 0.46, what, ha="center", va="center", fontsize=FS_SMALL, color=TEXT,
                        transform=ax.transAxes, wrap=True))
    if needs:
        arts.append(ax.text(0.5, 0.32, f"needs {needs}", ha="center", va="center", fontsize=FS_LABEL,
                            color=MUTED, transform=ax.transAxes, family="monospace"))
    return arts


def card(fig, rect, facecolor=PANEL, edgecolor=PANEL_EDGE, lw=1.5, radius=0.02):
    """Rounded panel in figure coords; returns an Axes with 0..1 coords and no ticks."""
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle=f"round,pad=0,rounding_size={radius}",
                                facecolor=facecolor, edgecolor=edgecolor, lw=lw,
                                transform=ax.transAxes, clip_on=False))
    return ax


# --------------------------------------------------------------------------------------
# Maps
# --------------------------------------------------------------------------------------
def aoi_extent(cfg=None):
    cfg = cfg or load_cfg()
    a = cfg["domain"]["area_of_interest"]
    return (a["lon_deg"][0], a["lon_deg"][1], a["lat_deg"][0], a["lat_deg"][1])


def map_axes(fig, rect, extent=None, depth=True, coast=True, frame=False):
    """Dark map axes with GEBCO-grid bathymetry shading and Natural Earth coastline.

    extent = (lon0, lon1, lat0, lat1); default area of interest. Aspect corrected by cos(lat).
    """
    cfg = load_cfg()
    ext = extent or aoi_extent(cfg)
    ax = fig.add_axes(rect)
    ax.set_facecolor(BG)
    if depth:
        g = load_grid_cached()
        d = np.where(g.wet, g.depth, np.nan).astype(float)
        ax.imshow(np.log10(np.clip(d, 10, 5000)), origin="lower", cmap=DEPTH_CMAP,
                  extent=(g.lon[0], g.lon[-1], g.lat[0], g.lat[-1]), vmin=1, vmax=3.7,
                  interpolation="bilinear", zorder=0)
    if coast:
        viz.add_coastline(ax, cfg, extent=ext, fill_land=True, land_color=LAND,
                          coast_color=COAST, coast_lw=0.8)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_aspect(1 / np.cos(np.deg2rad(0.5 * (ext[2] + ext[3]))))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(frame)
        s.set_color(FAINT)
    return ax


# --------------------------------------------------------------------------------------
# Timing helpers (all in seconds of scene time)
# --------------------------------------------------------------------------------------
def ramp(t, t0, t1):
    """0 before t0, 1 after t1, smoothstep in between."""
    if t1 <= t0:
        return float(t >= t0)
    x = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    return float(x * x * (3 - 2 * x))


def fade(t, t_in, t_out=None, dur=0.6):
    """Alpha that fades in at t_in (over dur) and, optionally, out at t_out."""
    a = ramp(t, t_in, t_in + dur)
    if t_out is not None:
        a *= 1.0 - ramp(t, t_out, t_out + dur)
    return a


def lerp(a, b, x):
    return a + (b - a) * x


# --------------------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------------------
def load_json(path):
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_cards():
    """Merged dict of cards.yaml plus every scripts/demo/cards_d/*.yaml (one file per owner).

    Each entry should look like {text: ..., source: "Author (year), journal, doi/url", ...}.
    """
    import yaml
    out = {}
    files = ([CARDS] if CARDS.exists() else []) + sorted((DEMO_DIR / "cards_d").glob("*.yaml"))
    for p in files:
        with open(p) as f:
            out.update(yaml.safe_load(f) or {})
    return out


def exists(rel):
    p = Path(rel)
    return (p if p.is_absolute() else REPO / p).exists()


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------
# Encoder settings. "final" is what the joined cuts use; "draft" is for fast iteration
# (half resolution, 15 fps, fastest preset). Layout is identical in both because font sizes,
# line widths and marker sizes are in points and scale with the DPI.
QUALITY = {
    "final": dict(dpi_scale=1.0, fps=FPS, extra=["-preset", "veryfast", "-crf", "18"]),
    "draft": dict(dpi_scale=0.5, fps=15, extra=["-preset", "ultrafast", "-crf", "28"]),
}
CHUNK_S = 5.0  # seconds of video per parallel chunk


def video_path(num: int, cut: str, quality: str = "final", t_range=None) -> Path:
    mp4, _ = out_paths(num, cut)
    if quality == "final" and t_range is None:
        return mp4
    tag = "" if t_range is None else f"_t{t_range[0]:05.1f}-{t_range[1]:05.1f}"
    return OUT_DEMO / "draft" / f"{mp4.stem}{tag}{'' if quality == 'draft' else '_' + quality}.mp4"


def frame_range(num: int, cut: str, quality: str = "final", t_range=None):
    """Frame indices [k0, k1) to render, at the quality's frame rate."""
    fps = QUALITY[quality]["fps"]
    n = int(round(duration(num, cut) * fps))
    if t_range is None:
        return 0, n
    return max(0, int(round(t_range[0] * fps))), min(n, int(round(t_range[1] * fps)))


def plan_chunks(k0: int, k1: int, fps: int, chunk_s: float = CHUNK_S):
    step = max(1, int(round(chunk_s * fps)))
    return [(a, min(a + step, k1)) for a in range(k0, k1, step)]


def _build(num_or_mod, cut, quality):
    mod = load_scene_module(num_or_mod) if isinstance(num_or_mod, int) else num_or_mod
    scene = mod.build(cut, Ctx(cfg=load_cfg(), cut=cut))
    fig = scene.fig
    fig.set_size_inches(*FIGSIZE)
    fig.set_dpi(DPI * QUALITY[quality]["dpi_scale"])
    return scene


def render_chunk(num_or_mod, cut: str, k0: int, k1: int, out: Path, quality: str = "final") -> Path:
    """Render frames [k0, k1) of one scene to `out`. Scenes are pure functions of t (update(t)
    sets every artist from t alone), so any chunk can be rendered in its own process."""
    q = QUALITY[quality]
    scene = _build(num_or_mod, cut, quality)
    fig, dpi = scene.fig, DPI * q["dpi_scale"]
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=q["fps"], codec="libx264", bitrate=-1,
                          extra_args=q["extra"] + ["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    with writer.saving(fig, str(out), dpi=dpi):
        for k in range(k0, k1):
            scene.update(k / q["fps"])
            writer.grab_frame(facecolor=BG)
    plt.close(fig)
    return out


def concat_videos(parts, out: Path):
    """Join H.264 files that share encoder settings, without re-encoding."""
    import subprocess
    parts = [Path(x) for x in parts]
    if len(parts) == 1:
        parts[0].replace(out)
        return out
    lst = out.with_suffix(".parts.txt")
    lst.write_text("".join(f"file '{x.resolve()}'\n" for x in parts))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", "-movflags", "+faststart", str(out)], check=True)
    lst.unlink()
    for x in parts:
        x.unlink()
    return out


def part_path(out: Path, i: int) -> Path:
    return out.parent / ".parts" / f"{out.stem}.part{i:03d}.mp4"


def render_still(num_or_mod, cut: str = "full", t: float | None = None, out: Path | None = None,
                 quality: str = "final") -> Path:
    num = num_or_mod if isinstance(num_or_mod, int) else num_or_mod.SCENE
    dur = duration(num, cut)
    scene = _build(num_or_mod, cut, quality)
    st = t if t is not None else (scene.still_t if scene.still_t is not None else 0.85 * dur)
    scene.update(min(st, dur - 1.0 / FPS))
    out = out or out_paths(num, cut)[1]
    out.parent.mkdir(parents=True, exist_ok=True)
    scene.fig.savefig(out, dpi=DPI * QUALITY[quality]["dpi_scale"], facecolor=BG)
    plt.close(scene.fig)
    return out


def render_sheet(num: int, cut: str = "full", n: int = 12, cols: int = 4) -> Path:
    """Contact sheet: n evenly spaced frames of one scene in a grid, labelled with their time."""
    dur = duration(num, cut)
    scene = _build(num, cut, "draft")
    fig = scene.fig
    times = [dur * (i + 0.5) / n for i in range(n)]
    frames = []
    for t in times:
        scene.update(t)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    rows = -(-n // cols)
    h, w = frames[0].shape[:2]
    sheet = plt.figure(figsize=(cols * w / 100, rows * (h + 40) / 100), dpi=100, facecolor=BG)
    for i, (t, fr) in enumerate(zip(times, frames)):
        ax = sheet.add_axes((i % cols / cols, 1 - (i // cols + 1) / rows, 1 / cols, h / (h + 40) / rows))
        ax.imshow(fr)
        ax.set_axis_off()
        sheet.text((i % cols + 0.02) / cols, 1 - (i // cols) / rows - 0.004, f"t = {t:4.1f} s",
                   color=TEXT, fontsize=14, va="top")
    out = OUT_DEMO / "draft" / f"sheet_{SCENES[num].tag}{'' if cut == 'full' else '_' + cut}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.savefig(out, dpi=100, facecolor=BG)
    plt.close(sheet)
    return out


def default_jobs() -> int:
    import os
    return max(1, (os.cpu_count() or 2) - 1)


def render(mod, cut="full", still=True, video=True, verbose=True, jobs=1, quality="final", t_range=None):
    """Render one scene module for one cut. Returns (mp4_path or None, png_path or None).

    jobs > 1 splits the scene into CHUNK_S-second chunks rendered in parallel processes and
    joins them with a stream copy (only for scene modules loaded from scene_XX.py)."""
    num = mod.SCENE
    if duration(num, cut) is None:
        if verbose:
            print(f"scene {SCENES[num].tag}: not in the {cut} cut, skipped")
        return None, None
    t0 = time.time()
    mp4 = video_path(num, cut, quality, t_range) if video else None
    k0, k1 = frame_range(num, cut, quality, t_range)
    if video:
        fps = QUALITY[quality]["fps"]
        # ~3 chunks per process (at least 1.5 s each): M1 mixes fast and slow cores, so small chunks
        # handed out as workers free up keep every core busy until the end
        chunks = plan_chunks(k0, k1, fps, chunk_s=max(1.5, (k1 - k0) / fps / (3 * max(1, jobs))))
        from_file = (DEMO_DIR / f"{SCENES[num].stem}.py").exists() and getattr(mod, "__file__", None)
        if jobs > 1 and len(chunks) > 1 and from_file:
            from concurrent.futures import ProcessPoolExecutor
            parts = [part_path(mp4, i) for i in range(len(chunks))]
            with ProcessPoolExecutor(max_workers=min(jobs, len(chunks))) as ex:
                list(ex.map(render_chunk, [num] * len(chunks), [cut] * len(chunks),
                            [a for a, _ in chunks], [b for _, b in chunks], parts,
                            [quality] * len(chunks)))
            concat_videos(parts, mp4)
        else:
            render_chunk(mod, cut, k0, k1, mp4, quality)
    png = None
    if still and cut == "full" and quality == "final" and t_range is None:
        png = render_still(mod, cut)
    if verbose:
        print(f"scene {SCENES[num].tag} [{cut}, {quality}]: {k1 - k0} frames in {time.time() - t0:.1f}s -> "
              f"{mp4.relative_to(REPO) if mp4 else '-'}{', ' + png.name if png else ''}")
    return mp4, png


def load_scene_module(num: int):
    stem = SCENES[num].stem
    path = DEMO_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(file):
    """CLI for a scene script. Fast iteration, quickest first:

        scene_XX.py --still-only [--at T]   one frame (PNG), ~2 s
        scene_XX.py --sheet                 12 frames across the scene in one PNG, ~5 s
        scene_XX.py --draft [--from A --to B]  half-res 15 fps video in outputs/demo/draft/
        scene_XX.py [--cut full|3min|both]  final video (parallel chunks, --jobs N)
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", default="full", choices=["full", "3min", "both"])
    ap.add_argument("--still-only", action="store_true", help="only write the PNG still (fast iteration)")
    ap.add_argument("--at", type=float, default=None, help="with --still-only: scene time in seconds")
    ap.add_argument("--sheet", action="store_true", help="contact sheet of 12 frames (outputs/demo/draft/)")
    ap.add_argument("--draft", action="store_true", help="half resolution, 15 fps, into outputs/demo/draft/")
    ap.add_argument("--from", dest="t_from", type=float, default=None, help="render only from this time (s)")
    ap.add_argument("--to", dest="t_to", type=float, default=None, help="render only up to this time (s)")
    ap.add_argument("--jobs", type=int, default=default_jobs(), help="parallel chunk processes")
    a = ap.parse_args()
    num = scene_id(Path(file).stem)
    mod = load_scene_module(num)
    cuts = ["full", "3min"] if a.cut == "both" else [a.cut]
    if a.still_only:
        out = None if a.at is None else OUT_DEMO / f"{SCENES[num].stem}_t{a.at:05.1f}.png"
        print(render_still(mod, cuts[0], t=a.at, out=out))
        return
    if a.sheet:
        for c in cuts:
            if duration(num, c) is not None:
                print(render_sheet(num, c))
        return
    quality = "draft" if a.draft else "final"
    t_range = None
    if a.t_from is not None or a.t_to is not None:
        t_range = (a.t_from or 0.0, a.t_to if a.t_to is not None else duration(num, cuts[0]))
    for c in cuts:
        if c == "3min" and "full" in cuts and out_paths(num, "3min")[0] == out_paths(num, "full")[0]:
            continue  # same file as the full cut
        render(mod, c, jobs=a.jobs, quality=quality, t_range=t_range)
