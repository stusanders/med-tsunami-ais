"""Closing end-card scene: project title and public GitHub repository link.

Matches scene_01.py visual style (Eastern Mediterranean bathymetry + traffic backdrop with dark veil,
common.py palette and fonts). Built from: data/processed/grid.npz, Natural Earth coastline,
data/processed/receiver_pool.npz + ships.traffic_draw(index 0, purpose 1).
"""
import common as cm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

SCENE = 18
REPO_URL_SHORT = "github.com/stusanders/med-tsunami-ais"


def build(cut, ctx):
    dur = cm.duration(SCENE, cut)
    fig, chrome = cm.new_figure(SCENE, show_title=False)
    ax = cm.map_axes(fig, (0.0, 0.0, 1.0, 1.0))
    chrome["section"].set_zorder(10)
    chrome["num"].set_text("")

    ships = None
    try:
        from tsunami import ships as ships_mod
        pool = ctx.pool()
        draw = ships_mod.traffic_draw(pool, ctx.cfg, index=0, purpose=1)
        lon, lat = pool.lon[draw.pool_idx], pool.lat[draw.pool_idx]
        ships = ax.scatter(lon, lat, s=5, color=cm.C["ship"], lw=0, alpha=0.0, zorder=3)
    except Exception as e:  # noqa: BLE001
        print("scene end: traffic draw unavailable:", e)

    # Dark atmospheric veil over the map (same aesthetic as scene_01.py)
    veil = fig.add_axes((0, 0, 1, 1), zorder=5)
    veil.axis("off")
    veil_patch = veil.add_patch(mpatches.Rectangle(
        (0, 0), 1, 1, transform=veil.transAxes, color=cm.BG, alpha=0.72))

    # Centered title
    title = fig.text(0.5, 0.66, "Ships as a Tsunami Network", ha="center", va="center",
                     fontsize=cm.FS_HERO + 6, weight="bold", color=cm.TEXT, zorder=11, alpha=0)

    # Accent divider line
    div = fig.add_artist(mlines.Line2D([0.36, 0.64], [0.575, 0.575], transform=fig.transFigure,
                                       color=cm.C["network"], lw=2.5, zorder=11, alpha=0))

    # Label above URL card
    kicker = fig.text(0.5, 0.515, "Code, GPU solver, validation suite & reproducible demo",
                      ha="center", va="center", fontsize=cm.FS_BODY + 1, color=cm.MUTED, zorder=11, alpha=0)

    # Prominent URL pill card
    url_card = cm.card(fig, (0.14, 0.34, 0.72, 0.125), facecolor=cm.PANEL, edgecolor=cm.C["network"],
                       lw=2.5, radius=0.14)
    url_card.set_zorder(11)
    url_patch = url_card.patches[0]
    url_patch.set_alpha(0)
    url_txt = url_card.text(0.5, 0.5, REPO_URL_SHORT, ha="center", va="center",
                            fontsize=cm.FS_TITLE + 2, weight="bold", color=cm.C["network"],
                            family="monospace", zorder=12, alpha=0)

    # Clean footer note
    foot = fig.text(0.5, 0.255, "Open-source public repository on GitHub",
                    ha="center", va="center", fontsize=cm.FS_SMALL, color=cm.TEXT, zorder=11, alpha=0)

    def update(t):
        veil_patch.set_alpha(0.68 + 0.32 * (1 - cm.ramp(t, 0.0, 0.8)))
        if ships is not None:
            ships.set_alpha(0.55 * cm.ramp(t, 0.2, 1.2))
        title.set_alpha(cm.fade(t, 0.15, dur=0.6))
        div.set_alpha(cm.fade(t, 0.35, dur=0.5))
        kicker.set_alpha(cm.fade(t, 0.45, dur=0.5))
        a_url = cm.fade(t, 0.55, dur=0.6)
        url_patch.set_alpha(0.92 * a_url)
        url_txt.set_alpha(a_url)
        foot.set_alpha(cm.fade(t, 0.85, dur=0.5))

    return cm.Scene(fig, update, still_t=dur - 1.0)


if __name__ == "__main__":
    cm.main(__file__)
