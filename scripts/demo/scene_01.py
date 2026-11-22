"""Scene 1 (Intro): title and subtitle over the eastern Mediterranean with one traffic draw.

Built from: data/processed/grid.npz (GEBCO bathymetry), Natural Earth coastline,
data/processed/receiver_pool.npz + ships.traffic_draw(index 0, purpose 1).
"""
import common as cm

SCENE = 1


def build(cut, ctx):
    fig, chrome = cm.new_figure(SCENE, show_title=False)
    ax = cm.map_axes(fig, (0.0, 0.0, 1.0, 1.0))
    for k in ("section", "num"):
        chrome[k].set_zorder(10)

    ships = None
    try:
        from tsunami import ships as ships_mod
        pool = ctx.pool()
        draw = ships_mod.traffic_draw(pool, ctx.cfg, index=0, purpose=1)
        lon, lat = pool.lon[draw.pool_idx], pool.lat[draw.pool_idx]
        ships = ax.scatter(lon, lat, s=5, color=cm.C["ship"], lw=0, alpha=0.0, zorder=3)
    except Exception as e:  # noqa: BLE001
        print("scene 01: traffic draw unavailable:", e)

    # dim overlay that lifts as the map fades in
    veil = fig.add_axes((0, 0, 1, 1), zorder=5)
    veil.axis("off")
    veil_patch = veil.add_patch(__import__("matplotlib").patches.Rectangle(
        (0, 0), 1, 1, transform=veil.transAxes, color=cm.BG, alpha=1.0))

    title = fig.text(0.5, 0.58, "Ships as a Tsunami Network", ha="center", va="center",
                     fontsize=cm.FS_HERO + 8, weight="bold", color=cm.TEXT, zorder=11, alpha=0)
    q = fig.text(0.5, 0.435,
                 "Turning the ships already at sea into a tsunami sensor network,\n"
                 "and testing it 1.4 million times on one GPU",
                 ha="center", va="center", fontsize=cm.FS_H2,
                 color=cm.TEXT, zorder=11, alpha=0, linespacing=1.4)

    def update(t):
        veil_patch.set_alpha(0.55 + 0.45 * (1 - cm.ramp(t, 0.0, 3.0)))
        if ships is not None:
            ships.set_alpha(0.75 * cm.ramp(t, 1.0, 4.0))
        title.set_alpha(cm.fade(t, 0.4, dur=1.0))
        q.set_alpha(cm.fade(t, 1.6, dur=1.0))

    return cm.Scene(fig, update, still_t=cm.duration(SCENE, cut) - 1.0)


if __name__ == "__main__":
    cm.main(__file__)
