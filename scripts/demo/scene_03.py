"""Scene 3 (Problem): The Problem & The Opportunity: Can ships fill the offshore warning gap?

Three-step thesis:
  1. The Warning Gap — Seismic sensors guess, tide gauges wait for the coast, offshore buoys are scarce.
  2. The Opportunity — ~700 ships at sea carry GPS and broadcast live, open-source AIS.
  3. Research Question — Prior studies looked at single ships after a tsunami (Inazu et al., 2020; Hossen et al., 2021;
     Foster & Thomas, 2025). Can we network the entire Mediterranean fleet for live warnings?
"""
import common as cm
import matplotlib.patches as mpatches
import numpy as np

SCENE = 3
T_C1, T_C2, T_C3 = 0.6, 5.2, 9.8


def _draw_gap_icon(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
    arts = []
    sea, = ax.plot([0.5, 7.6], [2.4, 2.4], color=cm.C["wave_pos"], lw=2.2)
    coast = ax.add_patch(mpatches.Polygon([[7.6, 0.5], [8.3, 3.1], [9.5, 3.1], [9.5, 0.5]],
                                          closed=True, fc=cm.LAND, ec=cm.COAST, lw=1.5))
    arts.extend([sea, coast])
    quake = ax.scatter([1.8], [1.3], marker="*", s=240, color=cm.C["alarm"], zorder=5)
    q_ring = ax.add_patch(mpatches.Circle((1.8, 1.3), 0.52, fc="none", ec=cm.C["alarm"], lw=1.6, ls="--"))
    q_lbl = ax.text(1.8, 0.25, "Seismic: estimates\nfrom earthquake", fontsize=cm.FS_LABEL - 1,
                    color=cm.C["alarm"], ha="center", va="center", weight="bold")
    arts.extend([quake, q_ring, q_lbl])
    gauge = ax.scatter([7.9], [2.65], marker="^", s=130, color=cm.C["gauge"], zorder=6)
    g_lbl = ax.text(8.5, 3.85, "Tide gauges:\nonly at coast", fontsize=cm.FS_LABEL - 1,
                    color=cm.C["gauge"], ha="center", va="center", weight="bold")
    arts.extend([gauge, g_lbl])
    gap_arr = ax.add_patch(mpatches.FancyArrowPatch((2.6, 2.95), (7.1, 2.95), arrowstyle="<|-|>",
                                                    mutation_scale=12, lw=1.6, color=cm.MUTED, ls="--"))
    gap_lbl = ax.text(4.85, 3.65, "Offshore blind spot\n(buoys scarce)", fontsize=cm.FS_LABEL - 1,
                      color=cm.MUTED, ha="center", va="center", weight="bold")
    arts.extend([gap_arr, gap_lbl])
    return arts


def _draw_opp_icon(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
    arts = []
    xs = np.linspace(0.5, 9.5, 160)
    ys = 2.1 + 0.28 * np.sin((xs - 1.5) * 1.1)
    wave, = ax.plot(xs, ys, color=cm.C["wave_pos"], lw=2.4)
    arts.append(wave)
    sx = np.array([1.7, 3.4, 5.1, 6.8, 8.3])
    sy = 2.1 + 0.28 * np.sin((sx - 1.5) * 1.1) + 0.22
    ships = ax.scatter(sx, sy, s=75, color=cm.C["ship"], edgecolors="white", lw=0.9, zorder=5)
    arts.append(ships)
    for x, y in zip(sx, sy):
        for r, al in ((0.34, 0.65), (0.60, 0.35)):
            c = ax.add_patch(mpatches.Arc((x, y), 2 * r, 1.4 * r, angle=0, theta1=20, theta2=160,
                                          ec=cm.C["network"], lw=1.5))
            c._target_alpha = al
            arts.append(c)
    lbl_top = ax.text(5.0, 3.9, "~700 ships at sea at any moment", fontsize=cm.FS_LABEL,
                      color=cm.C["network"], ha="center", va="center", weight="bold")
    lbl_bot = ax.text(5.0, 0.85, "Live open-source AIS + onboard GPS",
                      fontsize=cm.FS_LABEL - 1, color=cm.TEXT, ha="center", va="center")
    arts.extend([lbl_top, lbl_bot])
    return arts


def _draw_rq_icon(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
    arts = []
    rng = np.random.default_rng(11)
    t = np.linspace(0.6, 9.4, 140)
    true_w = 0.85 * np.sin((t - 2.0) * 1.9) * np.exp(-0.22 * np.maximum(0, t - 2.5))
    true_w[t < 2.0] = 0.0
    noisy = true_w + rng.normal(0, 0.16, size=len(t))
    base, = ax.plot([0.6, 9.4], [2.4, 2.4], color=cm.FAINT, lw=1.0)
    l_n, = ax.plot(t, 2.4 + noisy, color=cm.C["signal"], lw=1.5)
    l_t, = ax.plot(t, 2.4 + true_w, color=cm.TEXT, lw=1.8, alpha=0.85)
    t_top = ax.text(5.0, 3.95, "Prior work: 1 ship post-event",
                    fontsize=cm.FS_LABEL - 1, color=cm.C["signal"], ha="center", va="center", weight="bold")
    t_bot = ax.text(5.0, 0.75, "Our question: network the whole fleet live?",
                    fontsize=cm.FS_LABEL - 1, color=cm.C["network"], ha="center", va="center", weight="bold")
    arts.extend([base, l_n, l_t, t_top, t_bot])
    return arts


def build(cut, ctx):
    fig, chrome = cm.new_figure(
        SCENE,
        caption="Can we turn ~700 ships already at sea into a live tsunami sensor network — first using free open-source AIS drift, and then testing precise vertical GPS?",
    )

    cards_spec = [
        (
            "1. The Warning Gap",
            "Why current alerts struggle",
            cm.C["alarm"],
            _draw_gap_icon,
            "• Seismic sensors rely on estimation\n"
            "• Tide gauges only record at the coast\n"
            "• Offshore buoys are scarce",
            "No live confirmation across open sea.",
            T_C1,
        ),
        (
            "2. The Opportunity",
            "700 ships already at sea",
            cm.C["network"],
            _draw_opp_icon,
            "• ~700 vessels in the Med at any time\n"
            "• Broadcast live open-source AIS\n"
            "• Carry onboard GPS receivers",
            "Every vessel is a potential sensor.",
            T_C2,
        ),
        (
            "3. Research Question",
            "From 1 ship to a live fleet",
            cm.C["ship"],
            _draw_rq_icon,
            "• Prior studies analysed single ships\n"
            "  after a tsunami (Inazu et al., 2020;\n"
            "  Hossen et al., 2021; Foster, 2025)",
            "Can we network the Med fleet live?",
            T_C3,
        ),
    ]

    W, H = 0.295, 0.72
    XS = [0.025, 0.3525, 0.68]
    Y0 = 0.12

    card_groups = []
    for x, (title, sub, col, icon_fn, body, punch, t0) in zip(XS, cards_spec):
        cax = cm.card(fig, (x, Y0, W, H), edgecolor=col)
        cax.patches[0].set_linewidth(1.8)
        cax.patches[0].set_alpha(0.0)

        h1 = cax.text(0.06, 0.91, title, fontsize=cm.FS_H2 - 2, weight="bold", color=col, va="center", alpha=0.0)
        h2 = cax.text(0.06, 0.835, sub, fontsize=cm.FS_SMALL, weight="bold", color=cm.MUTED, va="center", alpha=0.0)

        iax = fig.add_axes((x + 0.015, Y0 + 0.28, W - 0.03, 0.28))
        icon_arts = icon_fn(iax)
        for a in icon_arts:
            a.set_alpha(0.0)

        b_txt = cax.text(0.06, 0.33, body, fontsize=cm.FS_SMALL - 1, color=cm.TEXT, va="top", ha="left",
                         linespacing=1.35, alpha=0.0)
        p_txt = cax.text(0.06, 0.08, punch, fontsize=cm.FS_SMALL - 1, weight="bold", color=col, va="center",
                         ha="left", alpha=0.0)

        card_groups.append((t0, cax.patches[0], [h1, h2, b_txt, p_txt], icon_arts))

    chrome["caption"].set_alpha(0.0)

    def update(t):
        chrome["caption"].set_alpha(cm.fade(t, 0.4, dur=0.7))
        for t0, patch, texts, icons in card_groups:
            a = cm.fade(t, t0, dur=0.7)
            patch.set_alpha(a)
            for tx in texts:
                tx.set_alpha(a)
            for ic in icons:
                target = getattr(ic, "_target_alpha", 1.0)
                ic.set_alpha(a * target)

    return cm.Scene(fig, update, still_t=cm.duration(SCENE, cut) - 1.0)


if __name__ == "__main__":
    cm.main(__file__)
