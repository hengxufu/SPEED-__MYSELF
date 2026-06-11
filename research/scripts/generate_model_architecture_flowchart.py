from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#172033"
MUTED = "#647084"
LINE = "#94A1B3"
BLUE = "#286F9E"
BLUE_LIGHT = "#E9F3F8"
GREEN = "#2F7651"
GREEN_LIGHT = "#EAF5EF"
PURPLE = "#76528C"
PURPLE_LIGHT = "#F2ECF6"
RED = "#B24B4B"
RED_LIGHT = "#FAECEC"
AMBER = "#A56B22"
AMBER_LIGHT = "#FAF2E6"


def rounded(ax, x, y, w, h, face="white", edge=LINE, lw=1.2, radius=0.016, z=3):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.005,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=LINE, lw=1.7, style="-", rad=0.0, z=2, scale=13):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=scale,
        linewidth=lw,
        linestyle=style,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, size=8.2, color=INK, weight="normal", ha="center", va="center", z=8):
    ax.text(x, y, text, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=z)


def image_icon(ax, x, y, w, h):
    rounded(ax, x, y, w, h, face="#F4F7FA", edge=BLUE, lw=1.3, radius=0.01, z=4)
    ax.add_patch(Circle((x + 0.19 * w, y + 0.77 * h), 0.055 * h, facecolor="#F4C95D", edgecolor="none", zorder=5))
    body = Polygon(
        [
            (x + 0.36 * w, y + 0.29 * h),
            (x + 0.62 * w, y + 0.34 * h),
            (x + 0.70 * w, y + 0.56 * h),
            (x + 0.56 * w, y + 0.70 * h),
            (x + 0.32 * w, y + 0.60 * h),
        ],
        closed=True,
        facecolor="#8DA0B4",
        edgecolor=INK,
        linewidth=0.75,
        zorder=6,
    )
    ax.add_patch(body)
    ax.add_patch(Rectangle((x + 0.10 * w, y + 0.38 * h), 0.24 * w, 0.16 * h, facecolor="#4E88AD", edgecolor=INK, lw=0.65, zorder=5))
    ax.add_patch(Rectangle((x + 0.66 * w, y + 0.38 * h), 0.24 * w, 0.16 * h, facecolor="#4E88AD", edgecolor=INK, lw=0.65, zorder=5))
    for px, py in [(0.35, 0.29), (0.62, 0.34), (0.70, 0.56), (0.56, 0.70), (0.32, 0.60)]:
        ax.add_patch(Circle((x + px * w, y + py * h), 0.022 * h, facecolor=RED, edgecolor="white", lw=0.4, zorder=7))


def crop_icon(ax, x, y, w, h):
    rounded(ax, x, y, w, h, face=BLUE_LIGHT, edge=BLUE, lw=1.3, radius=0.01, z=4)
    cx, cy, cw, ch = x + 0.22 * w, y + 0.22 * h, 0.56 * w, 0.56 * h
    ax.add_patch(Rectangle((cx, cy), cw, ch, fill=False, edgecolor=BLUE, lw=1.5, linestyle="--", zorder=6))
    d = 0.12 * w
    for sx, sy, dx, dy in [
        (cx, cy + ch, d, 0),
        (cx, cy + ch, 0, -d),
        (cx + cw, cy + ch, -d, 0),
        (cx + cw, cy + ch, 0, -d),
        (cx, cy, d, 0),
        (cx, cy, 0, d),
        (cx + cw, cy, -d, 0),
        (cx + cw, cy, 0, d),
    ]:
        ax.plot([sx, sx + dx], [sy, sy + dy], color=INK, lw=1.4, zorder=7)


def feature_stack(ax, x, y, w, h):
    rounded(ax, x, y, w, h, face=BLUE_LIGHT, edge=BLUE, lw=1.3, radius=0.01, z=4)
    colors = ["#A9CEE1", "#78B1CF", "#4D94BB", "#286F9E"]
    for i, c in enumerate(colors):
        ww = 0.46 * w - i * 0.045 * w
        hh = 0.62 * h - i * 0.06 * h
        xx = x + 0.19 * w + i * 0.13 * w
        yy = y + 0.19 * h + i * 0.055 * h
        ax.add_patch(Rectangle((xx, yy), ww, hh, facecolor=c, edgecolor="white", lw=0.8, zorder=5 + i))
        for k in range(1, 4):
            ax.plot([xx + k * ww / 4, xx + k * ww / 4], [yy, yy + hh], color="white", lw=0.35, alpha=0.7, zorder=6 + i)
            ax.plot([xx, xx + ww], [yy + k * hh / 4, yy + k * hh / 4], color="white", lw=0.35, alpha=0.7, zorder=6 + i)


def fpn_icon(ax, x, y, w, h):
    rounded(ax, x, y, w, h, face=GREEN_LIGHT, edge=GREEN, lw=1.3, radius=0.01, z=4)
    sizes = [(0.54, 0.12), (0.42, 0.12), (0.30, 0.12)]
    ys = [0.66, 0.44, 0.22]
    for (ww, hh), yy, c in zip(sizes, ys, ["#82B89B", "#5C9B78", "#347A55"]):
        xx = x + (w - ww * w) / 2
        ax.add_patch(Rectangle((xx, y + yy * h), ww * w, hh * h, facecolor=c, edgecolor="white", lw=0.7, zorder=6))
    arrow(ax, (x + 0.67 * w, y + 0.29 * h), (x + 0.67 * w, y + 0.48 * h), color=GREEN, lw=1.2, z=7, scale=9)
    arrow(ax, (x + 0.67 * w, y + 0.51 * h), (x + 0.67 * w, y + 0.70 * h), color=GREEN, lw=1.2, z=7, scale=9)


def heatmap_icon(ax, x, y, w, h):
    rounded(ax, x, y, w, h, face=GREEN_LIGHT, edge=GREEN, lw=1.3, radius=0.01, z=4)
    n = 4
    gap = 0.025 * w
    tw = (0.70 * w - (n - 1) * gap) / n
    left, bottom = x + 0.15 * w, y + 0.23 * h
    for r in range(n):
        for c in range(n):
            xx = left + c * (tw + gap)
            yy = bottom + r * (tw + gap)
            v = np.exp(-((r - 1.6) ** 2 + (c - 2.1) ** 2) / 1.5)
            color = plt.cm.YlOrRd(0.20 + 0.72 * v)
            ax.add_patch(Rectangle((xx, yy), tw, tw, facecolor=color, edgecolor="white", lw=0.45, zorder=6))
    ax.add_patch(Circle((left + 2.5 * (tw + gap), bottom + 1.5 * (tw + gap)), 0.032 * h, facecolor="white", edgecolor=RED, lw=1.0, zorder=7))


def keypoint_icon(ax, x, y, w, h):
    rounded(ax, x, y, w, h, face=PURPLE_LIGHT, edge=PURPLE, lw=1.3, radius=0.01, z=4)
    pts = np.array([[0.25, 0.28], [0.67, 0.25], [0.76, 0.57], [0.52, 0.74], [0.26, 0.61], [0.48, 0.47]])
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5)]
    for a, b in edges:
        ax.plot(
            [x + pts[a, 0] * w, x + pts[b, 0] * w],
            [y + pts[a, 1] * h, y + pts[b, 1] * h],
            color="#A99AB4",
            lw=0.9,
            zorder=5,
        )
    for px, py in pts:
        ax.add_patch(Circle((x + px * w, y + py * h), 0.032 * h, facecolor=RED, edgecolor="white", lw=0.65, zorder=7))


def pnp_icon(ax, x, y, w, h):
    rounded(ax, x, y, w, h, face=PURPLE_LIGHT, edge=PURPLE, lw=1.3, radius=0.01, z=4)
    origin = (x + 0.36 * w, y + 0.34 * h)
    arrow(ax, origin, (origin[0] + 0.33 * w, origin[1]), color=RED, lw=1.8, z=7, scale=10)
    arrow(ax, origin, (origin[0], origin[1] + 0.36 * h), color=GREEN, lw=1.8, z=7, scale=10)
    arrow(ax, origin, (origin[0] - 0.17 * w, origin[1] - 0.19 * h), color=BLUE, lw=1.8, z=7, scale=10)
    ax.add_patch(Polygon(
        [(x + 0.22 * w, y + 0.22 * h), (x + 0.76 * w, y + 0.29 * h), (x + 0.65 * w, y + 0.72 * h), (x + 0.30 * w, y + 0.65 * h)],
        closed=True, fill=False, edgecolor=INK, lw=0.9, zorder=6,
    ))


def mini_heatmap(ax, x, y, w, h, edge=RED):
    rounded(ax, x, y, w, h, face="white", edge=edge, lw=1.1, radius=0.008, z=4)
    for r in range(3):
        for c in range(3):
            v = np.exp(-((r - 1.1) ** 2 + (c - 1.7) ** 2) / 1.1)
            ax.add_patch(Rectangle(
                (x + 0.16 * w + c * 0.22 * w, y + 0.18 * h + r * 0.22 * h),
                0.20 * w,
                0.20 * h,
                facecolor=plt.cm.YlOrRd(0.18 + 0.70 * v),
                edgecolor="white",
                lw=0.3,
                zorder=5,
            ))


def flow_node(ax, x, y, w, h, title, subtitle, icon, color, fill):
    rounded(ax, x, y, w, h, face=fill, edge=color, lw=1.4, radius=0.012, z=3)
    icon(ax, x + 0.10 * w, y + 0.31 * h, 0.80 * w, 0.58 * h)
    label(ax, x + w / 2, y + 0.19 * h, title, size=8.4, weight="bold")
    label(ax, x + w / 2, y + 0.075 * h, subtitle, size=6.8, color=MUTED)


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(18, 8.2), dpi=260)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    label(
        ax,
        0.5,
        0.965,
        "Geometry-Consistent Heatmap Spacecraft Pose Estimation Pipeline",
        size=15.5,
        weight="bold",
    )
    label(ax, 0.5, 0.925, "Inference path", size=8.2, color=BLUE, weight="bold")
    ax.plot([0.035, 0.965], [0.895, 0.895], color="#D9E0E8", lw=0.9, zorder=1)

    nodes = [
        (0.025, "RGB Image", "target crop", image_icon, BLUE, BLUE_LIGHT),
        (0.165, "Preprocess", "224 x 224", crop_icon, BLUE, BLUE_LIGHT),
        (0.305, "Swin-Tiny", "multi-scale", feature_stack, BLUE, BLUE_LIGHT),
        (0.445, "FPN + Decoder", "256 -> 64 ch", fpn_icon, GREEN, GREEN_LIGHT),
        (0.585, "Heatmap Head", "11 x 56 x 56", heatmap_icon, GREEN, GREEN_LIGHT),
        (0.725, "Softargmax", "11 keypoints", keypoint_icon, PURPLE, PURPLE_LIGHT),
        (0.865, "RANSAC-PnP", "6-DoF pose", pnp_icon, PURPLE, PURPLE_LIGHT),
    ]
    y, w, h = 0.58, 0.11, 0.27
    for x, title, subtitle, icon, color, fill in nodes:
        flow_node(ax, x, y, w, h, title, subtitle, icon, color, fill)
    for i in range(len(nodes) - 1):
        arrow(ax, (nodes[i][0] + w + 0.005, y + 0.135), (nodes[i + 1][0] - 0.005, y + 0.135), color=LINE, lw=1.8, z=2)

    # Geometry priors entering pose recovery.
    rounded(ax, 0.765, 0.505, 0.16, 0.042, face="#F7F8FA", edge=LINE, lw=0.9, radius=0.01, z=3)
    label(ax, 0.845, 0.526, "K  +  11 Tango 3-D points", size=7.2, color=INK)
    arrow(ax, (0.905, 0.547), (0.92, 0.58), color=LINE, lw=1.1, z=2, scale=9)

    # Lower training/adaptation panel.
    rounded(ax, 0.025, 0.075, 0.95, 0.355, face="#FBFCFD", edge="#CBD4DF", lw=1.0, radius=0.014, z=1)
    label(ax, 0.055, 0.397, "TRAINING & TARGET PRE-ADAPTATION", size=8.1, color=MUTED, weight="bold", ha="left")

    # Source-supervised route.
    label(ax, 0.055, 0.325, "Synthetic source", size=8.0, color=AMBER, weight="bold", ha="left")
    image_icon(ax, 0.055, 0.165, 0.095, 0.125)
    label(ax, 0.1025, 0.138, "RGB + GT pose", size=6.8, color=MUTED)
    mini_heatmap(ax, 0.205, 0.165, 0.09, 0.125, edge=AMBER)
    label(ax, 0.25, 0.138, "GT heatmaps", size=6.8, color=MUTED)
    rounded(ax, 0.35, 0.165, 0.105, 0.125, face=AMBER_LIGHT, edge=AMBER, lw=1.2, radius=0.01, z=4)
    label(ax, 0.4025, 0.238, "KL", size=13, color=AMBER, weight="bold")
    label(ax, 0.4025, 0.197, "+ 0.2 Smooth-L1", size=6.9, color=INK)
    label(ax, 0.4025, 0.175, "valid mask", size=6.5, color=MUTED)
    arrow(ax, (0.155, 0.227), (0.2, 0.227), color=AMBER, lw=1.5, z=3)
    arrow(ax, (0.3, 0.227), (0.345, 0.227), color=AMBER, lw=1.5, z=3)
    arrow(ax, (0.455, 0.227), (0.485, 0.38), color=AMBER, lw=1.45, rad=-0.18, z=3)
    label(ax, 0.453, 0.330, "source update", size=6.5, color=AMBER)

    # Target pre-adaptation route.
    label(ax, 0.525, 0.325, "Target images + bbox", size=8.0, color=RED, weight="bold", ha="left")
    image_icon(ax, 0.525, 0.165, 0.095, 0.125)
    label(ax, 0.5725, 0.138, "lightbox / sunlamp ROI", size=6.8, color=MUTED)
    rounded(ax, 0.665, 0.165, 0.09, 0.125, face=RED_LIGHT, edge=RED, lw=1.2, radius=0.01, z=4)
    label(ax, 0.71, 0.245, "Teacher", size=8.2, color=RED, weight="bold")
    keypoint_icon(ax, 0.685, 0.178, 0.05, 0.055)
    rounded(ax, 0.795, 0.165, 0.09, 0.125, face=RED_LIGHT, edge=RED, lw=1.2, radius=0.01, z=4)
    label(ax, 0.84, 0.245, "Geometry", size=7.9, color=RED, weight="bold")
    label(ax, 0.84, 0.213, ">= 6 inliers", size=6.5, color=INK)
    label(ax, 0.84, 0.190, "<= 8 px", size=6.5, color=INK)
    mini_heatmap(ax, 0.92, 0.165, 0.045, 0.125, edge=RED)
    label(ax, 0.9425, 0.138, "pseudo", size=6.5, color=MUTED)
    arrow(ax, (0.625, 0.227), (0.66, 0.227), color=RED, lw=1.5, z=3)
    arrow(ax, (0.76, 0.227), (0.79, 0.227), color=RED, lw=1.5, z=3)
    arrow(ax, (0.89, 0.227), (0.915, 0.227), color=RED, lw=1.5, z=3)
    arrow(ax, (0.94, 0.295), (0.55, 0.58), color=RED, lw=1.35, style="--", rad=0.22, z=2)
    label(ax, 0.755, 0.407, "student update: FPN + decoder + head + backbone norms", size=6.8, color=RED)
    arrow(ax, (0.72, 0.29), (0.72, 0.34), color=RED, lw=1.25, style="--", rad=0.0, z=2, scale=9)
    arrow(ax, (0.72, 0.34), (0.68, 0.29), color=RED, lw=1.25, style="--", rad=0.0, z=2, scale=9)
    label(ax, 0.72, 0.365, "refresh", size=6.4, color=RED)

    label(ax, 0.5, 0.035, "Solid arrows: inference / supervision     Dashed arrows: iterative target-domain adaptation", size=7.0, color=MUTED)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"model_architecture_flowchart.{ext}", bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
