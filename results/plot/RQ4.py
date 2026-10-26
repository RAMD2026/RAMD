"""
RQ5 (Comparison with baselines): How does RAMD compare to baseline approaches?
Layout:
  Left  (main) : Accuracy (%) grouped bar chart — IoT / Self-driving car /
                 Smart parking / Overall (separated)
  Right (small): Runtime (s) bar chart — geometric mean per method + per-domain
                 dots, log-scale Y-axis
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

mpl.rcParams.update({
    "font.size":        13,
    "axes.titlesize":   15,
    "axes.labelsize":   13,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  12,
})

# ---- Data ----
# Accuracy [%]
acc = {
    "IoT":             [74.4137, 83.7489, 33.1000, 68.3000, 29.2770],
    "Self-driving car":[73.0442, 82.5697, 29.4000, 14.4200, 32.0350],
    "Smart parking":   [74.6610, 81.5560, 19.9601, 40.9100, 11.1570],
}
# Runtime [seconds]
rt = {
    "IoT":             [1131.5, 1833.3,  1.0000, 18.80, 0.030],
    "Self-driving car":[ 1055.5,  458.1,  0.3000, 17.10, 0.100],
    "Smart parking":   [ 2995.0,  725.0,  0.4802, 21.70, 0.100],
}

domains   = list(acc.keys())
n_methods = 5

# Overall: arithmetic mean for accuracy, geometric mean for runtime
overall_acc = [np.mean([acc[d][i] for d in domains]) for i in range(n_methods)]
overall_rt  = [np.exp(np.mean(np.log([rt[d][i]  for d in domains]))) for i in range(n_methods)]

# ---- Labels & colours ----
legend_labels = [
    r"RAMD",
    r"RAMD+KM",
    "TF-IDF lexical matching",
    "Sentence embedding similarity",
    "Graph structural similarity",
]
xticklabels_rt = ["RAMD", "RAMD\n+KM", "TF-IDF", "Sent.\nembed.", "Graph\nsim."]

colors = [
    "#8c564b",  # RAMD       — brown
    "#1f77b4",  # RAMD+KM    — blue
    "#4292c6",  # TF-IDF     — steel blue
    "#f16913",  # Sent. embed — orange
    "#41ab5d",  # Graph sim.  — green
]

# ---- Figure ----
fig = plt.figure(figsize=(20, 7))
gs  = GridSpec(1, 2, width_ratios=[3, 1.3], wspace=0.38)

# =============================================================
# LEFT — Accuracy grouped bar chart
# =============================================================
ax_acc = fig.add_subplot(gs[0, 0])

all_groups   = domains + ["Overall"]
all_acc_vals = [acc[d] for d in domains] + [overall_acc]
n_groups     = len(all_groups)

bar_w = 0.15
# Shift "Overall" group right to create a visual gap
x_groups     = np.arange(n_groups) * 1.0
x_groups[-1] += 0.30

for i in range(n_methods):
    offset = (i - (n_methods - 1) / 2) * bar_w
    x_pos  = x_groups + offset
    vals   = [all_acc_vals[g][i] for g in range(n_groups)]

    bars = ax_acc.bar(
        x_pos, vals,
        width=bar_w,
        color=colors[i],
        alpha=0.88,
        edgecolor="#5a3530" if i == 0 else "white",
        linewidth=0.6,
        zorder=3,
    )

    # Value labels on top of each bar
    for bar, val in zip(bars, vals):
        label_y = val + 0.7
        ax_acc.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{val:.1f}",
            ha="center", va="bottom",
            fontsize=11, color="#222",
        )

# Dashed separator before "Overall"
sep_x = (x_groups[-2] + x_groups[-1]) / 2
ax_acc.axvline(sep_x, color="#bbb", linestyle="--", linewidth=1.2, zorder=1)
ax_acc.text(
    sep_x + 0.04, 101, "Overall →",
    fontsize=12, color="#666", va="top", ha="left",
)

# "Better ↑" annotation
ax_acc.annotate(
    "Better ↑",
    xy=(0.01, 0.97), xycoords="axes fraction",
    fontsize=13, color="#2a7a2a", fontweight="bold", va="top",
)

ax_acc.set_xticks(x_groups)
ax_acc.set_xticklabels(all_groups, fontsize=13)
ax_acc.set_ylabel("Accuracy (%)", fontsize=15)
ax_acc.set_ylim(0, 107)
ax_acc.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}"))
ax_acc.grid(axis="y", alpha=0.3, zorder=0)
ax_acc.set_axisbelow(True)
ax_acc.set_title("Accuracy", fontsize=15, fontweight="bold", pad=10)

# Legend (placed below the figure to avoid overlapping bars)
handles_acc = [mpatches.Patch(color=colors[i], alpha=0.88, label=legend_labels[i])
               for i in range(n_methods)]

# =============================================================
# RIGHT — Runtime bar chart (log scale, geo mean + domain dots)
# =============================================================
ax_rt = fig.add_subplot(gs[0, 1])

x_m      = np.arange(n_methods)
bar_w_rt = 0.52

# Geometric-mean runtime bars
bars_rt = ax_rt.bar(
    x_m, overall_rt,
    width=bar_w_rt,
    color=colors,
    alpha=0.88,
    edgecolor="white",
    linewidth=0.6,
    zorder=3,
)

# Per-domain scatter dots on top of bars
domain_markers = ["o", "s", "^"]
for j, d in enumerate(domains):
    for i in range(n_methods):
        ax_rt.scatter(
            x_m[i], rt[d][i],
            marker=domain_markers[j],
            color=colors[i],
            edgecolors="black",
            linewidths=0.9,
            s=48,
            zorder=5,
            alpha=0.92,
        )

# Geo-mean value labels
for bar, val in zip(bars_rt, overall_rt):
    fmt = f"{val:.1f}s" if val >= 1 else f"{val:.2f}s"
    ax_rt.text(
        bar.get_x() + bar.get_width() / 2,
        val * 2.2,
        fmt,
        ha="center", va="bottom",
        fontsize=11, color="#222",
    )

ax_rt.set_yscale("log")
ax_rt.set_xticks(x_m)
ax_rt.set_xticklabels(xticklabels_rt, fontsize=12)
ax_rt.set_ylabel("Runtime (seconds, log scale)", fontsize=15)
ax_rt.grid(axis="y", which="both", alpha=0.25, zorder=0)
ax_rt.set_axisbelow(True)
ax_rt.set_title("Runtime", fontsize=15, fontweight="bold", pad=10)

# Domain dot legend
domain_handles = [
    Line2D([0], [0], marker=domain_markers[j], color="gray",
           markersize=6, markeredgecolor="black", markeredgewidth=0.8,
           linestyle="None", label=d)
    for j, d in enumerate(domains)
]
fig.legend(
    handles=handles_acc,
    loc="lower center", ncol=3, fontsize=12,
    bbox_to_anchor=(0.38, -0.08),
    frameon=True, facecolor="white", edgecolor="#ccc",
)

fig.legend(
    handles=domain_handles,
    loc="lower center", ncol=3, fontsize=12,
    bbox_to_anchor=(0.80, -0.08),
    frameon=True, facecolor="white", edgecolor="#ccc",
    title="Domain (dots)", title_fontsize=12,
)

plt.tight_layout()
out_path = "RQ5.pdf"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
