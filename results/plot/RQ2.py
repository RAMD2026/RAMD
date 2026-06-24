"""
RQ2 (Ablation): How do the individual components of RAMD affect its effectiveness?
Five configurations:
  - RAMD         : full system
  - RAMD_{noCC}  : without component context
  - RAMD_{noFS}  : without few-shot examples
  - RAMD_{noMR}  : without multi-round refinement
  - RAMD_{noSpec}: without spec knowledge
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

mpl.rcParams.update({
    "font.size":        20,
    "axes.titlesize":   22,
    "axes.labelsize":   20,
    "xtick.labelsize":  19,
    "ytick.labelsize":  19,
    "legend.fontsize":  19,
})

# ---- Data ----
methods = [
    r"$\text{RAMD}$",
    r"$\text{RAMD}_{\text{noCC}}$",
    r"$\text{RAMD}_{\text{noFS}}$",
    r"$\text{RAMD}_{\text{noMR}}$",
    r"$\text{RAMD}_{\text{noSpec}}$",
]

# (accuracy [%], runtime_seconds, total_llm_tokens)
# Accuracy values multiplied by 100 to convert to percentages
data = {
    "IoT": [
        (74.4137, 1131.5,  643057.3),
        (66.4390, 1282.5,  861441.7),
        (61.0241, 1770.0,  814827.8),
        (61.1397, 1847.3, 1379615.5),
        (69.4204, 1248.3,  749657.5),
    ],
    "Self-driving car": [
        (73.0442, 1055.5,  177513.1),
        (71.9547, 1305.0,  793115.7),
        (68.9761,  362.0,   89607.2),
        (71.1093,  341.0,  185453.4),
        (67.2366,  358.9,  140625.5),
    ],
    "Smart parking": [
        (74.6610, 2995.0,  187302.6),
        (72.5190, 2182.5, 1230317.6),
        (72.8245,  438.5,  121982.2),
        (72.6085,  495.0,  186675.8),
        (73.1302,  598.9,  152201.9),
    ],
}

colors = [
    "#1f77b4",  # RAMD          — blue
    "#ff7f0e",  # RAMD_noCC     — orange
    "#2ca02c",  # RAMD_noFS     — green
    "#d62728",  # RAMD_noMR     — red
    "#9467bd",  # RAMD_noSpec   — purple
]

# ---- Compute Overall data ----
# Accuracy : arithmetic mean
# Runtime & Total LLM Tokens : geometric mean (scale-invariant for log-scale axes)
domains_list = list(data.keys())
overall_rows = []
for m_idx in range(len(methods)):
    accs = [data[d][m_idx][0] for d in domains_list]
    rts  = [data[d][m_idx][1] for d in domains_list]
    toks = [data[d][m_idx][2] for d in domains_list]
    overall_rows.append((
        float(np.mean(accs)),
        float(np.exp(np.mean(np.log(rts)))),
        float(np.exp(np.mean(np.log(toks)))),
    ))

plot_data = {"Overall": overall_rows}
plot_data.update(data)

# ---- Pareto frontier ----
def pareto_indices(points):
    """points: list of (cost, accuracy). Lower cost + higher accuracy = better."""
    n = len(points)
    pareto = []
    for i in range(n):
        dominated = any(
            points[j][0] <= points[i][0] and points[j][1] >= points[i][1] and
            (points[j][0] < points[i][0] or points[j][1] > points[i][1])
            for j in range(n) if j != i
        )
        if not dominated:
            pareto.append(i)
    return pareto

# ---- Marker size scaling (shared across all panels) ----
all_runtimes = [r for d in data.values() for (_, r, _) in d]
rt_min, rt_max = min(all_runtimes), max(all_runtimes)

def runtime_to_size(rt):
    norm = (rt - rt_min) / (rt_max - rt_min)
    return 80 + norm * 420

# ---- Smart text positioning ----
def smart_offset(cost, acc, all_points_log, ax_xlim_log, ax_ylim):
    log_cost = np.log10(cost)
    offsets = {
        "top-right":    ( 10,  10),
        "top-left":     (-10,  10),
        "bottom-right": ( 10, -14),
        "bottom-left":  (-10, -14),
    }
    aligns = {
        "top-right":    ("left",  "bottom"),
        "top-left":     ("right", "bottom"),
        "bottom-right": ("left",  "top"),
        "bottom-left":  ("right", "top"),
    }
    best_dir, best_score = "top-right", -1
    x_range = ax_xlim_log[1] - ax_xlim_log[0]
    y_range = ax_ylim[1] - ax_ylim[0]
    for direction, (dx, dy) in offsets.items():
        vx = 1 if "right" in direction else -1
        vy = 1 if "top"   in direction else -1
        min_dist = 1e9
        for lc, a in all_points_log:
            if lc == log_cost and a == acc:
                continue
            ndx = (lc - log_cost) / x_range
            ndy = (a  - acc)      / y_range
            if np.sign(ndx) == vx and np.sign(ndy) == vy:
                min_dist = min(min_dist, np.sqrt(ndx**2 + ndy**2))
        edge_x = (ax_xlim_log[1] - log_cost) / x_range if vx > 0 else (log_cost - ax_xlim_log[0]) / x_range
        edge_y = (ax_ylim[1] - acc) / y_range           if vy > 0 else (acc - ax_ylim[0]) / y_range
        score = min(min_dist, min(edge_x, edge_y) * 2)
        if score > best_score:
            best_score, best_dir = score, direction
    return offsets[best_dir], aligns[best_dir]

# ---- Figure: Overall + 3 domains + size legend ----
fig = plt.figure(figsize=(30, 8))
gs = GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.35], wspace=0.28)

fig.suptitle(
    "Ablation study — Accuracy vs Total LLM Tokens (marker size ∝ Runtime)",
    fontsize=18, fontweight="bold", y=1.06,
)

for i, (domain, rows) in enumerate(plot_data.items()):
    ax = fig.add_subplot(gs[0, i])

    costs      = [r[2] for r in rows]
    accuracies = [r[0] for r in rows]
    runtimes   = [r[1] for r in rows]

    log_costs = [np.log10(c) for c in costs]
    log_pad = (max(log_costs) - min(log_costs)) * 0.30 + 0.08
    ax_xlim_log = (min(log_costs) - log_pad, max(log_costs) + log_pad)
    ax.set_xlim(10**ax_xlim_log[0], 10**ax_xlim_log[1])

    acc_pad = (max(accuracies) - min(accuracies)) * 0.25 + 0.3
    ax_ylim = (min(accuracies) - acc_pad, max(accuracies) + acc_pad)
    ax.set_ylim(ax_ylim)

    # Pareto frontier
    par_idx = pareto_indices(list(zip(costs, accuracies)))
    par_sorted = sorted(par_idx, key=lambda idx: costs[idx])
    if len(par_sorted) >= 2:
        ax.plot(
            [costs[idx] for idx in par_sorted],
            [accuracies[idx] for idx in par_sorted],
            linestyle="--", color="gray", alpha=0.6, linewidth=1.5, zorder=1,
        )

    all_points_log = list(zip(log_costs, accuracies))
    for k, (acc, rt, cost) in enumerate(rows):
        is_pareto = k in par_idx
        ax.scatter(
            cost, acc,
            s=runtime_to_size(rt),
            color=colors[k],
            alpha=0.85,
            edgecolors="black" if is_pareto else "white",
            linewidths=1.8   if is_pareto else 0.8,
            zorder=3,
        )
        (dx, dy), (ha, va) = smart_offset(cost, acc, all_points_log, ax_xlim_log, ax_ylim)
        ax.annotate(
            f"{rt:.0f}s",
            (cost, acc),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=18, color="#222", ha=ha, va=va,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.7),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Total LLM Tokens (log scale)", fontsize=18)
    if i == 0:
        ax.set_ylabel("Accuracy (%)", fontsize=18)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.tick_params(axis="x", which="both", labelsize=9 if domain == "Overall" else 19)
    ax.tick_params(axis="y", which="both", labelsize=19)
    ax.grid(True, which="both", alpha=0.25)

    # Overall panel: distinct background + aggregation note
    if domain == "Overall":
        ax.set_facecolor("#f4f6fa")
        ax.set_title(domain, fontsize=20, fontweight="bold", pad=8, color="#1a3a6b")
        ax.text(
            0.5, -0.20,
            "Accuracy: mean  |  Runtime & Tokens: geometric mean",
            transform=ax.transAxes, fontsize=17, ha="center",
            color="#555", style="italic",
        )
        ax.spines["right"].set_linewidth(1.8)
        ax.spines["right"].set_color("#aaaaaa")
        ax.spines["right"].set_linestyle((0, (4, 2)))
    else:
        ax.set_title(domain, fontsize=20, fontweight="bold", pad=8)

    # "Better" arrow — just above title
    ax.annotate(
        "", xy=(0.02, 1.10), xytext=(0.18, 1.10),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="#2a7a2a", lw=2),
    )
    ax.text(
        0.20, 1.10, "Better (↑ acc, ↓ tokens)",
        transform=ax.transAxes, fontsize=18, color="#2a7a2a",
        fontweight="bold", verticalalignment="center",
    )

# ---- Size legend panel ----
size_ax = fig.add_subplot(gs[0, 4])
size_ax.set_xlim(0, 1)
size_ax.set_ylim(0, 1)
size_ax.axis("off")
size_ax.set_title("Runtime → Marker size", fontsize=18, fontweight="bold", pad=12)

sample_runtimes = [
    int(rt_min),
    int(rt_min + (rt_max - rt_min) * 0.33),
    int(rt_min + (rt_max - rt_min) * 0.66),
    int(rt_max),
]
for rt, yp in zip(sample_runtimes, [0.78, 0.60, 0.42, 0.24]):
    size_ax.scatter(
        0.25, yp, s=runtime_to_size(rt),
        color="#888", alpha=0.7, edgecolors="white", linewidths=0.8,
        transform=size_ax.transAxes,
    )
    size_ax.text(0.52, yp, f"{rt} s", transform=size_ax.transAxes, fontsize=19, va="center")

size_ax.text(
    0.5, 0.08, "Black outline =\nPareto-optimal",
    transform=size_ax.transAxes, fontsize=18,
    ha="center", va="center", color="#333",
    bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="#666"),
)

# ---- Shared method legend ----
method_handles = [
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor=colors[k], markersize=10,
           markeredgecolor="black", markeredgewidth=0.5,
           label=methods[k])
    for k in range(len(methods))
]
pareto_handle = Line2D([0], [0], color="gray", linestyle="--",
                       linewidth=1.5, label="Pareto frontier")
fig.legend(
    handles=method_handles + [pareto_handle],
    loc="lower center", ncol=6, fontsize=19,
    bbox_to_anchor=(0.58, -0.06),
    frameon=True, facecolor="white", edgecolor="#ccc",
)

plt.tight_layout()
out_path = "RQ2_ablation.pdf"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
