import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
TABLES_DIR = BASE_DIR / "tables"
PLOTS_DIR = BASE_DIR / "plots"

METRICS = [
    ("num_nodes", "Size (#nodes)"),
    ("density", "Connectivity (density)"),
    ("hierarchy_max_depth", "Hierarchy (max containment depth)"),
    ("lma", "Naming ambiguity (LMA)"),
]

CONFIGS = [
    ("ramd_accuracy", "RAMD"),
    ("ramd_mk_accuracy", "RAMD +MK"),
]

DOMAIN_LABELS = {
    "iot": "IoT",
    "smartparking": "Smart Parking",
    "selfdriving": "Self-driving Car",
}

DOMAIN_COLORS = {
    "iot": "#1f77b4",
    "smartparking": "#ff7f0e",
    "selfdriving": "#2ca02c",
}


def effect_size_label(rho: float) -> str:
    abs_rho = abs(rho)

    if abs_rho < 0.10:
        return "negligible"
    if abs_rho < 0.30:
        return "weak"
    if abs_rho < 0.50:
        return "moderate"
    if abs_rho < 0.70:
        return "strong"
    return "very strong"


def load_merged_dataset() -> pd.DataFrame:
    complexity_df = pd.read_csv(DATA_DIR / "complexity_metrics.csv")
    accuracy_df = pd.read_csv(DATA_DIR / "model_accuracy.csv")

    merged = complexity_df.merge(
        accuracy_df,
        on=["graph_id", "domain"],
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(complexity_df) or len(merged) != len(accuracy_df):
        print(
            f"WARNING: merge size mismatch. complexity={len(complexity_df)} "
            f"accuracy={len(accuracy_df)} merged={len(merged)}"
        )

    merged.to_csv(DATA_DIR / "merged_dataset.csv", index=False)
    return merged


def run_family(
    df: pd.DataFrame, config_col: str, scope: str
) -> List[Dict]:
    """Run Spearman correlation for all 4 metrics against config_col on df,
    then Holm-Bonferroni-correct the 4 p-values as one family."""

    rows = []
    p_values = []

    for metric_col, metric_label in METRICS:
        sub = df[[metric_col, config_col]].dropna()
        n = len(sub)

        if n < 3 or sub[metric_col].nunique() < 2:
            rho, p = float("nan"), float("nan")
        else:
            rho, p = spearmanr(sub[metric_col], sub[config_col])

        rows.append(
            {
                "config": config_col,
                "scope": scope,
                "metric": metric_col,
                "metric_label": metric_label,
                "n": n,
                "rho": rho,
                "p_value": p,
            }
        )
        p_values.append(p)

    valid_idx = [i for i, p in enumerate(p_values) if not np.isnan(p)]

    if valid_idx:
        _, p_adj_valid, _, _ = multipletests(
            [p_values[i] for i in valid_idx], method="holm"
        )
        p_adj_full = [float("nan")] * len(p_values)
        for i, p_adj in zip(valid_idx, p_adj_valid):
            p_adj_full[i] = p_adj
    else:
        p_adj_full = [float("nan")] * len(p_values)

    for row, p_adj in zip(rows, p_adj_full):
        row["p_adj_holm"] = p_adj
        row["effect_size"] = (
            effect_size_label(row["rho"]) if not np.isnan(row["rho"]) else "n/a"
        )

    return rows


def compute_all_correlations(
    df: pd.DataFrame,
) -> Tuple[List[Dict], List[Dict]]:
    pooled_rows = []
    per_domain_rows = []

    for config_col, config_label in CONFIGS:
        pooled = run_family(df, config_col, scope="pooled")
        for r in pooled:
            r["config_label"] = config_label
        pooled_rows.extend(pooled)

        for domain_key, domain_label in DOMAIN_LABELS.items():
            domain_df = df[df["domain"] == domain_key]
            domain_rows = run_family(domain_df, config_col, scope=domain_key)
            for r in domain_rows:
                r["config_label"] = config_label
                r["scope_label"] = domain_label
            per_domain_rows.extend(domain_rows)

    return pooled_rows, per_domain_rows


def save_table(rows: List[Dict], path: Path, fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def make_scatter_plot(
    df: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    config_col: str,
    config_label: str,
    pooled_row: Dict,
) -> None:
    sub = df[[metric_col, config_col, "domain"]].dropna()

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for domain_key, domain_label in DOMAIN_LABELS.items():
        domain_sub = sub[sub["domain"] == domain_key]
        ax.scatter(
            domain_sub[metric_col],
            domain_sub[config_col],
            label=domain_label,
            color=DOMAIN_COLORS[domain_key],
            alpha=0.75,
            edgecolors="white",
            linewidths=0.5,
            s=45,
        )

    if len(sub) >= 2 and sub[metric_col].nunique() >= 2:
        coeffs = np.polyfit(sub[metric_col], sub[config_col], 1)
        x_line = np.linspace(sub[metric_col].min(), sub[metric_col].max(), 100)
        y_line = np.polyval(coeffs, x_line)
        ax.plot(x_line, y_line, color="black", linestyle="--", linewidth=1.2)

    rho = pooled_row["rho"]
    p_adj = pooled_row["p_adj_holm"]
    effect = pooled_row["effect_size"]

    rho_text = f"rho={rho:.3f}" if not np.isnan(rho) else "rho=n/a"
    p_text = f"p_adj={p_adj:.3f}" if not np.isnan(p_adj) else "p_adj=n/a"

    ax.set_title(
        f"{config_label}: {metric_label} vs. accuracy\n"
        f"(pooled, n={pooled_row['n']}, {rho_text}, {p_text}, {effect})"
    )
    ax.set_xlabel(metric_label)
    ax.set_ylabel(f"{config_label} accuracy (avg_score)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()

    safe_config = config_col.replace("_accuracy", "")
    out_path = PLOTS_DIR / f"scatter_{metric_col}_{safe_config}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_merged_dataset()
    print(f"Merged dataset: {len(df)} models across {df['domain'].nunique()} domains")

    pooled_rows, per_domain_rows = compute_all_correlations(df)

    pooled_fields = [
        "config",
        "config_label",
        "scope",
        "metric",
        "metric_label",
        "n",
        "rho",
        "p_value",
        "p_adj_holm",
        "effect_size",
    ]
    domain_fields = pooled_fields + ["scope_label"]

    save_table(pooled_rows, TABLES_DIR / "correlation_pooled.csv", pooled_fields)
    save_table(
        per_domain_rows, TABLES_DIR / "correlation_per_domain.csv", domain_fields
    )

    pooled_lookup = {
        (r["config"], r["metric"]): r for r in pooled_rows
    }

    for metric_col, metric_label in METRICS:
        for config_col, config_label in CONFIGS:
            pooled_row = pooled_lookup[(config_col, metric_col)]
            make_scatter_plot(
                df, metric_col, metric_label, config_col, config_label, pooled_row
            )

    print(f"\nSaved {len(METRICS) * len(CONFIGS)} scatter plots to {PLOTS_DIR}")

    print("\n=== Pooled correlations (summary) ===")
    for r in pooled_rows:
        print(
            f"[{r['config_label']:10s}] {r['metric_label']:38s} "
            f"rho={r['rho']:+.3f}  p_adj={r['p_adj_holm']:.4f}  "
            f"({r['effect_size']})"
        )


if __name__ == "__main__":
    main()
