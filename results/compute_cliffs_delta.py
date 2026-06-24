import pandas as pd
import numpy as np
from itertools import combinations

# ---------- helpers ----------

def cliffs_delta(x, y):
    """
    Cliff's delta: non-parametric effect size between two samples.
    delta = (# pairs where x > y  -  # pairs where x < y) / (n_x * n_y)
    Range [-1, 1].
    """
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    n_x, n_y = len(x), len(y)
    more = int(sum(xi > yj for xi in x for yj in y))
    less = int(sum(xi < yj for xi in x for yj in y))
    return float((more - less) / (n_x * n_y))

def interpret(d):
    """Standard magnitude thresholds from Cliff (1993)."""
    a = abs(d)
    if a < 0.147:
        return "negligible"
    elif a < 0.33:
        return "small"
    elif a < 0.474:
        return "medium"
    else:
        return "large"

# ---------- load & tidy ----------
df = pd.read_csv(
    "RA Component mapping - statistical_analysis.csv",
    dtype=str,
)
df.columns = ["Domain", "Method", "R1", "R2", "R3", "R4", "R5"]

# forward-fill Domain (merged cells appear as empty strings in CSV)
df["Domain"] = df["Domain"].replace("", np.nan).ffill()
df["Domain"] = df["Domain"].str.strip()
df["Method"] = df["Method"].str.strip()

run_cols = ["R1", "R2", "R3", "R4", "R5"]
for c in run_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ---------- build observations per method ----------
# Pool all 5 runs across all domains → up to 15 observations per method.
# This gives more statistical power than using only domain-level means.
method_scores: dict[str, list[float]] = {}
method_order = []  # preserve original order
for _, row in df.iterrows():
    m = row["Method"]
    if m not in method_scores:
        method_scores[m] = []
        method_order.append(m)
    method_scores[m].extend(row[run_cols].dropna().tolist())

methods = method_order

print(f"Methods ({len(methods)}): {methods}")
for m, v in method_scores.items():
    print(f"  {m:30s} n={len(v)}  values={np.round(v, 4)}")
print()

# ---------- pairwise Cliff's delta ----------
records = []
for m_a, m_b in combinations(methods, 2):
    vals_a = method_scores[m_a]
    vals_b = method_scores[m_b]
    d = cliffs_delta(vals_a, vals_b)
    records.append({
        "Config_A": m_a,
        "Config_B": m_b,
        "n_A": len(vals_a),
        "n_B": len(vals_b),
        "Cliffs_delta": round(d, 4),
        "magnitude": interpret(d),
    })

pairs_df = pd.DataFrame(records)

# ---------- square numeric matrix (antisymmetric: delta(A,B) = -delta(B,A)) ----------
matrix = pd.DataFrame(np.nan, index=methods, columns=methods)
for _, row in pairs_df.iterrows():
    matrix.loc[row["Config_A"], row["Config_B"]] = row["Cliffs_delta"]
    matrix.loc[row["Config_B"], row["Config_A"]] = -row["Cliffs_delta"]
for m in methods:
    matrix.loc[m, m] = 0.0

# ---------- magnitude label matrix ----------
mag_matrix = pd.DataFrame("", index=methods, columns=methods)
for _, row in pairs_df.iterrows():
    mag = row["magnitude"]
    mag_matrix.loc[row["Config_A"], row["Config_B"]] = mag
    mag_matrix.loc[row["Config_B"], row["Config_A"]] = mag
for m in methods:
    mag_matrix.loc[m, m] = "-"

# ---------- combined matrix: "delta (magnitude)" ----------
combined = pd.DataFrame("", index=methods, columns=methods)
for _, row in pairs_df.iterrows():
    d, mag = row["Cliffs_delta"], row["magnitude"]
    combined.loc[row["Config_A"], row["Config_B"]] = f"{d:+.4f} ({mag})"
    combined.loc[row["Config_B"], row["Config_A"]] = f"{-d:+.4f} ({mag})"
for m in methods:
    combined.loc[m, m] = "0 (-)"

# ---------- save ----------
pairs_df.to_csv("cliffs_delta_pairs.csv", index=False)
matrix.round(4).to_csv("cliffs_delta_matrix.csv")
mag_matrix.to_csv("cliffs_delta_magnitude_matrix.csv")
combined.to_csv("cliffs_delta_combined_matrix.csv")

print("Saved:")
print("  cliffs_delta_pairs.csv              — full pairwise list")
print("  cliffs_delta_matrix.csv             — N×N numeric delta matrix")
print("  cliffs_delta_magnitude_matrix.csv   — N×N magnitude label matrix")
print("  cliffs_delta_combined_matrix.csv    — N×N combined 'delta (magnitude)' matrix\n")

print("=== Numeric delta matrix ===")
print(matrix.round(3).to_string())
print()
print("=== Magnitude matrix ===")
print(mag_matrix.to_string())
