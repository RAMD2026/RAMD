# Complexity-Effectiveness Correlation Analysis

This analysis answers the question:

> Is there a correlation between the complexity of the architectural models
> and the effectiveness of the approach? How does the approach behave when
> model size, connectivity, hierarchy and naming ambiguity change?


## Complexity metrics (one per complexity dimension)

| Dimension | Metric | Definition |
|---|---|---|
| Size | `num_nodes` | number of AADL components in the model |
| Connectivity | `density` | `\|E\| / (\|V\|(\|V\|-1))`, size-normalized so it is not a proxy for size |
| Hierarchy | `hierarchy_max_depth` | maximum nesting depth of the true AADL containment tree, parsed directly from the raw `.aaxl2` instance models (`<componentInstance>` nesting), not a graph-based proxy |
| Naming ambiguity | `lma` (Lexical Match Ambiguity) | mean, over all nodes, of the Shannon entropy (normalized to [0,1] by `log(\|C_R\|)`) of the TF-IDF cosine-similarity distribution between a node's text and every RA component. 0 = the node's text lexically points to a single RA candidate; 1 = the text is equally (in)compatible with every RA candidate. Uses only the same text representation as `baselines/tfidf_lexical_matching.py`; does not use ground-truth labels, so it is not circular with the accuracy score it is later correlated against |

RA-level complexity (e.g. `#RA_components`) was intentionally **not** used as
a separate variable: with only 3 RAs there is no within-domain variance to
correlate against, so it cannot support a statistical test. Domain is instead
kept as an explicit stratification factor (see below), which controls for
RA-specific effects without inventing a composite metric.

## Effectiveness variable

Mean `avg_score` across the 5 repeated threads, per `graph_id`, for two
configurations (matching the paper's own aggregation convention):

- **RAMD** - full pipeline, no Known Mappings (`results/RQ2/*/RAMD`)
- **RAMD +MK** - full pipeline + Known Mappings, GPT-4o-mini, the paper's
  best configuration (`results/RQ3_RQ5/*/RAMD_RAG_mini`)

## Statistical method

- **Spearman's rank correlation** (rho, p-value) between each metric and
  accuracy, computed both **pooled** across all 64 models and **per RA
  domain** (IoT n=25, Smart Parking n=29, Self-driving Car n=10).
- **Holm-Bonferroni correction** across the 4 metrics within each
  (configuration, scope) family, to control the false-positive rate from
  testing 4 metrics at once.
- Effect-size interpretation of `|rho|`: `<0.10` negligible, `<0.30` weak,
  `<0.50` moderate, `<0.70` strong, `>=0.70` very strong. The `0.10/0.30/0.50`
  breakpoints follow Cohen's (1988, *Statistical Power Analysis for the
  Behavioral Sciences*, 2nd ed.) small/medium/large convention for
  correlation coefficients; the `negligible` (below 0.10) and
  `strong`/`very strong` (above 0.50/0.70) bands are a common extension of
  Cohen's 3-tier scale used in later empirical/methodological work, since
  Cohen (1988) itself only defines 3 tiers. This 5-tier framing was chosen
  to match, in spirit, the negligible/small/medium/large language already
  used for Cliff's delta elsewhere in the paper, though the two are
  different statistics with independently sourced thresholds.
- A scatter plot per (metric, configuration), colored by domain with a
  linear trend line, to visualize how effectiveness behaves as the metric
  changes.

## How to reproduce

```bash
cd "<repo_root>"
python3 results/RQ_complexity_correlation/scripts/extract_hierarchy_from_aadl.py
python3 results/RQ_complexity_correlation/scripts/compute_complexity_metrics.py
python3 results/RQ_complexity_correlation/scripts/aggregate_effectiveness.py
python3 results/RQ_complexity_correlation/scripts/correlation_analysis.py
```

`extract_hierarchy_from_aadl.py` reads the raw `.aaxl2` AADL instance models.

## Files

```
data/
  hierarchy_metrics.csv    # per-model containment depth (from raw .aaxl2)
  complexity_metrics.csv   # per-model size/connectivity/hierarchy/LMA
  model_accuracy.csv       # per-model mean accuracy, RAMD & RAMD +MK
  merged_dataset.csv       # the joined dataset used for the analysis (n=64)
tables/
  correlation_pooled.csv       # Spearman rho/p, pooled across all 64 models
  correlation_per_domain.csv   # Spearman rho/p, per RA domain
plots/
  scatter_<metric>_<config>.png
```

## Summary of pooled results (n=64)

| Metric | RAMD rho (p_adj) | RAMD +MK rho (p_adj) |
|---|---|---|
| Size (#nodes) | -0.282 (0.095, weak) | **-0.466 (0.0004, moderate)** |
| Connectivity (density) | -0.076 (1.000, negligible) | +0.191 (0.262, weak) |
| Hierarchy (max depth) | -0.179 (0.472, weak) | -0.219 (0.245, weak) |
| Naming ambiguity (LMA) | -0.041 (1.000, negligible) | +0.164 (0.262, weak) |

Model **size** is the only dimension with a statistically significant,
Holm-corrected correlation with accuracy (negative: larger models are
mapped less accurately), and the effect is stronger for RAMD +MK than for
plain RAMD. Connectivity, hierarchy and naming ambiguity show weak-to-
negligible pooled correlations; see `correlation_per_domain.csv` for the
per-RA breakdown, since some of these relationships are domain-dependent
(e.g. size correlates more strongly with accuracy within Smart Parking and
IoT than within Self-driving Car).
