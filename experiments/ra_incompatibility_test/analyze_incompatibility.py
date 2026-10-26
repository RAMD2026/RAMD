"""
Analyze the results produced by run_experiment.py and summarize the
"incompatibility signals" RAMD exposes when a model is paired with the
wrong Reference Architecture:

  1. Compliance never reached (persistent RA-constraint violations).
  2. Structural completeness violated (components left unmapped, phi(c_m) = -).

Reads only from experiments/ra_incompatibility_test/results/ and writes
REPORT.md + summary.csv into experiments/ra_incompatibility_test/.
Does not import or touch any file of the main pipeline.
"""

import csv
import glob
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"

_CONSTRAINT_COUNT_RE = re.compile(
    r"(\d+)\s+constraint\(s\)\s+have\s+not\s+been\s+satisfied",
    re.IGNORECASE,
)


def extract_violation_count(check_constraints_log: str) -> Optional[int]:
    if not check_constraints_log:
        return None
    m = _CONSTRAINT_COUNT_RE.search(check_constraints_log)
    return int(m.group(1)) if m else None


def load_pair_results(pair_name: str) -> List[Dict[str, Any]]:
    paths = sorted(glob.glob(str(RESULTS_DIR / pair_name / "*__result.json")))
    results = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            results.append(json.load(f))
    return results


def analyze_pair(pair_name: str) -> Dict[str, Any]:
    results = load_pair_results(pair_name)
    rows = []
    n_compliant = 0
    for r in results:
        violation_count = None if r["compliant"] else extract_violation_count(r["check_constraints_log"])
        if r["compliant"]:
            n_compliant += 1
        rows.append(
            {
                "pair": pair_name,
                "file": r["file"],
                "source_domain": r.get("source_domain"),
                "injected_ra_domain": r.get("injected_ra_domain"),
                "total_nodes": r["total_nodes"],
                "unmapped_count": r["unmapped_count"],
                "compliant": r["compliant"],
                "final_violation_count": violation_count,
                "execution_time_seconds": r["execution_time_seconds"],
                "total_llm_cost_tokens": r["total_llm_cost"],
            }
        )
    n = len(results)
    return {
        "pair_name": pair_name,
        "n_models": n,
        "n_compliant": n_compliant,
        "pct_compliant": (n_compliant / n * 100.0) if n else 0.0,
        "n_with_unmapped": sum(1 for r in rows if r["unmapped_count"] > 0),
        "avg_unmapped": sum(r["unmapped_count"] for r in rows) / n if n else 0.0,
        "avg_final_violation_count": (
            sum(r["final_violation_count"] for r in rows if r["final_violation_count"] is not None)
            / max(1, sum(1 for r in rows if r["final_violation_count"] is not None))
        ),
        "rows": rows,
    }


def main():
    pair_dirs = sorted(d.name for d in RESULTS_DIR.iterdir() if d.is_dir())
    if not pair_dirs:
        raise SystemExit(f"No pair result folders found under {RESULTS_DIR}. Run run_experiment.py first.")

    all_rows = []
    pair_stats = []
    for pair_name in pair_dirs:
        stats = analyze_pair(pair_name)
        pair_stats.append(stats)
        all_rows.extend(stats["rows"])

    # --- CSV ---
    csv_path = EXPERIMENT_DIR / "summary.csv"
    fieldnames = [
        "pair", "file", "source_domain", "injected_ra_domain", "total_nodes",
        "unmapped_count", "compliant", "final_violation_count",
        "execution_time_seconds", "total_llm_cost_tokens",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # --- REPORT.md ---
    lines = []
    lines.append("# RA-Incompatibility Test — Results\n")
    lines.append(
        "This experiment addresses the question: *\"What happens when architecture "
        "models incompatible with the selected reference architecture are "
        "considered? Is the approach able to identify such cases?\"*\n"
    )
    lines.append(
        "Each pair below runs the unmodified RAMD pipeline (Option 2 / "
        "per-component strategy, RA-compliance refinement loop, few-shot "
        "examples, RA descriptions, RAG over the RA-DB, no Known Mappings DB) "
        "on real AADL models paired with a Reference Architecture from a "
        "**different domain** than the one they were built for.\n"
    )
    lines.append("| Pair | #models | % compliant | avg final violations (non-compliant) | #models with unmapped component(s) | avg unmapped/model |")
    lines.append("|---|---|---|---|---|---|")
    for s in pair_stats:
        lines.append(
            f"| {s['pair_name']} | {s['n_models']} | {s['pct_compliant']:.0f}% "
            f"| {s['avg_final_violation_count']:.2f} | {s['n_with_unmapped']} "
            f"| {s['avg_unmapped']:.2f} |"
        )

    lines.append("\n## Per-model detail\n")
    lines.append("| Pair | File | Compliant | Unmapped/Total | Final violations |")
    lines.append("|---|---|---|---|---|")
    for r in all_rows:
        lines.append(
            f"| {r['pair']} | {r['file']} | {r['compliant']} "
            f"| {r['unmapped_count']}/{r['total_nodes']} "
            f"| {r['final_violation_count']} |"
        )

    lines.append(
        "\n## Interpretation\n\n"
        "A model is flagged as incompatible with the selected RA when RAMD "
        "cannot drive the violation count to zero within `N_max` iterations "
        "and/or is forced to leave components unassigned. In that case, "
        "`run_option2_mapping` (mapping_core.py) returns `best_invalid_mapping` "
        "together with `best_invalid_log`, i.e. the best-effort mapping plus "
        "its unresolved violations, rather than a validated compliant result "
        "-- this persistent violation list is the signal used to flag "
        "incompatibility.\n"
    )

    report_path = EXPERIMENT_DIR / "REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")
    for s in pair_stats:
        print(
            f"{s['pair_name']}: {s['n_compliant']}/{s['n_models']} compliant, "
            f"avg unmapped/model={s['avg_unmapped']:.2f}"
        )


if __name__ == "__main__":
    main()
