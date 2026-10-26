import csv
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_CSV = Path(__file__).resolve().parents[1] / "data" / "model_accuracy.csv"

# domain key (matching complexity_metrics.csv) -> results subfolder name
RESULT_DOMAIN_DIR = {
    "iot": "iot",
    "smartparking": "smartparking",
    "selfdriving": "autonomous",
}

RAMD_DIR_TEMPLATE = "results/RQ2/{result_domain}/RAMD"
RAMD_MK_DIR_TEMPLATE = "results/RQ3_RQ5/{result_domain}/RAMD_RAG_mini"


def collect_avg_scores(run_dir: Path) -> Dict[str, List[float]]:
    scores: Dict[str, List[float]] = defaultdict(list)

    thread_dirs = sorted(run_dir.glob("thread*"))

    for thread_dir in thread_dirs:
        for jp in thread_dir.glob("*.json"):
            if jp.name.startswith("summary_thread"):
                continue

            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                continue

            graph_id = data.get("graph_id")
            avg_score = data.get("avg_score")

            if graph_id is not None and avg_score is not None:
                scores[graph_id].append(float(avg_score))

    return scores


def main() -> None:
    rows = []

    for domain, result_domain in RESULT_DOMAIN_DIR.items():
        ramd_dir = REPO_ROOT / RAMD_DIR_TEMPLATE.format(result_domain=result_domain)
        ramd_mk_dir = REPO_ROOT / RAMD_MK_DIR_TEMPLATE.format(
            result_domain=result_domain
        )

        ramd_scores = collect_avg_scores(ramd_dir)
        ramd_mk_scores = collect_avg_scores(ramd_mk_dir)

        all_graph_ids = sorted(set(ramd_scores) | set(ramd_mk_scores))

        for graph_id in all_graph_ids:
            ramd_vals = ramd_scores.get(graph_id, [])
            ramd_mk_vals = ramd_mk_scores.get(graph_id, [])

            ramd_mean = sum(ramd_vals) / len(ramd_vals) if ramd_vals else ""
            ramd_mk_mean = (
                sum(ramd_mk_vals) / len(ramd_mk_vals) if ramd_mk_vals else ""
            )

            rows.append(
                {
                    "graph_id": graph_id,
                    "domain": domain,
                    "ramd_accuracy": (
                        round(ramd_mean, 6) if ramd_mean != "" else ""
                    ),
                    "ramd_mk_accuracy": (
                        round(ramd_mk_mean, 6) if ramd_mk_mean != "" else ""
                    ),
                    "ramd_n_threads": len(ramd_vals),
                    "ramd_mk_n_threads": len(ramd_mk_vals),
                }
            )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "graph_id",
                "domain",
                "ramd_accuracy",
                "ramd_mk_accuracy",
                "ramd_n_threads",
                "ramd_mk_n_threads",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")

    incomplete = [
        r
        for r in rows
        if r["ramd_n_threads"] != 5 or r["ramd_mk_n_threads"] != 5
    ]

    if incomplete:
        print(f"WARNING: {len(incomplete)} models do not have exactly 5/5 threads:")
        for r in incomplete[:10]:
            print(
                f"  {r['graph_id']}: ramd={r['ramd_n_threads']} "
                f"ramd_mk={r['ramd_mk_n_threads']}"
            )


if __name__ == "__main__":
    main()
