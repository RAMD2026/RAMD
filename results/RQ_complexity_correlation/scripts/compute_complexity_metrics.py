"""
Compute per-model complexity metrics for the correlation study, one row per
graph_id, covering four complexity dimensions:

  - size:               num_nodes
  - connectivity:        density (size-normalized, independent of raw size)
  - hierarchy:            hierarchy_max_depth (from extract_hierarchy_from_aadl.py)
  - naming ambiguity:    LMA (Lexical Match Ambiguity)

LMA definition
--------------
For a node n and the RA catalog C_R = {r_1, ..., r_k} of the model's
reference architecture:
  1. Build a TF-IDF text representation for n and for each r_i, reusing the
     exact same text-building functions as baselines/tfidf_lexical_matching.py
     (name + category + classifier + ports for nodes; name + definition +
     parent + tag + mandatory for RA components).
  2. Compute cosine similarity sim(n, r_i) for every r_i.
  3. Normalize the similarity vector into a probability distribution
     p(r_i | n) = sim(n, r_i) / sum_j sim(n, r_j).
     If all similarities are 0 (no lexical overlap with anything), fall back
     to a uniform distribution over C_R (maximal ambiguity).
  4. H(n) = -sum_i p(r_i|n) * log(p(r_i|n))   (Shannon entropy)
  5. H_norm(n) = H(n) / log(|C_R|)             (normalized to [0, 1])
  LMA(model) = mean_n H_norm(n) over all nodes n in the model.

H_norm(n) = 0 means the TF-IDF similarity vector is a spike on a single RA
candidate (lexically decisive, whether or not that candidate is correct).
H_norm(n) close to 1 means similarities are spread near-uniformly across
RA candidates (the node's text carries no lexical signal to prefer one RA
candidate over another) -- i.e. real naming ambiguity.

This metric does not use ground-truth labels: it only reflects how
separable the RA candidates are from the surface text of a node.
"""

import csv
import glob
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "baselines"))

from tfidf_lexical_matching import (  # noqa: E402
    build_node_text,
    build_ra_texts,
    parse_ra_definitions,
)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"
HIERARCHY_CSV = OUTPUT_DIR / "hierarchy_metrics.csv"
OUTPUT_CSV = OUTPUT_DIR / "complexity_metrics.csv"

# domain key -> (mapping_data subfolder, RA description file)
DOMAINS = {
    "iot": ("mapping_data/iot_mapping_files", "RA/iot_ra_components.txt"),
    "smartparking": (
        "mapping_data/SMART_PARKING_mapping_files",
        "RA/smartparking_ra_components.txt",
    ),
    "selfdriving": (
        "mapping_data/SELF_DRIVING_CAR_mapping_files",
        "RA/autonomous_driving_system_ra_components.txt",
    ),
}


def compute_density(num_nodes: int, num_edges: int) -> float:
    if num_nodes <= 1:
        return 0.0
    return num_edges / (num_nodes * (num_nodes - 1))


def compute_node_entropy(sims: np.ndarray) -> float:
    sims = np.clip(sims, 0.0, None)
    total = sims.sum()

    if total <= 0:
        # No lexical overlap with any RA candidate: maximal ambiguity.
        probs = np.ones_like(sims) / len(sims)
    else:
        probs = sims / total

    nonzero = probs[probs > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)))

    return entropy


def compute_lma_for_model(data: Dict, ra_defs: Dict[str, str]) -> float:
    ra_items = build_ra_texts(data, ra_defs)
    num_ra = len(ra_items)

    if num_ra <= 1:
        return 0.0

    ra_texts = [it["ra_text"] for it in ra_items]
    log_num_ra = math.log(num_ra)

    node_entropies = []

    for node in data["nodes"]:
        node_text = build_node_text(node)
        documents = [node_text] + ra_texts

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), stop_words="english", lowercase=True
        )
        tfidf_matrix = vectorizer.fit_transform(documents)

        sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

        entropy = compute_node_entropy(sims)
        node_entropies.append(entropy / log_num_ra)

    return float(np.mean(node_entropies)) if node_entropies else 0.0


def load_hierarchy_map() -> Dict[str, Dict[str, str]]:
    hierarchy_map = {}

    with open(HIERARCHY_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            hierarchy_map[row["graph_id"]] = row

    return hierarchy_map


def main() -> None:
    hierarchy_map = load_hierarchy_map()

    rows = []

    for domain, (mapping_subdir, ra_desc_path) in DOMAINS.items():
        ra_defs = parse_ra_definitions(str(REPO_ROOT / ra_desc_path))

        json_paths = sorted(
            glob.glob(str(REPO_ROOT / mapping_subdir / "*.json"))
        )

        for jp in json_paths:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)

            graph_id = data["graph_id"]
            num_nodes = len(data["nodes"])
            num_edges = len(data["edges"])
            density = compute_density(num_nodes, num_edges)
            lma = compute_lma_for_model(data, ra_defs)

            hierarchy_row = hierarchy_map.get(graph_id, {})
            hierarchy_max_depth = hierarchy_row.get("hierarchy_max_depth", "")
            hierarchy_avg_depth = hierarchy_row.get("hierarchy_avg_depth", "")

            rows.append(
                {
                    "graph_id": graph_id,
                    "domain": domain,
                    "num_nodes": num_nodes,
                    "num_edges": num_edges,
                    "density": round(density, 6),
                    "hierarchy_max_depth": hierarchy_max_depth,
                    "hierarchy_avg_depth": hierarchy_avg_depth,
                    "lma": round(lma, 6),
                }
            )

            print(
                f"[{domain}] {graph_id}: nodes={num_nodes} edges={num_edges} "
                f"density={density:.4f} depth={hierarchy_max_depth} lma={lma:.4f}"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "graph_id",
                "domain",
                "num_nodes",
                "num_edges",
                "density",
                "hierarchy_max_depth",
                "hierarchy_avg_depth",
                "lma",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
