import csv
import glob
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple


AADL_MODELS_DIR = (
    "RA/AADL_MODELS"
)

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_CSV = Path(__file__).resolve().parents[1] / "data" / "hierarchy_metrics.csv"

# domain key -> (mapping_data subfolder, raw .aaxl2 subfolder under AADL_MODELS_DIR)
DOMAINS = {
    "iot": ("mapping_data/iot_mapping_files", "SMART_HOME"),
    "smartparking": (
        "mapping_data/SMART_PARKING_mapping_files",
        "SMART_PARKING",
    ),
    "selfdriving": (
        "mapping_data/SELF_DRIVING_CAR_mapping_files",
        "SELF_DRIVING_CAR",
    ),
}


def build_depth_map(root: ET.Element) -> Dict[str, int]:
    """Map component name (lowercased) -> nesting depth for every
    <componentInstance> under root. Root itself is depth 0 but is not a
    componentInstance so it is not added here directly."""

    depth_map: Dict[str, int] = {}

    def recurse(elem: ET.Element, depth: int) -> None:
        for child in elem:
            tag = child.tag.split("}")[-1]  # strip any namespace
            if tag == "componentInstance":
                name = child.get("name", "")
                if name:
                    key = name.strip().lower()
                    # Keep the shallowest depth if a name repeats.
                    if key not in depth_map or depth < depth_map[key]:
                        depth_map[key] = depth
                recurse(child, depth + 1)

    recurse(root, 1)
    return depth_map


def compute_model_hierarchy(
    aaxl2_path: str, node_names: List[str]
) -> Tuple[int, float, int, int]:
    tree = ET.parse(aaxl2_path)
    root = tree.getroot()

    depth_map = build_depth_map(root)

    depths = []
    matched = 0

    for name in node_names:
        key = name.strip().lower()
        depth = depth_map.get(key, 0)
        if key in depth_map:
            matched += 1
        depths.append(depth)

    max_depth = max(depths) if depths else 0
    avg_depth = sum(depths) / len(depths) if depths else 0.0

    return max_depth, avg_depth, matched, len(node_names)


def main() -> None:
    rows = []
    missing_files = []

    for domain, (mapping_subdir, aaxl2_subdir) in DOMAINS.items():
        json_paths = sorted(
            glob.glob(str(REPO_ROOT / mapping_subdir / "*.json"))
        )

        for jp in json_paths:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)

            graph_id = data["graph_id"]
            node_names = [n["name"] for n in data["nodes"]]

            aaxl2_path = os.path.join(AADL_MODELS_DIR, aaxl2_subdir, graph_id)

            if not os.path.isfile(aaxl2_path):
                missing_files.append(aaxl2_path)
                continue

            max_depth, avg_depth, matched, total = compute_model_hierarchy(
                aaxl2_path, node_names
            )

            rows.append(
                {
                    "graph_id": graph_id,
                    "domain": domain,
                    "hierarchy_max_depth": max_depth,
                    "hierarchy_avg_depth": round(avg_depth, 4),
                    "matched_nodes": matched,
                    "total_nodes": total,
                }
            )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "graph_id",
                "domain",
                "hierarchy_max_depth",
                "hierarchy_avg_depth",
                "matched_nodes",
                "total_nodes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")

    if missing_files:
        print(f"WARNING: {len(missing_files)} .aaxl2 files not found:")
        for m in missing_files[:10]:
            print(f"  {m}")

    total_matched = sum(r["matched_nodes"] for r in rows)
    total_nodes = sum(r["total_nodes"] for r in rows)
    if total_nodes:
        print(
            f"Node-name match rate: {total_matched}/{total_nodes} "
            f"({100 * total_matched / total_nodes:.1f}%)"
        )


if __name__ == "__main__":
    main()
