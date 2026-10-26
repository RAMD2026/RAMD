"""
Run RAMD (unmodified pipeline code) on the mismatched (model, RA) pair
built by build_mismatched_inputs.py.

This script only IMPORTS the existing pipeline modules (config_utils,
graph_utils, mapping_core, llm_clients, semantic_utils) -- it does not
edit them, and it does not touch LLM_mapping_RA.yml. All outputs of this
script are written under experiments/ra_incompatibility_test/results/.

Configuration mirrors the paper's "RAMD" ablation (Option 2 / per-component
strategy, with RA-compliance refinement loop, few-shot examples, RA
descriptions and RAG over the RA-DB) but WITHOUT the Known Mappings DB
(+MK), since historical mappings from the correct domain would not be a
meaningful signal for a deliberately incompatible RA target.

Scope (narrowed after exploration): several domain pairs were tried
(Smart Parking / Autonomous Driving forced onto the IoT RA, and onto a
tightened Smart Parking RA). Only one pairing produced genuine, substantive
RA-constraint violations: 3 IoT models forced onto the (tightened) Smart
Parking RA. This script now only runs that pair.

Requires OPENAI_API_KEY to be set in the environment before running.
"""

import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]

# Make the main pipeline modules importable without copying/modifying them.
sys.path.insert(0, str(REPO_ROOT))

# The pipeline modules use paths (e.g. "java/...") relative to the repo root.
os.chdir(REPO_ROOT)

from config_utils import load_ra_descriptions  # noqa: E402
from graph_utils import (  # noqa: E402
    load_mapping_file,
    get_graph_nodes,
    get_graph_edges,
    get_ra_components,
    get_ra_edge_set,
    get_ra_parent_map,
)
from llm_clients import build_llm_client  # noqa: E402
from mapping_core import run_option2_mapping  # noqa: E402
from semantic_utils import load_semantic_model  # noqa: E402

MISMATCHED_DIR = EXPERIMENT_DIR / "mismatched_inputs"
RESULTS_DIR = EXPERIMENT_DIR / "results"

PAIR_NAME = "iot_vs_smart_parking"
RA_ECORE_FILE = "smartparking.ecore"
RA_DESCRIPTION_FILE = "RA/smartparking_ra_components.txt"
THREAD_ID = 9003  # tags the weaving .model side-effect files as ours

# "RAMD" (no +MK) configuration -- same values as LLM_mapping_RA.yml's
# mapping: block, minus the Known Mappings DB (mapping_knowledge_index=None).
MODEL_CFG = {"name": "chatgpt", "provider": "openai", "model": "gpt-4o-mini", "temperature": 0}
API_KEYS_CFG = {"openai_env": "OPENAI_API_KEY"}

MAX_ITERATIONS = 30
STOP_IF_SAME = True
USE_RA_DESCRIPTIONS = True
USE_RA_STRUCTURAL_INFO = False
USE_CHUNKING = False
CHUNK_SIZE = 3
CHECK_RA_CONSTRAINTS = True
SAME_MAPPING_PATIENCE = 3
FEWSHOT_LEARNING = True
USE_SEMANTIC_FILTER = True
SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_TOP_K = 5
RECREATE_MAPPING_WITH_LLM = True
RECREATE_WINDOW = 10
BREAK_AFTER_RECREATE = True

WEAVING_DIR = REPO_ROOT / "java" / "src" / "it" / "uni" / "sim" / "architecturemodeling" / "launcher" / "models" / "weaving"


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_pair(llm, semantic_model) -> List[Dict[str, Any]]:
    input_dir = MISMATCHED_DIR / PAIR_NAME
    json_paths = sorted(glob.glob(str(input_dir / "*.json")))
    if not json_paths:
        raise FileNotFoundError(f"No mismatched input files found in {input_dir}")

    ra_descriptions = load_ra_descriptions(RA_DESCRIPTION_FILE) if USE_RA_DESCRIPTIONS else {}

    pair_results_dir = RESULTS_DIR / PAIR_NAME
    pair_weaving_dir = pair_results_dir / "weaving_models"
    pair_results_dir.mkdir(parents=True, exist_ok=True)

    pair_summaries = []

    for path in json_paths:
        file_name = os.path.basename(path)
        print(f"[{PAIR_NAME}] Processing {file_name} ...")

        mapping_data = load_mapping_file(path)
        graph_id = mapping_data["graph_id"]

        nodes = get_graph_nodes(mapping_data)
        edges = get_graph_edges(mapping_data)
        ra_components = get_ra_components(mapping_data)  # now the INJECTED (wrong) RA
        ra_edges = get_ra_edge_set(mapping_data)
        ra_parent = get_ra_parent_map(mapping_data)

        node_id_to_name = {n.id: n.name for n in nodes}
        ra_id_to_name = {c.id: c.name for c in ra_components}

        start = time.perf_counter()
        (
            predicted_mapping,
            check_constraints_log,
            avg_prompt_cost,
            total_llm_cost,
        ) = run_option2_mapping(
            THREAD_ID,
            llm,
            RA_ECORE_FILE,
            graph_id,
            nodes,
            edges,
            ra_components,
            ra_edges,
            MAX_ITERATIONS,
            STOP_IF_SAME,
            USE_RA_DESCRIPTIONS,
            USE_RA_STRUCTURAL_INFO,
            ra_descriptions,
            USE_CHUNKING,
            CHUNK_SIZE,
            USE_SEMANTIC_FILTER,
            semantic_model,
            SEMANTIC_TOP_K,
            ra_parent,
            check_RA_constraints=CHECK_RA_CONSTRAINTS,
            same_mapping_patience=SAME_MAPPING_PATIENCE,
            fewshot_learning=FEWSHOT_LEARNING,
            mapping_knowledge_index=None,
            mapping_knowledge_top_k=5,
            mapping_knowledge_weight=0.0,
            llm_model_name=MODEL_CFG["model"],
            recreate_mapping_with_llm=RECREATE_MAPPING_WITH_LLM,
            recreate_window=RECREATE_WINDOW,
            break_after_recreate=BREAK_AFTER_RECREATE,
        )
        execution_time = time.perf_counter() - start

        unmapped_count = sum(1 for v in predicted_mapping.values() if not v)
        compliant = "All constraints have been satisfied" in (check_constraints_log or "")

        predicted_mapping_by_name = {
            node_id_to_name.get(nid, nid): (ra_id_to_name.get(cid, cid) if cid else None)
            for nid, cid in predicted_mapping.items()
        }

        result = {
            "pair": PAIR_NAME,
            "graph_id": graph_id,
            "file": file_name,
            "injected_ra_domain": mapping_data.get("_mismatch_meta", {}).get("injected_ra_domain"),
            "source_domain": mapping_data.get("_mismatch_meta", {}).get("source_domain"),
            "predicted_mapping": predicted_mapping_by_name,
            "total_nodes": len(nodes),
            "unmapped_count": unmapped_count,
            "compliant": compliant,
            "check_constraints_log": check_constraints_log,
            "avg_prompt_cost": avg_prompt_cost,
            "total_llm_cost": total_llm_cost,
            "execution_time_seconds": round(execution_time, 3),
        }

        out_path = pair_results_dir / f"{os.path.splitext(file_name)[0]}__result.json"
        save_json(out_path, result)
        pair_summaries.append(result)

        print(
            f"[{PAIR_NAME}] {file_name}: compliant={compliant}, "
            f"unmapped={unmapped_count}/{len(nodes)}, "
            f"time={execution_time:.1f}s"
        )

    # Move (not copy) this run's weaving side-effect files out of the shared
    # pipeline directory into this experiment's own results folder, so no
    # extra output is left behind outside experiments/ra_incompatibility_test/.
    pair_weaving_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    if WEAVING_DIR.exists():
        for f in WEAVING_DIR.glob(f"*_thread_{THREAD_ID}.model"):
            shutil.move(str(f), str(pair_weaving_dir / f.name))
            moved += 1
    print(f"[{PAIR_NAME}] moved {moved} weaving model file(s) into {pair_weaving_dir}")

    save_json(RESULTS_DIR / f"{PAIR_NAME}__summary.json", pair_summaries)
    return pair_summaries


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it before running this script, e.g.:\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "  python3 experiments/ra_incompatibility_test/run_experiment.py"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    llm = build_llm_client(MODEL_CFG, API_KEYS_CFG)
    semantic_model = load_semantic_model(SEMANTIC_MODEL_NAME) if USE_SEMANTIC_FILTER else None

    run_pair(llm, semantic_model)

    print("Done. Per-file results under:", RESULTS_DIR)


if __name__ == "__main__":
    main()
