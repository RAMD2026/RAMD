import os
import glob
import json
import csv
import time
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from config_utils import load_config, load_ra_descriptions
from llm_clients import build_llm_client
from graph_utils import (
    load_mapping_file,
    get_graph_nodes,
    get_graph_edges,
    get_ra_components,
    get_ra_edge_set,
    get_ra_parent_map,
)
from mapping_core import run_option1_mapping, run_option2_mapping
from scoring import compute_scores_for_file
from text_utils import normalize_name
from semantic_utils import load_semantic_model, preprocess_mapping_knowledge_csv


def build_run_folder_name(
    option: int,
    check_ra_constraints: bool,
    fewshot_learning: bool,
    recreate_mapping_with_llm: bool,
    recreate_window: int,
    break_after_recreate: bool,
) -> str:
    name = f"Prompt{option}"

    if check_ra_constraints:
        name += "_constraints"

    if fewshot_learning:
        name += "_fewshot"

    if recreate_mapping_with_llm:
        name += f"_recreate{recreate_window}"
        if break_after_recreate:
            name += "_break"

    return name


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    if not rows:
        return

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_one_thread(
    thread_idx: int,
    json_paths: List[str],
    models_cfg: List[Dict[str, Any]],
    api_keys_cfg: Dict[str, Any],
    ra_ecore_file: str,
    ra_descriptions: Dict[str, Any],
    summary_config: Dict[str, Any],
    option: int,
    max_iterations: int,
    stop_if_same: bool,
    use_ra_descriptions: bool,
    use_ra_structural_info: bool,
    use_chunking: bool,
    chunk_size: int,
    use_semantic_filter: bool,
    semantic_model_name: str,
    semantic_top_k: int,
    check_RA_constraints: bool,
    same_mapping_patience: int,
    fewshot_learning: bool,
    recreate_mapping_with_llm: bool,
    recreate_window: int,
    break_after_recreate: bool,
    thread_output_dir: str,
    mapping_knowledge_csv_path: str,
    mapping_knowledge_top_k: int,
    mapping_knowledge_weight: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:

    thread_start_time = time.perf_counter()

    local_summary: List[Dict[str, Any]] = []
    local_csv_rows: List[Dict[str, Any]] = []
    thread_model_summary: List[Dict[str, Any]] = []

    semantic_model = None
    if use_semantic_filter:
        semantic_model = load_semantic_model(semantic_model_name)

    if not json_paths:
        print(f"[Thread {thread_idx}] No files to process.")
        return local_summary, local_csv_rows, thread_model_summary

    first_mapping_data = load_mapping_file(json_paths[0])
    shared_ra_components = get_ra_components(first_mapping_data)
    print("first_mapping_data: ", first_mapping_data)
    print("shared_ra_components: ", shared_ra_components)
    if use_semantic_filter: 
        mapping_knowledge_index = preprocess_mapping_knowledge_csv(
            csv_path=mapping_knowledge_csv_path,
            ra_components=shared_ra_components,
            model=semantic_model,
        )
    else:
        mapping_knowledge_index = None

    print(f"[Thread {thread_idx}] Started. Total files: {len(json_paths)}")
    print(f"[Thread {thread_idx}] Output folder: {thread_output_dir}")

    for model_cfg in models_cfg:
        model_start_time = time.perf_counter()

        model_name = model_cfg["name"]
        print(f"[Thread {thread_idx}] === Running model: {model_name} ===")

        llm = build_llm_client(model_cfg, api_keys_cfg)

        total_correct_all = 0.0
        total_nodes_all = 0
        total_avg_score = 0.0

        model_total_llm_cost = 0
        model_total_avg_prompt_cost = 0.0
        model_file_count = 0

        for path in json_paths:
            file_start_time = time.perf_counter()

            mapping_data = load_mapping_file(path)
            graph_id = mapping_data["graph_id"]
            file_name = os.path.basename(path)

            print(f"[Thread {thread_idx}] File: {file_name} | graph_id={graph_id}")

            nodes = get_graph_nodes(mapping_data)
            edges = get_graph_edges(mapping_data)

            ra_components = shared_ra_components
            ra_edges = get_ra_edge_set(mapping_data)
            ra_parent = get_ra_parent_map(mapping_data)

            node_id_to_name = {n.id: n.name for n in nodes}
            ra_id_to_name = {c.id: c.name for c in ra_components}

            if option == 1:
                (
                    predicted_mapping,
                    check_constraints_log,
                    avg_prompt_cost,
                    total_llm_cost,
                ) = run_option1_mapping(
                    thread_idx,
                    llm,
                    ra_ecore_file,
                    graph_id,
                    nodes,
                    edges,
                    ra_components,
                    ra_edges,
                    ra_parent,
                    max_iterations,
                    stop_if_same,
                    use_ra_descriptions,
                    use_ra_structural_info,
                    ra_descriptions,
                    check_RA_constraints,
                    same_mapping_patience,
                    fewshot_learning,
                    model_name,
                    recreate_mapping_with_llm,
                    recreate_window,
                    break_after_recreate,
                )

            elif option == 2:
                (
                    predicted_mapping,
                    check_constraints_log,
                    avg_prompt_cost,
                    total_llm_cost,
                ) = run_option2_mapping(
                    thread_idx,
                    llm,
                    ra_ecore_file,
                    graph_id,
                    nodes,
                    edges,
                    ra_components,
                    ra_edges,
                    max_iterations,
                    stop_if_same,
                    use_ra_descriptions,
                    use_ra_structural_info,
                    ra_descriptions,
                    use_chunking,
                    chunk_size,
                    use_semantic_filter,
                    semantic_model,
                    semantic_top_k,
                    ra_parent,
                    check_RA_constraints,
                    same_mapping_patience,
                    fewshot_learning,
                    mapping_knowledge_index,
                    mapping_knowledge_top_k,
                    mapping_knowledge_weight,
                    model_name,
                    recreate_mapping_with_llm,
                    recreate_window,
                    break_after_recreate,
                )

            else:
                raise ValueError(f"Unsupported mapping option: {option}")

            total_score, avg_score, per_node_scores = compute_scores_for_file(
                mapping_data,
                predicted_mapping,
            )

            file_execution_time = time.perf_counter() - file_start_time

            node_count = len(per_node_scores)
            total_correct_all += total_score
            total_nodes_all += node_count
            total_avg_score += avg_score

            model_total_llm_cost += total_llm_cost
            model_total_avg_prompt_cost += avg_prompt_cost
            model_file_count += 1

            predicted_mapping_by_name = {}
            per_node_scores_by_name = {}

            for nid, ra_cid in predicted_mapping.items():
                raw_node_name = node_id_to_name.get(nid, nid)
                norm_node_name = normalize_name(raw_node_name)

                ra_name = ""
                if ra_cid:
                    ra_name = ra_id_to_name.get(ra_cid, ra_cid)

                predicted_mapping_by_name[norm_node_name] = ra_name
                per_node_scores_by_name[norm_node_name] = per_node_scores.get(nid, 0.0)

            result = {
                "thread": thread_idx,
                "model": model_name,
                "option": option,
                "graph_id": graph_id,
                "file": file_name,
                "predicted_mapping": predicted_mapping_by_name,
                "total_score": total_score,
                "avg_score": avg_score,
                "per_node_scores": per_node_scores_by_name,
                "check_constraints_log": check_constraints_log,
                "avg_prompt_cost": avg_prompt_cost,
                "total_llm_cost": total_llm_cost,
                "execution_time_seconds": round(file_execution_time, 6),
                "config": summary_config,
            }

            out_name = (
                f"{os.path.splitext(file_name)[0]}__{model_name}__opt{option}.json"
            )
            out_path = os.path.join(thread_output_dir, out_name)
            save_json(out_path, result)

            local_summary.append(
                {
                    "thread": thread_idx,
                    "model": model_name,
                    "option": option,
                    "graph_id": graph_id,
                    "file": file_name,
                    "total_score": total_score,
                    "avg_score": avg_score,
                    "avg_prompt_cost": avg_prompt_cost,
                    "total_llm_cost": total_llm_cost,
                    "execution_time_seconds": round(file_execution_time, 6),
                    "config": summary_config,
                }
            )

            for nid, ra_cid in predicted_mapping.items():
                local_csv_rows.append(
                    {
                        "thread": thread_idx,
                        "model": model_name,
                        "option": option,
                        "file": file_name,
                        "graph_id": graph_id,
                        "node_id": nid,
                        "node_name": node_id_to_name.get(nid, ""),
                        "predicted_ra_component_id": ra_cid or "",
                        "predicted_ra_component_name": (
                            ra_id_to_name.get(ra_cid, "") if ra_cid else ""
                        ),
                        "node_score": per_node_scores.get(nid, 0.0),
                        "check_constraints_log": check_constraints_log,
                        "avg_prompt_cost": avg_prompt_cost,
                        "total_llm_cost": total_llm_cost,
                    }
                )

        model_execution_time = time.perf_counter() - model_start_time

        model_avg_over_nodes = (
            total_correct_all / total_nodes_all if total_nodes_all > 0 else 0.0
        )
        model_avg_over_files = (
            total_avg_score / len(json_paths) if json_paths else 0.0
        )

        model_avg_prompt_cost = (
            model_total_avg_prompt_cost / model_file_count
            if model_file_count > 0
            else 0.0
        )

        thread_model_summary.append(
            {
                "thread": thread_idx,
                "model": model_name,
                "option": option,
                "graph_id": "__ALL__",
                "file": "__ALL__",
                "total_score": total_correct_all,
                "micro_score": model_avg_over_nodes,
                "macro_score": model_avg_over_files,
                "avg_prompt_cost": model_avg_prompt_cost,
                "total_llm_cost": model_total_llm_cost,
                "execution_time_seconds": round(model_execution_time, 6),
                "config": summary_config,
            }
        )

        print(
            f"[Thread {thread_idx}] ==> Model {model_name} average over nodes "
            f"(micro score): {model_avg_over_nodes:.4f}"
        )
        print(
            f"[Thread {thread_idx}] ==> Model {model_name} average over files "
            f"(macro score): {model_avg_over_files:.4f}"
        )
        print(
            f"[Thread {thread_idx}] ==> Model {model_name} average prompt cost: "
            f"{model_avg_prompt_cost:.4f}"
        )
        print(
            f"[Thread {thread_idx}] ==> Model {model_name} total LLM cost: "
            f"{model_total_llm_cost}"
        )
        print(
            f"[Thread {thread_idx}] ==> Model {model_name} execution time: "
            f"{model_execution_time:.6f} seconds"
        )

    thread_execution_time = time.perf_counter() - thread_start_time

    save_json(
        os.path.join(thread_output_dir, f"summary_thread{thread_idx}.json"),
        local_summary + thread_model_summary,
    )

    csv_fieldnames = [
        "thread",
        "model",
        "option",
        "file",
        "graph_id",
        "node_id",
        "node_name",
        "predicted_ra_component_id",
        "predicted_ra_component_name",
        "node_score",
        "check_constraints_log",
        "avg_prompt_cost",
        "total_llm_cost",
    ]

    save_csv(
        os.path.join(thread_output_dir, f"mappings_thread{thread_idx}.csv"),
        local_csv_rows,
        csv_fieldnames,
    )

    print(
        f"[Thread {thread_idx}] Finished. Total execution time: "
        f"{thread_execution_time:.6f} seconds"
    )

    return local_summary, local_csv_rows, thread_model_summary


def main():
    overall_start_time = time.perf_counter()

    config = load_config("LLM_mapping_RA.yml")

    data_cfg = config["data"]
    mapping_cfg = config["mapping"]
    models_cfg = config["models"]
    api_keys_cfg = config["api_keys"]

    check_RA_constraints = bool(mapping_cfg.get("check_RA_constraints", False))
    same_mapping_patience = int(mapping_cfg.get("same_mapping_patience", 1))
    fewshot_learning = bool(mapping_cfg.get("fewshot_learning", False))

    recreate_mapping_with_llm = bool(
        mapping_cfg.get("recreate_mapping_with_llm", False)
    )
    recreate_window = int(mapping_cfg.get("recreate_window", 3))
    break_after_recreate = bool(mapping_cfg.get("break_after_recreate", False))

    use_ra_descriptions = bool(mapping_cfg.get("use_ra_descriptions", False))
    use_ra_structural_info = bool(mapping_cfg.get("use_ra_structural_info", False))
    use_chunking = bool(mapping_cfg.get("use_chunking", False))
    chunk_size = int(mapping_cfg.get("chunk_size", 0))

    use_semantic_filter = bool(mapping_cfg.get("use_semantic_filter", False))
    semantic_model_name = mapping_cfg.get(
        "semantic_model_name",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    semantic_top_k = int(mapping_cfg.get("semantic_top_k", 5))

    mapping_knowledge_csv_path = mapping_cfg.get("mapping_knowledge_csv_path", "")
    mapping_knowledge_top_k = int(mapping_cfg.get("mapping_knowledge_top_k", 5))
    mapping_knowledge_weight = float(mapping_cfg.get("mapping_knowledge_weight", 0.7))

    ra_ecore_file = data_cfg.get("ra_ecore_file", "")
    ra_desc_file = data_cfg.get("ra_description_file", "")
    ra_descriptions = load_ra_descriptions(ra_desc_file) if use_ra_descriptions else {}

    input_folder = data_cfg["input_folder"]
    file_glob = data_cfg.get("file_glob", "*.json")
    base_output_folder = data_cfg.get("output_folder", "results")

    option = int(mapping_cfg.get("option", 1))
    max_iterations = int(mapping_cfg.get("max_iterations", 5))
    stop_if_same = bool(mapping_cfg.get("stop_if_same", True))
    num_threads = int(mapping_cfg.get("num_threads", 1))

    execution_mode = str(mapping_cfg.get("execution_mode", "parallel")).strip().lower()

    if execution_mode not in {"parallel", "sequential"}:
        raise ValueError("execution_mode must be either 'parallel' or 'sequential'")

    if num_threads < 1:
        raise ValueError("num_threads must be >= 1")

    if recreate_window < 1:
        raise ValueError("recreate_window must be >= 1")

    summary_config = {
        "use_ra_descriptions": use_ra_descriptions,
        "use_ra_structural_info": use_ra_structural_info,
        "check_RA_constraints": check_RA_constraints,
        "same_mapping_patience": same_mapping_patience,
        "fewshot_learning": fewshot_learning,
        "recreate_mapping_with_llm": recreate_mapping_with_llm,
        "recreate_window": recreate_window,
        "break_after_recreate": break_after_recreate,
        "use_chunking": use_chunking,
        "chunk_size": chunk_size,
        "use_semantic_filter": use_semantic_filter,
        "semantic_model_name": semantic_model_name,
        "semantic_top_k": semantic_top_k,
        "mapping_knowledge_csv_path": mapping_knowledge_csv_path,
        "mapping_knowledge_top_k": mapping_knowledge_top_k,
        "mapping_knowledge_weight": mapping_knowledge_weight,
        "num_threads": num_threads,
        "execution_mode": execution_mode,
    }

    run_folder_name = build_run_folder_name(
        option=option,
        check_ra_constraints=check_RA_constraints,
        fewshot_learning=fewshot_learning,
        recreate_mapping_with_llm=recreate_mapping_with_llm,
        recreate_window=recreate_window,
        break_after_recreate=break_after_recreate,
    )

    output_folder = os.path.join(base_output_folder, run_folder_name)
    os.makedirs(output_folder, exist_ok=True)

    thread_folders = []

    for i in range(num_threads):
        thread_dir = os.path.join(output_folder, f"thread{i + 1}")
        os.makedirs(thread_dir, exist_ok=True)
        thread_folders.append(thread_dir)

    json_paths = sorted(glob.glob(os.path.join(input_folder, file_glob)))

    if not json_paths:
        print(f"No JSON files found in {input_folder} matching {file_glob}")
        return

    print(f"Run folder: {output_folder}")
    print(f"Execution mode: {execution_mode}")
    print(f"Number of threads: {num_threads}")
    print(f"Each thread will process all {len(json_paths)} file(s).")
    print(f"Recreate mapping with LLM: {recreate_mapping_with_llm}")
    print(f"Recreate window: {recreate_window}")
    print(f"Break after recreate: {break_after_recreate}")

    all_thread_model_summaries: List[Dict[str, Any]] = []

    if execution_mode == "parallel":
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []

            for i in range(num_threads):
                futures.append(
                    executor.submit(
                        process_one_thread,
                        i + 1,
                        json_paths,
                        models_cfg,
                        api_keys_cfg,
                        ra_ecore_file,
                        ra_descriptions,
                        summary_config,
                        option,
                        max_iterations,
                        stop_if_same,
                        use_ra_descriptions,
                        use_ra_structural_info,
                        use_chunking,
                        chunk_size,
                        use_semantic_filter,
                        semantic_model_name,
                        semantic_top_k,
                        check_RA_constraints,
                        same_mapping_patience,
                        fewshot_learning,
                        recreate_mapping_with_llm,
                        recreate_window,
                        break_after_recreate,
                        thread_folders[i],
                        mapping_knowledge_csv_path,
                        mapping_knowledge_top_k,
                        mapping_knowledge_weight,
                    )
                )

            for future in as_completed(futures):
                result = future.result()
                thread_model_summary = result[2]
                all_thread_model_summaries.extend(thread_model_summary)

    else:
        for i in range(num_threads):
            result = process_one_thread(
                i + 1,
                json_paths,
                models_cfg,
                api_keys_cfg,
                ra_ecore_file,
                ra_descriptions,
                summary_config,
                option,
                max_iterations,
                stop_if_same,
                use_ra_descriptions,
                use_ra_structural_info,
                use_chunking,
                chunk_size,
                use_semantic_filter,
                semantic_model_name,
                semantic_top_k,
                check_RA_constraints,
                same_mapping_patience,
                fewshot_learning,
                recreate_mapping_with_llm,
                recreate_window,
                break_after_recreate,
                thread_folders[i],
                mapping_knowledge_csv_path,
                mapping_knowledge_top_k,
                mapping_knowledge_weight,
            )

            thread_model_summary = result[2]
            all_thread_model_summaries.extend(thread_model_summary)

    avg_across_threads = []

    for model_cfg in models_cfg:
        model_name = model_cfg["name"]

        model_rows = [
            row
            for row in all_thread_model_summaries
            if row["model"] == model_name and row["graph_id"] == "__ALL__"
        ]

        if model_rows:
            avg_micro_score_across_threads = sum(
                r["micro_score"] for r in model_rows
            ) / len(model_rows)

            avg_macro_score_across_threads = sum(
                r["macro_score"] for r in model_rows
            ) / len(model_rows)

            avg_prompt_cost_threads = sum(
                r["avg_prompt_cost"] for r in model_rows
            ) / len(model_rows)

            total_llm_cost_threads = sum(
                r["total_llm_cost"] for r in model_rows
            ) / len(model_rows)

            avg_time_across_threads = sum(
                r["execution_time_seconds"] for r in model_rows
            ) / len(model_rows)

        else:
            avg_micro_score_across_threads = 0.0
            avg_macro_score_across_threads = 0.0
            avg_prompt_cost_threads = 0.0
            total_llm_cost_threads = 0.0
            avg_time_across_threads = 0.0

        avg_across_threads.append(
            {
                "thread": "__AVG_THREADS__",
                "model": model_name,
                "option": option,
                "graph_id": "__ALL__",
                "file": "__ALL__",
                "total_score": None,
                "avg_micro_score": round(avg_micro_score_across_threads, 6),
                "avg_macro_score": round(avg_macro_score_across_threads, 6),
                "avg_prompt_cost": round(avg_prompt_cost_threads, 6),
                "total_llm_cost": round(total_llm_cost_threads, 6),
                "execution_time_seconds": round(avg_time_across_threads, 6),
                "config": summary_config,
            }
        )

    overall_execution_time = time.perf_counter() - overall_start_time

    final_summary = []
    final_summary.extend(avg_across_threads)

    summary_path = os.path.join(output_folder, "summary_all_threads.json")
    save_json(summary_path, final_summary)

    print(f"Done. Summary JSON: {summary_path}")
    print(f"Done. Overall execution time: {overall_execution_time:.6f} seconds")


if __name__ == "__main__":
    main()