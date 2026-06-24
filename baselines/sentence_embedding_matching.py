import os
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import time

import numpy as np
from sentence_transformers import SentenceTransformer


def load_json(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_text(text: str, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def parse_ra_definitions(def_file: str) -> Dict[str, str]:
    """
    Parse file RA/iot_ra_components.txt, format:
    ComponentName: Definition text...

    """
    with open(def_file, "r", encoding="utf-8") as f:
        content = f.read().strip()

    blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
    definitions = {}

    for block in blocks:
        if ":" not in block:
            continue
        name, definition = block.split(":", 1)
        definitions[name.strip()] = definition.strip().replace("\n", " ")

    return definitions


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    return " ".join(str(text).replace("_", " ").replace("/", " ").split()).strip()


def build_node_text(node: Dict[str, Any]) -> str:
    name = normalize_text(node.get("name", ""))
    category = normalize_text(node.get("category", ""))
    classifier = normalize_text(node.get("classifier", ""))
    ports = node.get("ports", []) or []
    ports_text = ", ".join(normalize_text(p) for p in ports if p)

    parts = []
    if name:
        parts.append(f"Component name: {name}.")
    if category:
        parts.append(f"Category: {category}.")
    if classifier:
        parts.append(f"Classifier: {classifier}.")
    if ports_text:
        parts.append(f"Ports: {ports_text}.")

    return " ".join(parts).strip()


def build_ra_text(
    ra_comp: Dict[str, Any],
    ra_definitions: Dict[str, str],
    connectors: List[Dict[str, Any]],
    comp_id_to_name: Dict[str, str]
) -> str:
    ra_name = normalize_text(ra_comp.get("name", ""))
    ra_id = ra_comp.get("id", "")
    parent_name = normalize_text(ra_comp.get("parent_name", ""))
    tag = normalize_text(ra_comp.get("tag", ""))

    definition = normalize_text(ra_definitions.get(ra_name, ""))

    incoming_ids = normalize_text(ra_comp.get("incoming", ""))
    outgoing_ids = normalize_text(ra_comp.get("outgoing", ""))

    connector_name_map = {c["id"]: normalize_text(c.get("name", "")) for c in connectors}

    incoming_names = []
    if incoming_ids:
        for cid in incoming_ids.split():
            if cid in connector_name_map:
                incoming_names.append(connector_name_map[cid])

    outgoing_names = []
    if outgoing_ids:
        for cid in outgoing_ids.split():
            if cid in connector_name_map:
                outgoing_names.append(connector_name_map[cid])

    parts = []
    if ra_name:
        parts.append(f"RA component: {ra_name}.")
    if definition:
        parts.append(f"Definition: {definition}.")
    if tag:
        parts.append(f"Tag: {tag}.")
    # if parent_name:
    #     parts.append(f"Parent component: {parent_name}.")
    # if incoming_names:
    #     parts.append(f"Incoming connectors: {', '.join(incoming_names)}.")
    # if outgoing_names:
    #     parts.append(f"Outgoing connectors: {', '.join(outgoing_names)}.")

    return " ".join(parts).strip()


def get_ground_truth_ra_names(node: Dict[str, Any]) -> List[str]:
    gt = []
    for item in node.get("ra_mappings", []) or []:
        ra_name = item.get("ra_name")
        if ra_name:
            gt.append(ra_name)
    return gt


def rank_ra_components(
    model: SentenceTransformer,
    node_text: str,
    ra_texts: List[str],
    ra_components: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    node_emb = model.encode(node_text, convert_to_numpy=True, normalize_embeddings=True)
    ra_embs = model.encode(ra_texts, convert_to_numpy=True, normalize_embeddings=True)

    ranked = []
    for ra_comp, ra_text, ra_emb in zip(ra_components, ra_texts, ra_embs):
        score = float(np.dot(node_emb, ra_emb))  # do đã normalize
        ranked.append({
            "ra_id": ra_comp.get("id"),
            "ra_name": ra_comp.get("name"),
            "score": score,
            "ra_text": ra_text
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def evaluate_prediction(predicted_ra_name: str, gt_ra_names: List[str]) -> int:
    return int(predicted_ra_name in gt_ra_names)


def process_mapping_file(
    file_path: str,
    model: SentenceTransformer,
    ra_definitions: Dict[str, str],
    output_dir: str,
    top_k: int = 3
) -> Dict[str, Any]:
    data = load_json(file_path)

    graph_id = data.get("graph_id", Path(file_path).stem)
    ra_ref = data.get("ra_reference", {})
    ra_components = ra_ref.get("components", [])
    connectors = ra_ref.get("connectors", [])
    nodes = data.get("nodes", [])

    comp_id_to_name = {c.get("id"): c.get("name", "") for c in ra_components}

    ra_texts = [
        build_ra_text(rc, ra_definitions, connectors, comp_id_to_name)
        for rc in ra_components
    ]

    node_results = []
    correct_count = 0
    total_count = 0

    for node in nodes:
        node_text = build_node_text(node)
        gt_ra_names = get_ground_truth_ra_names(node)

        if not node_text.strip():
            node_text = normalize_text(node.get("name", ""))

        ranked = rank_ra_components(
            model=model,
            node_text=node_text,
            ra_texts=ra_texts,
            ra_components=ra_components
        )

        best = ranked[0]
        predicted_ra_name = best["ra_name"]
        predicted_ra_id = best["ra_id"]
        predicted_score = best["score"]

        is_correct = evaluate_prediction(predicted_ra_name, gt_ra_names)

        correct_count += is_correct
        total_count += 1

        node_results.append({
            "node_id": node.get("id"),
            "graph_node_id": node.get("graph_node_id"),
            "node_name": node.get("name"),
            "category": node.get("category"),
            "classifier": node.get("classifier"),
            "ports": node.get("ports", []),
            "node_text_used": node_text,
            "ground_truth_ra_names": gt_ra_names,
            "predicted_ra_name": predicted_ra_name,
            "predicted_ra_id": predicted_ra_id,
            "predicted_score": predicted_score,
            "is_correct": is_correct,
            "top_k_candidates": ranked[:top_k]
        })

    model_accuracy = (correct_count / total_count) if total_count > 0 else 0.0

    result = {
        "graph_id": graph_id,
        "source_file": os.path.basename(file_path),
        "method": "sentence_embedding_similarity_all-MiniLM-L6-v2",
        "num_nodes": total_count,
        "num_correct": correct_count,
        "average_score_for_model": model_accuracy,
        "predictions": node_results
    }

    out_file = os.path.join(output_dir, f"{Path(file_path).stem}_embedding_mapping.json")
    save_json(result, out_file)

    return {
        "graph_id": graph_id,
        "source_file": os.path.basename(file_path),
        "num_nodes": total_count,
        "num_correct": correct_count,
        "average_score_for_model": model_accuracy,
        "output_file": out_file
    }


def summarize_dataset(per_model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not per_model_results:
        return {
            "num_models": 0,
            "mean_of_model_average_scores": 0.0,
            "dataset_average_score": 0.0,
            "total_correct": 0,
            "total_mappings": 0
        }

    model_averages = [x["average_score_for_model"] for x in per_model_results]
    total_correct = sum(x["num_correct"] for x in per_model_results)
    total_mappings = sum(x["num_nodes"] for x in per_model_results)

    mean_of_model_average_scores = float(np.mean(model_averages)) if model_averages else 0.0
    dataset_average_score = (total_correct / total_mappings) if total_mappings > 0 else 0.0

    return {
        "num_models": len(per_model_results),
        "mean_of_model_average_scores": mean_of_model_average_scores,
        "dataset_average_score": dataset_average_score,
        "total_correct": total_correct,
        "total_mappings": total_mappings,
        "per_model_results": per_model_results
    }


def find_mapping_files(input_dir: str) -> List[str]:
    exts = {".json"}
    files = []
    for root, _, filenames in os.walk(input_dir):
        for fname in filenames:
            if Path(fname).suffix.lower() in exts:
                files.append(os.path.join(root, fname))
    return sorted(files)


def main():
    start_time = time.perf_counter()

    input_dir = "mapping_data/SMART_PARKING_mapping_files"
    ra_definitions = "RA/smartparking_ra_components.txt"
    output_dir = "smartparking_output_embedding_matching"

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    top_k = 1

    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Loading RA definitions from: {ra_definitions}")
    ra_definitions = parse_ra_definitions(ra_definitions)

    mapping_files = find_mapping_files(input_dir)
    print(f"Found {len(mapping_files)} mapping files.")

    per_model_results = []

    for idx, file_path in enumerate(mapping_files, start=1):
        print(f"[{idx}/{len(mapping_files)}] Processing: {file_path}")
        try:
            result = process_mapping_file(
                file_path=file_path,
                model=model,
                ra_definitions=ra_definitions,
                output_dir=output_dir,
                top_k=top_k
            )
            per_model_results.append(result)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    summary = summarize_dataset(per_model_results)
    end_time = time.perf_counter()
    runtime_seconds = end_time - start_time
    summary_path = os.path.join(output_dir, "summary_metrics.json")
    save_json(summary, summary_path)

    report_lines = []
    report_lines.append("Sentence embedding similarity")
    report_lines.append(f"Model: {model_name}")
    report_lines.append(f"Number of models: {summary['num_models']}")
    report_lines.append(f"Runtime: {runtime_seconds}")
    report_lines.append(f"Mean of model average scores: {summary['mean_of_model_average_scores']:.4f}")
    report_lines.append(f"Dataset average score: {summary['dataset_average_score']:.4f}")
    report_lines.append(f"Total correct mappings: {summary['total_correct']}")
    report_lines.append(f"Total mappings: {summary['total_mappings']}")
    report_lines.append("")
    report_lines.append("Per-model results:")
    for item in per_model_results:
        report_lines.append(
            f"- {item['graph_id']}: "
            f"{item['num_correct']}/{item['num_nodes']} = {item['average_score_for_model']:.4f}"
        )

    report_path = os.path.join(output_dir, "summary_report.txt")
    save_text("\n".join(report_lines), report_path)

    print("\nDone.")
    print(f"Summary saved to: {summary_path}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()