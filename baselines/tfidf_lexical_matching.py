import os
import re
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)

    # parse snake_case / ALL_CAPS / CamelCase
    text = text.replace("_", " ")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[^a-zA-Z0-9\s/#.-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def parse_ra_definitions(ra_txt_path: str) -> Dict[str, str]:
    """
    Parse file RA definitions:
    Application: Provides ...
    
    IoTIM: Acts as ...
    
    ...
    """
    with open(ra_txt_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    blocks = re.split(r"\n\s*\n", content)
    ra_defs = {}

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        m = re.match(r"^\s*([^:]+)\s*:\s*(.+)$", block, flags=re.DOTALL)
        if m:
            name = m.group(1).strip()
            definition = m.group(2).strip().replace("\n", " ")
            ra_defs[name] = definition

    if not ra_defs:
        raise ValueError(f"Cannot parse RA definitions from file: {ra_txt_path}")

    return ra_defs


def build_ra_texts(mapping_data: Dict[str, Any], ra_defs: Dict[str, str]) -> List[Dict[str, Any]]:
    ra_components = mapping_data.get("ra_reference", {}).get("components", [])
    results = []

    for comp in ra_components:
        ra_name = comp.get("name", "").strip()
        ra_id = comp.get("id", "").strip()
        parent_name = comp.get("parent_name") or ""
        tag = comp.get("tag") or ""
        mandatory = comp.get("mandatory")
        definition = ra_defs.get(ra_name, "")

        text_parts = [
            ra_name,
            f"definition {definition}" if definition else "",
            f"parent {parent_name}" if parent_name else "",
            f"tag {tag}" if tag else "",
            f"mandatory {mandatory}" if mandatory is not None else "",
        ]
        ra_text = normalize_text(" ".join([p for p in text_parts if p]))

        results.append({
            "ra_id": ra_id,
            "ra_name": ra_name,
            "ra_text": ra_text,
            "definition": definition,
            "parent_name": parent_name,
            "tag": tag,
        })

    return results


def build_node_text(node: Dict[str, Any]) -> str:
    name = node.get("name", "")
    category = node.get("category", "")
    classifier = node.get("classifier", "")
    ports = node.get("ports", []) or []

    classifier_text = classifier.replace("/", " ").replace("#", " ").replace(".", " ")

    text_parts = [
        name,
        f"category {category}" if category else "",
        f"classifier {classifier_text}" if classifier_text else "",
        f"ports {' '.join(ports)}" if ports else "",
    ]

    return normalize_text(" ".join([p for p in text_parts if p]))


def predict_node_ra(
    node_text: str,
    ra_items: List[Dict[str, Any]],
    top_k: int = 3
) -> List[Tuple[str, str, float]]:
    """
    TF-IDF + cosine similarity between 1 node and all RA components.
    Return top_k: [(ra_id, ra_name, score), ...]
    """
    documents = [node_text] + [item["ra_text"] for item in ra_items]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        lowercase=True
    )
    tfidf_matrix = vectorizer.fit_transform(documents)

    node_vec = tfidf_matrix[0:1]
    ra_vecs = tfidf_matrix[1:]

    sims = cosine_similarity(node_vec, ra_vecs)[0]

    ranked = sorted(
        zip(ra_items, sims),
        key=lambda x: x[1],
        reverse=True
    )

    results = []
    for item, score in ranked[:top_k]:
        results.append((item["ra_id"], item["ra_name"], float(score)))

    return results


def evaluate_prediction(pred_ra_name: str, node: Dict[str, Any]) -> Tuple[int, List[str]]:
    gt_mappings = node.get("ra_mappings", []) or []
    gt_names = [m.get("ra_name", "").strip() for m in gt_mappings if m.get("ra_name")]

    is_correct = 1 if pred_ra_name in gt_names else 0
    return is_correct, gt_names


def process_mapping_file(
    file_path: str,
    ra_defs: Dict[str, str],
    top_k: int = 3
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    graph_id = data.get("graph_id", Path(file_path).stem)
    nodes = data.get("nodes", []) or []

    ra_items = build_ra_texts(data, ra_defs)

    node_records = []
    total_nodes = 0
    correct_nodes = 0

    for node in nodes:
        node_id = node.get("id", "")
        node_name = node.get("name", "")
        node_category = node.get("category", "")
        node_text = build_node_text(node)

        top_matches = predict_node_ra(node_text, ra_items, top_k=top_k)
        pred_ra_id, pred_ra_name, pred_score = top_matches[0]

        is_correct, gt_names = evaluate_prediction(pred_ra_name, node)

        total_nodes += 1
        correct_nodes += is_correct

        node_records.append({
            "graph_id": graph_id,
            "node_id": node_id,
            "node_name": node_name,
            "node_category": node_category,
            "node_text": node_text,
            "predicted_ra_id": pred_ra_id,
            "predicted_ra_name": pred_ra_name,
            "predicted_score": pred_score,
            "ground_truth_ra_names": "|".join(gt_names),
            "correct": is_correct,
            "top_k_matches": json.dumps(
                [
                    {"ra_id": rid, "ra_name": rname, "score": score}
                    for rid, rname, score in top_matches
                ],
                ensure_ascii=False
            )
        })

    accuracy = correct_nodes / total_nodes if total_nodes > 0 else 0.0

    model_summary = {
        "graph_id": graph_id,
        "num_nodes": total_nodes,
        "num_correct": correct_nodes,
        "accuracy": accuracy,
        "file_path": file_path
    }

    return node_records, model_summary


def run_experiment(mapping_folder: str, ra_txt_path: str, output_dir: str, top_k: int = 3):
    start_time = time.perf_counter()
    ra_defs = parse_ra_definitions(ra_txt_path)

    mapping_files = sorted([
        str(p) for p in Path(mapping_folder).glob("*.json")
    ])

    if not mapping_files:
        raise FileNotFoundError(f"Cannot find any files .json in the folder: {mapping_folder}")

    all_node_records = []
    all_model_summaries = []

    for fp in mapping_files:
        try:
            node_records, model_summary = process_mapping_file(fp, ra_defs, top_k=top_k)
            all_node_records.extend(node_records)
            all_model_summaries.append(model_summary)
            print(f"[OK] {model_summary['graph_id']} | accuracy={model_summary['accuracy']:.4f}")
        except Exception as e:
            print(f"[ERROR] File {fp}: {e}")

    if not all_node_records:
        raise RuntimeError("ERROR")

    node_df = pd.DataFrame(all_node_records)
    model_df = pd.DataFrame(all_model_summaries)

    # ===== overall accuracy (micro) =====
    total_correct = int(node_df["correct"].sum())
    total_nodes = int(len(node_df))
    overall_accuracy = total_correct / total_nodes if total_nodes > 0 else 0.0

    # ===== mean accuracy per model (macro) =====
    mean_model_accuracy = model_df["accuracy"].mean()

    os.makedirs(output_dir, exist_ok=True)

    node_csv = os.path.join(output_dir, "node_mappings.csv")
    model_csv = os.path.join(output_dir, "model_accuracy.csv")
    summary_json = os.path.join(output_dir, "summary.json")

    node_df.to_csv(node_csv, index=False, encoding="utf-8-sig")
    model_df.to_csv(model_csv, index=False, encoding="utf-8-sig")
    end_time = time.perf_counter()
    runtime_seconds = end_time - start_time
    summary = {
        "mapping_folder": mapping_folder,
        "ra_txt_path": ra_txt_path,

        "num_models": int(len(model_df)),

        # micro accuracy
        "total_nodes": total_nodes,
        "total_correct": total_correct,
        "overall_accuracy_micro": overall_accuracy,

        # macro accuracy (NEW)
        "mean_accuracy_per_model": float(mean_model_accuracy),

        "output_files": {
            "node_mappings.csv": node_csv,
            "model_accuracy.csv": model_csv,
            "summary.json": summary_json
        },
        
        "runtime_seconds": runtime_seconds

    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== FINAL SUMMARY =====")
    print(f"Num models              : {summary['num_models']}")
    print(f"Total nodes             : {total_nodes}")
    print(f"Total correct           : {total_correct}")
    print(f"Overall accuracy (micro): {overall_accuracy:.4f}")
    print(f"Mean accuracy (macro)   : {mean_model_accuracy:.4f}")


def main():

    mapping_folder = "mapping_data/SMART_PARKING_mapping_files"
    ra_txt_path = "RA/smartparking_ra_components.txt"
    output_dir = "smartparking_tfidf_ra_matching"
    top_k = 1

    run_experiment(
        mapping_folder,
        ra_txt_path,
        output_dir,
        top_k
    )


if __name__ == "__main__":
    main()