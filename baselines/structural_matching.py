import os
import json
import time
import argparse
from collections import defaultdict

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def safe_div(a, b):
    return a / b if b != 0 else 0.0

def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

def extract_filename_stem(path):
    return os.path.splitext(os.path.basename(path))[0]

# RA graph
def build_ra_index(ra_reference):
    components = ra_reference.get("components", [])
    connectors = ra_reference.get("connectors", [])

    comp_by_id = {}
    comp_ids = []
    neighbors = defaultdict(set)
    parent_children = defaultdict(list)

    for comp in components:
        cid = comp["id"]
        comp_by_id[cid] = comp
        comp_ids.append(cid)

    for comp in components:
        pid = comp.get("parent_id")
        if pid:
            parent_children[pid].append(comp["id"])

    for conn in connectors:
        src = conn.get("source")
        tgt = conn.get("target")
        if src and tgt:
            neighbors[src].add(tgt)
            neighbors[tgt].add(src)

    features = {}
    for cid, comp in comp_by_id.items():
        deg = len(neighbors[cid])
        features[cid] = {
            "id": cid,
            "name": comp.get("name", ""),
            "tag": comp.get("tag"),
            "mandatory": comp.get("mandatory"),
            "parent_id": comp.get("parent_id"),
            "parent_name": comp.get("parent_name"),
            "degree": deg,
            "is_leaf": deg <= 1 and len(parent_children[cid]) == 0,
            "num_children": len(parent_children[cid]),
            "neighbors": set(neighbors[cid]),
        }

    return {
        "components": components,
        "connectors": connectors,
        "comp_by_id": comp_by_id,
        "comp_ids": comp_ids,
        "neighbors": neighbors,
        "parent_children": parent_children,
        "features": features,
    }


# AADL graph
def build_aadl_index(data):
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    node_by_id = {n["id"]: n for n in nodes}
    node_ids = [n["id"] for n in nodes]

    out_neighbors = defaultdict(set)
    in_neighbors = defaultdict(set)
    undirected_neighbors = defaultdict(set)

    for e in edges:
        s = e.get("source")
        t = e.get("target")
        if s in node_by_id and t in node_by_id:
            out_neighbors[s].add(t)
            in_neighbors[t].add(s)
            undirected_neighbors[s].add(t)
            undirected_neighbors[t].add(s)

    features = {}
    for node in nodes:
        nid = node["id"]
        category = (node.get("category") or "").lower()
        degree = len(undirected_neighbors[nid])

        features[nid] = {
            "id": nid,
            "name": node.get("name", ""),
            "category": category,
            "ports_count": len(node.get("ports", []) or []),
            "in_degree": len(in_neighbors[nid]),
            "out_degree": len(out_neighbors[nid]),
            "degree": degree,
            "is_leaf": degree <= 1,
            "is_root_like": category == "system" or (node.get("name", "").strip().lower() == "system"),
            "neighbors": set(undirected_neighbors[nid]),
        }

    return {
        "nodes": nodes,
        "edges": edges,
        "node_by_id": node_by_id,
        "node_ids": node_ids,
        "out_neighbors": out_neighbors,
        "in_neighbors": in_neighbors,
        "neighbors": undirected_neighbors,
        "features": features,
    }


# Scoring
def degree_similarity(a, b, max_deg):
    if max_deg <= 0:
        return 1.0
    return 1.0 - abs(a - b) / max_deg


def port_count_compatibility(aadl_ports, ra_degree, ra_is_leaf):
    """
    Very weak structural cue.
    No semantic assumption, only that peripheral nodes often expose fewer ports.
    """
    if ra_is_leaf:
        if aadl_ports <= 2:
            return 1.0
        if aadl_ports <= 4:
            return 0.7
        return 0.4
    else:
        if aadl_ports >= 2:
            return 0.8
        return 0.5


def root_leaf_compatibility(aadl_feat, ra_feat):
    score = 0.5

    if aadl_feat["is_root_like"]:
        if ra_feat["degree"] >= 2:
            score += 0.25
        if ra_feat["is_leaf"]:
            score -= 0.20

    if aadl_feat["is_leaf"]:
        if ra_feat["is_leaf"]:
            score += 0.25
        if ra_feat["degree"] >= 2:
            score -= 0.10

    return clamp(score)


def child_relation_compatibility(aadl_feat, ra_feat):
    """
    If RA component is a child element, prefer structurally peripheral AADL nodes.
    """
    score = 0.5
    if ra_feat.get("parent_id"):
        if aadl_feat["degree"] <= 2:
            score += 0.20
        if aadl_feat["is_leaf"]:
            score += 0.15
    else:
        if aadl_feat["degree"] >= 2:
            score += 0.10

    return clamp(score)


def initial_pair_score(aadl_feat, ra_feat, aadl_max_deg, ra_max_deg):
    deg_sim = degree_similarity(aadl_feat["degree"], ra_feat["degree"], max(aadl_max_deg, ra_max_deg, 1))
    port_sim = port_count_compatibility(aadl_feat["ports_count"], ra_feat["degree"], ra_feat["is_leaf"])
    rl_sim = root_leaf_compatibility(aadl_feat, ra_feat)
    child_sim = child_relation_compatibility(aadl_feat, ra_feat)

    total = (
        0.40 * deg_sim +
        0.20 * port_sim +
        0.20 * rl_sim +
        0.20 * child_sim
    )
    return clamp(total)


def neighborhood_consistency_score(nid, rid, aadl_index, ra_index, current_scores):
    """
    Reward (AADL node -> RA component) if the neighbors of the node
    can also be matched well to neighbors of the RA component.
    """
    aadl_neighbors = aadl_index["features"][nid]["neighbors"]
    ra_neighbors = ra_index["features"][rid]["neighbors"]

    if not aadl_neighbors:
        return 0.0

    per_neighbor_scores = []
    for aadl_nb in aadl_neighbors:
        best = 0.0
        if ra_neighbors:
            for ra_nb in ra_neighbors:
                best = max(best, current_scores[aadl_nb].get(ra_nb, 0.0))
        per_neighbor_scores.append(best)

    neighbor_score = sum(per_neighbor_scores) / len(per_neighbor_scores) if per_neighbor_scores else 0.0

    parent_bonus = 0.0
    parent_id = ra_index["features"][rid].get("parent_id")
    if parent_id:
        best_parent_match = 0.0
        for aadl_nb in aadl_neighbors:
            best_parent_match = max(best_parent_match, current_scores[aadl_nb].get(parent_id, 0.0))
        parent_bonus = 0.5 * best_parent_match

    return clamp(0.75 * neighbor_score + 0.25 * parent_bonus)


def iterative_refinement(aadl_index, ra_index, base_scores, num_iters=4, alpha=0.70):
    """
    updated_score = alpha * base_score + (1 - alpha) * neighborhood_consistency
    """
    current = {nid: dict(scores) for nid, scores in base_scores.items()}

    for _ in range(num_iters):
        updated = {}
        for nid in aadl_index["node_ids"]:
            updated[nid] = {}
            for rid in ra_index["comp_ids"]:
                base = base_scores[nid][rid]
                neigh = neighborhood_consistency_score(nid, rid, aadl_index, ra_index, current)
                updated[nid][rid] = clamp(alpha * base + (1.0 - alpha) * neigh)
        current = updated

    return current

def predict_mapping_for_model(data):
    ra_index = build_ra_index(data["ra_reference"])
    aadl_index = build_aadl_index(data)

    aadl_max_deg = max([aadl_index["features"][nid]["degree"] for nid in aadl_index["node_ids"]] + [1])
    ra_max_deg = max([ra_index["features"][rid]["degree"] for rid in ra_index["comp_ids"]] + [1])

    base_scores = {}
    for nid in aadl_index["node_ids"]:
        aadl_feat = aadl_index["features"][nid]
        base_scores[nid] = {}
        for rid in ra_index["comp_ids"]:
            ra_feat = ra_index["features"][rid]
            base_scores[nid][rid] = initial_pair_score(aadl_feat, ra_feat, aadl_max_deg, ra_max_deg)

    refined_scores = iterative_refinement(
        aadl_index=aadl_index,
        ra_index=ra_index,
        base_scores=base_scores,
        num_iters=4,
        alpha=0.70
    )

    predictions = []
    correct = 0
    total = 0

    for node in data["nodes"]:
        nid = node["id"]
        ranked = sorted(refined_scores[nid].items(), key=lambda x: x[1], reverse=True)

        best_rid, best_score = ranked[0]
        best_ra = ra_index["comp_by_id"][best_rid]

        gt_ids = set()
        gt_names = set()
        for m in node.get("ra_mappings", []) or []:
            if m.get("ra_id"):
                gt_ids.add(m["ra_id"])
            if m.get("ra_name"):
                gt_names.add(m["ra_name"])

        is_correct = (best_rid in gt_ids) or (best_ra.get("name") in gt_names)

        if gt_ids or gt_names:
            total += 1
            if is_correct:
                correct += 1

        top_5 = []
        for rid, sc in ranked[:5]:
            top_5.append({
                "ra_id": rid,
                "ra_name": ra_index["comp_by_id"][rid].get("name"),
                "score": round(float(sc), 6)
            })

        predictions.append({
            "node_id": nid,
            "graph_node_id": node.get("graph_node_id"),
            "node_name": node.get("name"),
            "category": node.get("category"),
            "classifier": node.get("classifier"),
            "predicted_ra_id": best_rid,
            "predicted_ra_name": best_ra.get("name"),
            "prediction_score": round(float(best_score), 6),
            "top_5_candidates": top_5,
            "ground_truth": node.get("ra_mappings", []),
            "is_correct": is_correct if (gt_ids or gt_names) else None
        })

    per_model_accuracy = safe_div(correct, total)

    return {
        "graph_id": data.get("graph_id"),
        "ra_name": data.get("ra_reference", {}).get("name"),
        "num_nodes": len(data.get("nodes", [])),
        "num_edges": len(data.get("edges", [])),
        "evaluated_nodes": total,
        "correct_predictions": correct,
        "accuracy": per_model_accuracy,
        "predictions": predictions
    }

def find_json_files(input_dir):
    files = []
    for name in os.listdir(input_dir):
        path = os.path.join(input_dir, name)
        if os.path.isfile(path) and name.lower().endswith(".json"):
            files.append(path)
    files.sort()
    return files


def main():

    input_dir = "mapping_data/SMART_PARKING_mapping_files"
    output_dir = "smartparking_output_structural_matching"

    os.makedirs(output_dir, exist_ok=True)
    
    per_model_dir = os.path.join(output_dir, "per_model_mappings")
    os.makedirs(per_model_dir, exist_ok=True)
    input_files = find_json_files(input_dir)
    if not input_files:
        raise ValueError(f"No JSON files found in: {input_dir}")

    total_correct = 0
    total_evaluated = 0
    per_model_accuracies = []
    per_model_results = []

    t_all_start = time.perf_counter()

    for path in input_files:
        t_start = time.perf_counter()

        data = load_json(path)
        result = predict_mapping_for_model(data)

        runtime_sec = time.perf_counter() - t_start
        result["runtime_seconds"] = round(runtime_sec, 6)

        out_name = extract_filename_stem(path) + "_mapping.json"
        out_path = os.path.join(per_model_dir, out_name)
        save_json(result, out_path)

        total_correct += result["correct_predictions"]
        total_evaluated += result["evaluated_nodes"]
        per_model_accuracies.append(result["accuracy"])

        per_model_results.append({
            "graph_id": result["graph_id"],
            "input_file": os.path.basename(path),
            "output_file": os.path.basename(out_path),
            "num_nodes": result["num_nodes"],
            "num_edges": result["num_edges"],
            "evaluated_nodes": result["evaluated_nodes"],
            "correct_predictions": result["correct_predictions"],
            "accuracy": round(result["accuracy"], 6),
            "runtime_seconds": round(runtime_sec, 6)
        })

    total_runtime = time.perf_counter() - t_all_start

    mean_of_model_accuracies = safe_div(sum(per_model_accuracies), len(per_model_accuracies))
    dataset_average_accuracy = safe_div(total_correct, total_evaluated)
    summary = {
        "method": "Structural matching",
        "description": (
            "A fully unsupervised baseline that uses only graph structure, "
            "node degree, leaf/root-like role, parent-child relation, and "
            "iterative neighborhood consistency."
        ),
        "input_dir": input_dir,
        "output_dir": output_dir,
        "num_models": len(input_files),
        "mean_of_model_accuracies": round(mean_of_model_accuracies, 6),
        "dataset_average_accuracy": round(dataset_average_accuracy, 6),
        "total_correct_predictions": total_correct,
        "total_evaluated_nodes": total_evaluated,
        "total_runtime_seconds": round(total_runtime, 6),
        "per_model_results": per_model_results
    }

    save_json(summary, os.path.join(output_dir, "summary.json"))

    print("=" * 80)
    print("Done.")
    print(f"Models processed         : {len(input_files)}")
    print(f"Mean model accuracy      : {mean_of_model_accuracies:.6f}")
    print(f"Dataset average accuracy : {dataset_average_accuracy:.6f}")
    print(f"Total runtime (seconds)  : {total_runtime:.6f}")
    print(f"Output folder            : {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()