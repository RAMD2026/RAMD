# scoring.py
from typing import Dict, Any, Tuple, List


def compute_scores_for_file(
    mapping_data: Dict[str, Any],
    predicted_mapping: Dict[str, str],
) -> Tuple[float, float, Dict[str, float]]:
    """Trả về: (total_score, avg_score, per_node_score_dict)."""
    per_node_scores: Dict[str, float] = {}
    total_score = 0.0
    node_count = 0

    # Build map node_id -> list[ (ra_name, ra_id, confidence) ]
    gt: Dict[str, List[Tuple[str, str, float]]] = {}
    for n in mapping_data["nodes"]:
        nid = n["id"]
        mappings = []
        for m in n.get("ra_mappings", []):
            mappings.append((m["ra_name"], m["ra_id"], float(m["confidence"])))
        gt[nid] = mappings

    for nid, pred_cid in predicted_mapping.items():
        node_count += 1
        gt_mappings = gt.get(nid, [])
        score = 0.0
        if gt_mappings and pred_cid:
            for (ra_name, ra_id, conf) in gt_mappings:
                # Match by ra_id or ra_name
                if pred_cid == ra_id or pred_cid == ra_name:
                    score = conf
                    break
        per_node_scores[nid] = score
        total_score += score

    avg_score = total_score / node_count if node_count > 0 else 0.0
    return total_score, avg_score, per_node_scores
