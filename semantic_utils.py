from dataclasses import dataclass
from typing import List, Dict, Optional
import csv
import os

import numpy as np

from graph_utils import RAComponent, GraphNode
from text_utils import normalize_name

_model_cache = None


@dataclass
class MappingKnowledgeRecord:
    aadl_model: str
    component_name: str
    normalized_name: str
    ra_component_names: List[str]
    text: str


@dataclass
class MappingKnowledgeIndex:
    records: List[MappingKnowledgeRecord]
    embeddings: np.ndarray
    ra_by_norm_name: Dict[str, RAComponent]


def load_semantic_model(model_name: str):
    """
    Lazy-load SentenceTransformer model.
    """
    global _model_cache
    if _model_cache is None:
        from sentence_transformers import SentenceTransformer
        _model_cache = SentenceTransformer(model_name)
    return _model_cache


def normalize_model_id(model_id: str) -> str:
    """
    Normalize AADL model id to avoid data leakage.

    Example:
    An-AADL-Model_Clients_impl_1.aaxl2
    -> An-AADL-Model_Clients_impl_1
    """
    if not model_id:
        return ""

    base = os.path.basename(str(model_id).strip())
    base, _ = os.path.splitext(base)
    return base


def _split_ra_component_names(raw: str) -> List[str]:
    """
    One AADL component can map to multiple RA components.
    Example: Driver/Sensor -> ["Driver", "Sensor"]
    """
    if not raw:
        return []

    return [x.strip() for x in str(raw).split("/") if x.strip()]


def _build_mapping_knowledge_text(
    component_name: str,
    normalized_name: str,
) -> str:
    """
    Text representation of one historical AADL component mapping.
    Used for semantic retrieval.
    """
    name = normalized_name or normalize_name(component_name)
    return name.strip()


def _build_node_query_text(node: GraphNode) -> str:
    """
    Text representation of the current AADL node.
    """
    return normalize_name(node.name)


def _cosine_similarities(query_emb: np.ndarray, matrix_embs: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarities between one query embedding and a matrix of embeddings.
    """
    if matrix_embs.size == 0:
        return np.array([])

    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    matrix_norms = matrix_embs / (
        np.linalg.norm(matrix_embs, axis=1, keepdims=True) + 1e-9
    )

    return np.dot(matrix_norms, query_norm)


def preprocess_mapping_knowledge_csv(
    csv_path: str,
    ra_components: List[RAComponent],
    model,
) -> MappingKnowledgeIndex:
    """
    Preprocess historical ground-truth mapping CSV once.

    This function:
    - reads the CSV once
    - splits multi-mapping RA components by "/"
    - removes RA names that do not exist in current RA components
    - builds text representation for each historical mapped component
    - precomputes embeddings for all records

    Important:
    This function does NOT remove records from the current graph_id.
    Data leakage is handled later in select_top_k_ra_for_node_with_mapping_knowledge.
    """
    ra_by_norm_name = {normalize_name(c.name): c for c in ra_components}
    records: List[MappingKnowledgeRecord] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            aadl_model = row.get("AADL_model", "").strip()
            component_name = row.get("component_name", "").strip()
            normalized_name = row.get("normalized_name", "").strip()
            ra_raw = row.get("ra_component", "").strip()

            ra_component_names = _split_ra_component_names(ra_raw)

            ra_component_names = [
                name
                for name in ra_component_names
                if normalize_name(name) in ra_by_norm_name
            ]

            if not ra_component_names:
                continue

            text = _build_mapping_knowledge_text(
                component_name=component_name,
                normalized_name=normalized_name,
            )

            if not text:
                continue

            records.append(
                MappingKnowledgeRecord(
                    aadl_model=aadl_model,
                    component_name=component_name,
                    normalized_name=normalized_name,
                    ra_component_names=ra_component_names,
                    text=text,
                )
            )

    if records:
        texts = [r.text for r in records]
        embeddings = model.encode(texts, convert_to_numpy=True)
    else:
        embeddings = np.empty((0, 0))

    return MappingKnowledgeIndex(
        records=records,
        embeddings=embeddings,
        ra_by_norm_name=ra_by_norm_name,
    )


def _build_ra_text(
    c: RAComponent,
    ra_descriptions: Dict[str, str],
    use_descriptions: bool,
) -> str:
    """
    Text đại diện cho một RA component để tính similarity.
    """
    if use_descriptions:
        desc = ra_descriptions.get(c.name, "")
        if desc:
            return f"{c.name}. {desc}"
    return c.name


def select_top_k_ra_for_node(
    node_name: str,
    ra_components: List[RAComponent],
    ra_descriptions: Dict[str, str],
    use_descriptions: bool,
    model,
    top_k: int,
) -> List[RAComponent]:
    """
    Chọn top-k RA components giống node_name nhất (semantic similarity).
    node_name nên là normalized name.
    """
    if top_k <= 0 or not ra_components:
        return ra_components

    ra_texts = [
        _build_ra_text(c, ra_descriptions, use_descriptions) for c in ra_components
    ]

    node_text = normalize_name(node_name)
    node_emb = model.encode([node_text], convert_to_numpy=True)[0]
    ra_embs = model.encode(ra_texts, convert_to_numpy=True)

    sims = _cosine_similarities(node_emb, ra_embs)

    k = min(top_k, len(ra_components))
    top_indices = np.argsort(-sims)[:k]

    return [ra_components[i] for i in top_indices]


def select_top_k_ra_for_node_with_mapping_knowledge(
    node: GraphNode,
    graph_id: str,
    ra_components: List[RAComponent],
    ra_descriptions: Dict[str, str],
    use_descriptions: bool,
    model,
    top_k: int,
    mapping_knowledge_index: Optional[MappingKnowledgeIndex],
    mapping_knowledge_top_k: int = 5,
    mapping_knowledge_weight: float = 0.3,
) -> List[RAComponent]:
    """
    Hybrid RAG selection.

    Sources:
    1. RA catalog:
       RA component name + optional RA description.

    2. Historical mapping knowledge:
       Similar AADL components from other models and their mapped RA components.

    Data leakage prevention:
    Records whose AADL_model matches current graph_id are excluded.
    """
    if top_k <= 0 or not ra_components:
        return ra_components

    if mapping_knowledge_index is None or not mapping_knowledge_index.records:
        return select_top_k_ra_for_node(
            normalize_name(node.name),
            ra_components,
            ra_descriptions,
            use_descriptions,
            model,
            top_k,
        )

    mapping_knowledge_weight = max(0.0, min(1.0, mapping_knowledge_weight))
    catalog_weight = 1.0 - mapping_knowledge_weight

    node_text = _build_node_query_text(node)
    node_emb = model.encode([node_text], convert_to_numpy=True)[0]

    # 1. RA catalog similarity 
    ra_texts = [
        _build_ra_text(c, ra_descriptions, use_descriptions)
        for c in ra_components
    ]
    ra_embs = model.encode(ra_texts, convert_to_numpy=True)
    catalog_sims = _cosine_similarities(node_emb, ra_embs)

    catalog_scores = {
        normalize_name(ra_components[i].name): float(catalog_sims[i])
        for i in range(len(ra_components))
    }

    # 2. Historical mapping knowledge similarity 
    current_model_id = normalize_model_id(graph_id)

    usable_indices = [
        i
        for i, r in enumerate(mapping_knowledge_index.records)
        if normalize_model_id(r.aadl_model) != current_model_id
    ]

    knowledge_scores: Dict[str, float] = {}

    if usable_indices and mapping_knowledge_index.embeddings.size > 0:
        usable_embs = mapping_knowledge_index.embeddings[usable_indices]
        knowledge_sims = _cosine_similarities(node_emb, usable_embs)

        k_knowledge = min(mapping_knowledge_top_k, len(usable_indices))

        if k_knowledge > 0:
            top_local_indices = np.argsort(-knowledge_sims)[:k_knowledge]

            for local_idx in top_local_indices:
                global_idx = usable_indices[local_idx]
                record = mapping_knowledge_index.records[global_idx]
                sim = float(knowledge_sims[local_idx])

                for ra_name in record.ra_component_names:
                    ra_norm = normalize_name(ra_name)
                    knowledge_scores[ra_norm] = max(
                        knowledge_scores.get(ra_norm, float("-inf")),
                        sim,
                    )

    # 3. Combine scores
    ra_by_norm = {normalize_name(c.name): c for c in ra_components}
    final_scores: Dict[str, float] = {}

    for c in ra_components:
        ra_norm = normalize_name(c.name)

        catalog_score = catalog_scores.get(ra_norm, 0.0)
        knowledge_score = knowledge_scores.get(ra_norm, 0.0)

        final_scores[ra_norm] = (
            catalog_weight * catalog_score
            + mapping_knowledge_weight * knowledge_score
        )

    ranked = sorted(
        final_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        ra_by_norm[ra_norm]
        for ra_norm, _ in ranked[: min(top_k, len(ranked))]
        if ra_norm in ra_by_norm
    ]