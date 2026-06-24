import json
import os
import re
import uuid
import subprocess
import difflib
import tiktoken
import xml.etree.ElementTree as ET

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict

import networkx as nx

from llm_clients import BaseLLMClient
from graph_utils import (
    GraphNode,
    GraphEdge,
    RAComponent,
    build_neighbors,
    get_ra_parent_map,
)
from prompts import (
    SYSTEM_PROMPT_OPTION1,
    SYSTEM_PROMPT_OPTION2,
    SYSTEM_PROMPT_RECREATE_MAPPING,
    build_option1_prompt,
    build_option2_prompt_for_node,
    build_recreate_mapping_prompt,
)
from text_utils import normalize_name
from semantic_utils import (
    select_top_k_ra_for_node,
    MappingKnowledgeIndex,
    select_top_k_ra_for_node_with_mapping_knowledge,
)


JAVA_DIR = Path("java")
CLASSPATH = "bin:lib/*"
_SEQ_RE = re.compile(r"Sequence\s*\{([^}]*)\}", re.IGNORECASE)
INVALID_RA_ID = "RA_ID_NOT_FOUND"


def _extract_bad_node_names_from_constraints_log(check_constraints_log: str) -> Set[str]:
    if not check_constraints_log:
        return set()

    bad: Set[str] = set()

    for m in _SEQ_RE.finditer(check_constraints_log):
        inside = m.group(1)
        names = re.findall(r'"([^"]+)"', inside)
        for n in names:
            bad.add(normalize_name(n))

    return bad


def _to_java_relative_path(path_str: str) -> str:
    try:
        p = Path(path_str)

        if not p.is_absolute():
            s = str(p).replace("\\", "/")
            if s.startswith("java/"):
                return s[len("java/") :]
            return str(p)

        return os.path.relpath(str(p), str(JAVA_DIR))

    except Exception:
        s = path_str.replace("\\", "/")
        if s.startswith("java/"):
            return s[len("java/") :]
        return path_str


def _indent_xml(elem, level: int = 0) -> None:
    i = "\n" + level * "  "

    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "

        for child in elem:
            _indent_xml(child, level + 1)

        if not elem.tail or not elem.tail.strip():
            elem.tail = i

    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def write_mapping_model(
    thread_idx: int,
    mapping: Dict[str, Optional[str]],
    ra_components: List[RAComponent],
    ra_ecore_file: str,
    graph_id: str,
    base_dir: str = (
        "java/src/it/univaq/disim/architecturemodeling/launcher/models/weaving"
    ),
) -> str:
    ra_id_to_name: Dict[str, str] = {c.id: c.name for c in ra_components}

    grouped: Dict[str, List[str]] = {}

    for node_id, ra_id in mapping.items():
        if not ra_id:
            continue

        if ra_id not in ra_id_to_name:
            raise ValueError(
                f"RA component id '{ra_id}' in mapping "
                f"does not exist in ra_components"
            )

        grouped.setdefault(ra_id, []).append(node_id)

    if graph_id.endswith(".aaxl2"):
        graph_base = graph_id[: -len(".aaxl2")]
    else:
        graph_base, _ = os.path.splitext(graph_id)

    ra_domain = ra_ecore_file.split(".")[0]
    filename = f"{ra_domain}_{graph_base}_thread_{thread_idx}.model"

    os.makedirs(base_dir, exist_ok=True)
    out_path = os.path.join(base_dir, filename)

    root_attrib = {
        "xmi:version": "2.0",
        "xmlns:xmi": "http://www.omg.org/XMI",
        "xmlns:ecore": "http://it.univaq.disim/ra_adl",
        "xmi:id": "_" + uuid.uuid4().hex,
    }

    root = ET.Element("ecore:Implementation_model", root_attrib)

    for ra_id, node_ids in grouped.items():
        rel_elem = ET.SubElement(
            root,
            "relations",
            {"xmi:id": "_" + uuid.uuid4().hex},
        )

        ra_name = ra_id_to_name[ra_id]
        ra_href = f"../referencearchitectures/{ra_ecore_file}#//{ra_name}"
        ET.SubElement(rel_elem, "RAcomp", {"href": ra_href})

        for node_id in node_ids:
            comp_href = f"../architectures/{graph_base}.model#{node_id}"
            ET.SubElement(rel_elem, "component", {"href": comp_href})

    _indent_xml(root)
    xml_body = ET.tostring(root, encoding="unicode")
    xml_text = '<?xml version="1.0" encoding="ASCII"?>\n' + xml_body

    with open(out_path, "w", encoding="ascii", errors="xmlcharrefreplace") as f:
        f.write(xml_text)

    return out_path


def _extract_json_from_text(raw_text: str) -> str:
    text = raw_text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    stripped = text.lstrip()

    if not stripped.startswith("{") and "{" in stripped and "}" in stripped:
        start = stripped.find("{")
        end = stripped.rfind("}")

        if start != -1 and end != -1 and end > start:
            text = stripped[start : end + 1]

    return text


def _build_ra_name_to_id(ra_components: List[RAComponent]) -> Dict[str, str]:
    name_to_id: Dict[str, str] = {}

    for c in ra_components:
        key = c.name.strip().lower()
        name_to_id[key] = c.id

    return name_to_id


def _chunk_list(items: List[Any], size: int) -> List[List[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _extract_constraint_messages(stderr: str) -> str:
    if not stderr:
        return ""

    lines = [ln.rstrip() for ln in stderr.splitlines() if ln.strip()]

    cut_markers = (
        "Exception in thread",
        "java.lang.",
        "Caused by:",
        "\tat ",
        "at ",
    )

    trimmed: List[str] = []

    for ln in lines:
        if any(ln.startswith(m) for m in cut_markers):
            break
        trimmed.append(ln)

    start_idx = None

    for i, ln in enumerate(trimmed):
        if "constraint(s) have not been satisfied" in ln:
            start_idx = i
            break

    if start_idx is None:
        return ""

    msg_lines = trimmed[start_idx:]
    return "\n".join(msg_lines).strip()


_CONSTRAINT_COUNT_RE = re.compile(
    r"(\d+)\s+constraint\(s\)\s+have\s+not\s+been\s+satisfied",
    re.IGNORECASE,
)


def _extract_constraint_violation_count(check_constraints_log: str) -> Optional[int]:
    if not check_constraints_log:
        return None

    m = _CONSTRAINT_COUNT_RE.search(check_constraints_log)

    if not m:
        return None

    try:
        return int(m.group(1))
    except Exception:
        return None


def check_constraints(mapping_model_path, reference_arch_name):
    metamodel_path = (
        "src/it/univaq/disim/architecturemodeling/launcher/metamodels/RA_ADL.ecore"
    )

    reference_arch_prefix = (
        "src/it/univaq/disim/architecturemodeling/launcher/models/"
        "referencearchitectures/"
    )

    reference_arch_ecore_path = reference_arch_prefix + reference_arch_name

    cmd = [
        "java",
        "-cp",
        CLASSPATH,
        "it.univaq.disim.architecturemodeling.launcher.validation.RAV",
        mapping_model_path,
        metamodel_path,
        reference_arch_ecore_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=JAVA_DIR,
            capture_output=True,
            text=True,
        )
    except Exception as e:
        print(f"[check_constraints] Error running Java: {e}")
        return False, ""

    stdout = result.stdout or ""
    stderr = result.stderr or ""

    if "All constraints have been satisfied" in stdout:
        return True, "All constraints have been satisfied"

    msg = _extract_constraint_messages(stderr)

    if msg:
        print(msg)
    else:
        print("Constraints not satisfied (no detailed message found).")

    return False, msg


def parse_option1_output(
    raw_text: str,
    node_name_to_id: Dict[str, str],
    ra_components: List[RAComponent],
) -> Dict[str, Optional[str]]:
    cleaned = _extract_json_from_text(raw_text)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        raise ValueError(
            f"LLM output is not valid JSON.\nRAW:\n{raw_text}\nCLEANED:\n{cleaned}"
        )

    if "mappings" not in data or not isinstance(data["mappings"], list):
        raise ValueError(f"LLM JSON missing 'mappings' list: {data}")

    ra_name_to_id = _build_ra_name_to_id(ra_components)
    mapping: Dict[str, Optional[str]] = {}

    for item in data["mappings"]:
        raw_node_name = item.get("node_name", "")
        norm_node_name = normalize_name(raw_node_name)

        raw_ra_name = item.get("ra_component_name", "")
        ra_key = raw_ra_name.strip().lower()
        ra_id = ra_name_to_id.get(ra_key)

        node_id = node_name_to_id.get(norm_node_name)

        if node_id and ra_id:
            mapping[node_id] = ra_id

    for node_id in node_name_to_id.values():
        mapping.setdefault(node_id, None)

    return mapping


def count_tokens_for_gpt(text: str, model_name: str = "gpt-4o-mini") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except Exception:
        encoding = tiktoken.get_encoding("o200k_base")

    return len(encoding.encode(text))


def estimate_llm_token_cost(
    system_prompt: str,
    user_prompt: str,
    raw_output: str,
    model_name: str = "gpt-4o-mini",
) -> int:
    input_text = system_prompt + "\n" + user_prompt

    input_tokens = count_tokens_for_gpt(input_text, model_name)
    output_tokens = count_tokens_for_gpt(raw_output, model_name)

    return input_tokens + output_tokens


def parse_recreated_mapping_output(
    raw_text: str,
    valid_node_ids: Set[str],
    valid_ra_ids: Set[str],
) -> Dict[str, Optional[str]]:
    cleaned = _extract_json_from_text(raw_text)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        raise ValueError(
            f"Recreated mapping output is not valid JSON.\n"
            f"RAW:\n{raw_text}\nCLEANED:\n{cleaned}"
        )

    if "mapping" not in data or not isinstance(data["mapping"], dict):
        raise ValueError(f"Recreated mapping JSON missing 'mapping' object: {data}")

    raw_mapping = data["mapping"]
    mapping: Dict[str, Optional[str]] = {}

    for node_id in valid_node_ids:
        ra_id = raw_mapping.get(node_id, None)

        if ra_id is None:
            mapping[node_id] = None

        elif isinstance(ra_id, str) and ra_id in valid_ra_ids:
            mapping[node_id] = ra_id

        else:
            print(
                f"[parse_recreated_mapping_output] Invalid RA id for node "
                f"'{node_id}': {ra_id}. Set to None."
            )
            mapping[node_id] = None

    extra_keys = set(raw_mapping.keys()) - valid_node_ids

    if extra_keys:
        print(
            "[parse_recreated_mapping_output] Ignored unknown node ids: "
            f"{sorted(extra_keys)}"
        )

    return mapping


def maybe_recreate_mapping_with_llm(
    llm: BaseLLMClient,
    graph_id: str,
    nodes: List[GraphNode],
    ra_components: List[RAComponent],
    mapping_history: List[Dict[str, Optional[str]]],
    recreate_mapping_with_llm: bool,
    recreate_window: int,
    iterations_since_recreate: int,
    check_constraints_log: str,
    llm_model_name: str = "gpt-4o-mini",
) -> Tuple[Optional[Dict[str, Optional[str]]], Optional[int], bool, int]:
    if not recreate_mapping_with_llm:
        return None, None, False, iterations_since_recreate

    if recreate_window <= 0:
        return None, None, False, iterations_since_recreate

    if iterations_since_recreate < recreate_window:
        return None, None, False, iterations_since_recreate

    if len(mapping_history) < recreate_window:
        return None, None, False, iterations_since_recreate

    recent_mappings = mapping_history[-recreate_window:]

    user_prompt = build_recreate_mapping_prompt(
        graph_id=graph_id,
        recent_mappings=recent_mappings,
        nodes=nodes,
        ra_components=ra_components,
        constraint_feedback=check_constraints_log,
    )

    raw = llm.generate(SYSTEM_PROMPT_RECREATE_MAPPING, user_prompt)

    recreate_cost = estimate_llm_token_cost(
        system_prompt=SYSTEM_PROMPT_RECREATE_MAPPING,
        user_prompt=user_prompt,
        raw_output=raw,
        model_name=llm_model_name,
    )

    valid_node_ids = {n.id for n in nodes}
    valid_ra_ids = {c.id for c in ra_components}

    recreated_mapping = parse_recreated_mapping_output(
        raw_text=raw,
        valid_node_ids=valid_node_ids,
        valid_ra_ids=valid_ra_ids,
    )

    return recreated_mapping, recreate_cost, True, 0


def run_option1_mapping(
    thread_idx: int,
    llm: BaseLLMClient,
    ra_ecore_file: str,
    graph_id: str,
    nodes: List[GraphNode],
    edges: List[GraphEdge],
    ra_components: List[RAComponent],
    ra_edges: Set[Tuple[str, str]],
    ra_parent: Dict[str, Optional[str]],
    max_iterations: int = 3,
    stop_if_same: bool = True,
    use_ra_descriptions: bool = True,
    use_ra_structural_info: bool = False,
    ra_descriptions: Optional[Dict[str, str]] = None,
    check_RA_constraints: bool = True,
    same_mapping_patience: int = 3,
    fewshot_learning: bool = False,
    llm_model_name: str = "gpt-4o-mini",
    recreate_mapping_with_llm: bool = False,
    recreate_window: int = 3,
    break_after_recreate: bool = False,
) -> Tuple[Dict[str, Optional[str]], str, float, int]:

    name_to_id = {normalize_name(n.name): n.id for n in nodes}
    ra_domain = ra_ecore_file.split(".")[0]

    last_mapping: Optional[Dict[str, Optional[str]]] = None
    last_valid_mapping: Optional[Dict[str, Optional[str]]] = None

    best_invalid_mapping: Optional[Dict[str, Optional[str]]] = None
    best_invalid_constraint_count: Optional[int] = None
    best_invalid_log: str = ""

    same_count = 0

    bad_node_names_prev_iter: Set[str] = set()
    constraint_feedback_lines_prev_iter: List[str] = []
    check_constraints_log = ""

    llm_prompt_costs: List[int] = []
    total_llm_cost = 0

    mapping_history: List[Dict[str, Optional[str]]] = []
    iterations_since_recreate = 0

    for _ in range(max_iterations):
        user_prompt = build_option1_prompt(
            graph_id=graph_id,
            nodes=nodes,
            edges=edges,
            ra_components=ra_components,
            ra_edges=ra_edges,
            use_ra_descriptions=use_ra_descriptions,
            use_ra_structural_info=use_ra_structural_info,
            ra_descriptions=ra_descriptions,
            ra_domain=ra_domain,
            fewshot_learning=fewshot_learning,
        )

        if check_RA_constraints and constraint_feedback_lines_prev_iter:
            user_prompt += (
                "\n\nConstraint feedback (previous mapping violated RA rules):\n"
                + "\n".join(constraint_feedback_lines_prev_iter)
                + "\nIMPORTANT: Try to preserve the existing correct mappings as much "
                "as possible. If some mappings violate the constraints, modify them "
                "and replace them with more appropriate alternatives when available."
            )
            print(user_prompt)

        raw = llm.generate(SYSTEM_PROMPT_OPTION1, user_prompt)

        prompt_cost = estimate_llm_token_cost(
            system_prompt=SYSTEM_PROMPT_OPTION1,
            user_prompt=user_prompt,
            raw_output=raw,
            model_name=llm_model_name,
        )

        llm_prompt_costs.append(prompt_cost)
        total_llm_cost += prompt_cost

        mapping = parse_option1_output(raw, name_to_id, ra_components)

        mapping_history.append(mapping.copy())
        iterations_since_recreate += 1
        recreated_this_iter = False

        try:
            (
                recreated_mapping,
                recreate_cost,
                recreated_this_iter,
                iterations_since_recreate,
            ) = maybe_recreate_mapping_with_llm(
                llm=llm,
                graph_id=graph_id,
                nodes=nodes,
                ra_components=ra_components,
                mapping_history=mapping_history,
                recreate_mapping_with_llm=recreate_mapping_with_llm,
                recreate_window=recreate_window,
                iterations_since_recreate=iterations_since_recreate,
                check_constraints_log=check_constraints_log,
                llm_model_name=llm_model_name,
            )

            if recreated_mapping is not None:
                mapping = recreated_mapping
                mapping_history[-1] = mapping.copy()

            if recreate_cost is not None:
                llm_prompt_costs.append(recreate_cost)
                total_llm_cost += recreate_cost

        except Exception as e:
            print(f"[run_option1_mapping] Error during recreate mapping: {e}")
            recreated_this_iter = False

        current_constraints_satisfied = False
        current_violation_count: Optional[int] = None
        current_log = ""

        if check_RA_constraints:
            try:
                model_path = write_mapping_model(
                    thread_idx,
                    mapping,
                    ra_components,
                    ra_ecore_file,
                    graph_id,
                )

                model_path = _to_java_relative_path(model_path)

                current_constraints_satisfied, current_log = check_constraints(
                    model_path,
                    ra_ecore_file,
                )

                check_constraints_log = current_log

                if current_constraints_satisfied:
                    print("All constraints have been satisfied")
                    last_valid_mapping = mapping
                    break

                current_violation_count = _extract_constraint_violation_count(
                    current_log
                )

                if current_violation_count is not None:
                    if (
                        best_invalid_constraint_count is None
                        or current_violation_count < best_invalid_constraint_count
                    ):
                        best_invalid_constraint_count = current_violation_count
                        best_invalid_mapping = mapping.copy()
                        best_invalid_log = current_log

                if current_log:
                    bad_node_names_prev_iter = (
                        _extract_bad_node_names_from_constraints_log(current_log)
                    )
                    constraint_feedback_lines_prev_iter = [
                        "The following RA constraints were violated:",
                        current_log,
                    ]
                else:
                    bad_node_names_prev_iter = set()
                    constraint_feedback_lines_prev_iter = []

            except Exception as e:
                print(f"[run_option1_mapping] Error during constraint check: {e}")
                bad_node_names_prev_iter = set()
                constraint_feedback_lines_prev_iter = []

        else:
            bad_node_names_prev_iter = set()
            constraint_feedback_lines_prev_iter = []

        if stop_if_same and last_mapping is not None and mapping == last_mapping:
            same_count += 1
        else:
            same_count = 0

        last_mapping = mapping

        if recreated_this_iter and break_after_recreate:
            break

        if stop_if_same and same_count >= max(1, same_mapping_patience):
            break

    avg_prompt_cost = (
        total_llm_cost / len(llm_prompt_costs) if llm_prompt_costs else 0.0
    )

    if last_valid_mapping is not None:
        return last_valid_mapping, check_constraints_log, avg_prompt_cost, total_llm_cost

    if best_invalid_mapping is not None:
        return best_invalid_mapping, best_invalid_log, avg_prompt_cost, total_llm_cost

    if last_mapping is not None:
        return last_mapping, check_constraints_log, avg_prompt_cost, total_llm_cost

    return {n.id: None for n in nodes}, check_constraints_log, avg_prompt_cost, total_llm_cost


def run_option2_mapping(
    thread_idx: int,
    llm: BaseLLMClient,
    ra_ecore_file: str,
    graph_id: str,
    nodes: List[GraphNode],
    edges: List[GraphEdge],
    ra_components: List[RAComponent],
    ra_edges: set,
    max_iterations: int,
    stop_if_same: bool,
    use_ra_descriptions: bool,
    use_ra_structural_info: bool,
    ra_descriptions: Dict[str, str],
    use_chunking: bool,
    chunk_size: int,
    use_semantic_filter: bool,
    semantic_model,
    semantic_top_k: int,
    ra_parent: Dict[str, Optional[str]],
    check_RA_constraints: bool = True,
    same_mapping_patience: int = 3,
    fewshot_learning: bool = False,
    mapping_knowledge_index: Optional[MappingKnowledgeIndex] = None,
    mapping_knowledge_top_k: int = 5,
    mapping_knowledge_weight: float = 0.3,
    llm_model_name: str = "gpt-4o-mini",
    recreate_mapping_with_llm: bool = False,
    recreate_window: int = 3,
    break_after_recreate: bool = False,
) -> Tuple[Dict[str, Optional[str]], str, float, int]:

    node_ids = [n.id for n in nodes]
    neighbors = build_neighbors(edges)
    id_to_node = {n.id: n for n in nodes}
    ra_domain = ra_ecore_file.split(".")[0]

    node_name_to_id = {normalize_name(n.name): n.id for n in nodes}

    last_mapping: Optional[Dict[str, Optional[str]]] = None
    last_valid_mapping: Optional[Dict[str, Optional[str]]] = None

    best_invalid_mapping: Optional[Dict[str, Optional[str]]] = None
    best_invalid_constraint_count: Optional[int] = None
    best_invalid_log: str = ""

    effective_chunking = use_chunking and use_ra_descriptions and (
        not use_semantic_filter
    )

    same_count = 0

    bad_node_norm_names_prev_iter: Set[str] = set()
    constraint_feedback_lines_prev_iter: List[str] = []
    check_constraints_log = ""

    llm_prompt_costs: List[int] = []
    total_llm_cost = 0

    mapping_history: List[Dict[str, Optional[str]]] = []
    iterations_since_recreate = 0

    for _ in range(max_iterations):
        mapping: Dict[str, Optional[str]] = {}

        for node in nodes:
            nbr_ids = neighbors.get(node.id, [])
            nbr_nodes = [id_to_node[nid] for nid in nbr_ids if nid in id_to_node]

            node_norm_name = normalize_name(node.name)

            if use_semantic_filter and semantic_model is not None:
                target_name = node_norm_name

                if mapping_knowledge_index is not None:
                    ra_candidates = select_top_k_ra_for_node_with_mapping_knowledge(
                        node=node,
                        graph_id=graph_id,
                        ra_components=ra_components,
                        ra_descriptions=ra_descriptions,
                        use_descriptions=use_ra_descriptions,
                        model=semantic_model,
                        top_k=semantic_top_k,
                        mapping_knowledge_index=mapping_knowledge_index,
                        mapping_knowledge_top_k=mapping_knowledge_top_k,
                        mapping_knowledge_weight=mapping_knowledge_weight,
                    )
                else:
                    ra_candidates = select_top_k_ra_for_node(
                        target_name,
                        ra_components,
                        ra_descriptions,
                        use_ra_descriptions,
                        semantic_model,
                        semantic_top_k,
                    )
            else:
                ra_candidates = ra_components

            def _build_user_prompt_for_candidates(cands: List[RAComponent]) -> str:
                p = build_option2_prompt_for_node(
                    graph_id=graph_id,
                    node=node,
                    neighbor_nodes=nbr_nodes,
                    ra_components=cands,
                    ra_edges=ra_edges,
                    use_ra_descriptions=use_ra_descriptions,
                    use_ra_structural_info=use_ra_structural_info,
                    ra_descriptions=ra_descriptions,
                    ra_domain=ra_domain,
                    fewshot_learning=fewshot_learning,
                )

                if check_RA_constraints and constraint_feedback_lines_prev_iter:
                    # print("constraint_feedback_lines_prev_iter: ", constraint_feedback_lines_prev_iter)
                    p += (
                        "\n\nConstraint feedback "
                        "(previous mapping violated RA rules):\n"
                        + "\n".join(constraint_feedback_lines_prev_iter)
                        + "\nIMPORTANT: Try to preserve the existing correct mappings "
                        "as much as possible. If some mappings violate the constraints, "
                        "modify them and replace them with more appropriate alternatives "
                        "when available."
                    )

                return p

            if effective_chunking:
                best_ra_id = None
                best_score = float("-inf")

                for chunk in _chunk_list(ra_candidates, chunk_size):
                    user_prompt = _build_user_prompt_for_candidates(chunk)

                    raw = llm.generate(SYSTEM_PROMPT_OPTION2, user_prompt)

                    prompt_cost = estimate_llm_token_cost(
                        system_prompt=SYSTEM_PROMPT_OPTION2,
                        user_prompt=user_prompt,
                        raw_output=raw,
                        model_name=llm_model_name,
                    )

                    llm_prompt_costs.append(prompt_cost)
                    total_llm_cost += prompt_cost

                    ra_id, score = parse_option2_output(raw, chunk)

                    if ra_id == INVALID_RA_ID:
                        print(
                            f"[run_option2_mapping] WARNING: Could not map node "
                            f"'{node.name}' in chunk. raw output skipped."
                        )
                        continue

                    if score > best_score:
                        best_score = score
                        best_ra_id = ra_id

                mapping[node.id] = best_ra_id

            else:
                user_prompt = _build_user_prompt_for_candidates(ra_candidates)

                raw = llm.generate(SYSTEM_PROMPT_OPTION2, user_prompt)

                prompt_cost = estimate_llm_token_cost(
                    system_prompt=SYSTEM_PROMPT_OPTION2,
                    user_prompt=user_prompt,
                    raw_output=raw,
                    model_name=llm_model_name,
                )

                llm_prompt_costs.append(prompt_cost)
                total_llm_cost += prompt_cost

                ra_id, _ = parse_option2_output(raw, ra_candidates)

                if ra_id == INVALID_RA_ID:
                    print("=======================================")
                    print("ra_candidates: ", ra_candidates)
                    print("=======================================")
                    print("raw llm_output: ", raw)
                    print(
                        f"[run_option2_mapping] WARNING: Could not map node "
                        f"'{node.name}'. Set mapping to None for this iteration."
                    )
                    mapping[node.id] = None
                else:
                    mapping[node.id] = ra_id

        mapping_history.append(mapping.copy())
        print("current mapping: ", mapping)
        iterations_since_recreate += 1
        recreated_this_iter = False

        try:
            (
                recreated_mapping,
                recreate_cost,
                recreated_this_iter,
                iterations_since_recreate,
            ) = maybe_recreate_mapping_with_llm(
                llm=llm,
                graph_id=graph_id,
                nodes=nodes,
                ra_components=ra_components,
                mapping_history=mapping_history,
                recreate_mapping_with_llm=recreate_mapping_with_llm,
                recreate_window=recreate_window,
                iterations_since_recreate=iterations_since_recreate,
                check_constraints_log=check_constraints_log,
                llm_model_name=llm_model_name,
            )

            if recreated_mapping is not None:
                mapping = recreated_mapping
                mapping_history[-1] = mapping.copy()

            if recreate_cost is not None:
                llm_prompt_costs.append(recreate_cost)
                total_llm_cost += recreate_cost

        except Exception as e:
            print(f"[run_option2_mapping] Error during recreate mapping: {e}")
            recreated_this_iter = False

        current_constraints_satisfied = False
        current_violation_count: Optional[int] = None
        current_log = ""

        if check_RA_constraints:
            try:
                if any(v is None for v in mapping.values()):
                    print(
                        "[run_option2_mapping] WARNING: Some nodes are unmapped. "
                        "Skip constraint check for this iteration."
                    )
                    bad_node_norm_names_prev_iter = set()
                    constraint_feedback_lines_prev_iter = []

                else:
                    model_path = write_mapping_model(
                        thread_idx,
                        mapping,
                        ra_components,
                        ra_ecore_file,
                        graph_id,
                    )

                    model_path = _to_java_relative_path(model_path)

                    current_constraints_satisfied, current_log = check_constraints(
                        model_path,
                        ra_ecore_file,
                    )

                    check_constraints_log = current_log

                    if current_constraints_satisfied:
                        print("All constraints have been satisfied")
                        last_valid_mapping = mapping
                        break

                    current_violation_count = _extract_constraint_violation_count(
                        current_log
                    )

                    if current_violation_count is not None:
                        if (
                            best_invalid_constraint_count is None
                            or current_violation_count < best_invalid_constraint_count
                        ):
                            best_invalid_constraint_count = current_violation_count
                            best_invalid_mapping = mapping.copy()
                            best_invalid_log = current_log

                    if current_log:
                        bad_node_norm_names_prev_iter = (
                            _extract_bad_node_names_from_constraints_log(current_log)
                        )
                        constraint_feedback_lines_prev_iter = [
                            "The following RA constraints were violated:",
                            current_log,
                        ]
                    else:
                        bad_node_norm_names_prev_iter = set()
                        constraint_feedback_lines_prev_iter = []

            except Exception as e:
                print(f"[run_option2_mapping] Error during constraint check: {e}")
                bad_node_norm_names_prev_iter = set()
                constraint_feedback_lines_prev_iter = []

        else:
            bad_node_norm_names_prev_iter = set()
            constraint_feedback_lines_prev_iter = []

        if stop_if_same and last_mapping is not None and mapping == last_mapping:
            same_count += 1
        else:
            same_count = 0

        last_mapping = mapping

        if recreated_this_iter and break_after_recreate:
            break

        if stop_if_same and same_count >= max(1, same_mapping_patience):
            break

    avg_prompt_cost = (
        total_llm_cost / len(llm_prompt_costs) if llm_prompt_costs else 0.0
    )

    if last_valid_mapping is not None:
        return last_valid_mapping, check_constraints_log, avg_prompt_cost, total_llm_cost

    if best_invalid_mapping is not None:
        return best_invalid_mapping, best_invalid_log, avg_prompt_cost, total_llm_cost

    if last_mapping is not None:
        return last_mapping, check_constraints_log, avg_prompt_cost, total_llm_cost

    return {nid: None for nid in node_ids}, check_constraints_log, avg_prompt_cost, total_llm_cost


def _normalize_label(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _repair_ra_label(
    predicted_label: str,
    ra_components: List[RAComponent],
    threshold: float = 0.85,
):
    for c in ra_components:
        if c.name == predicted_label:
            return c.id, c.name

    pred_norm = _normalize_label(predicted_label)
    print("predicted_label: ", pred_norm)
    print("ra_components: ", ra_components)
    for c in ra_components:
        if _normalize_label(c.name) == pred_norm:
            return c.id, c.name

    norm_to_comp = {_normalize_label(c.name): c for c in ra_components}

    matches = difflib.get_close_matches(
        pred_norm,
        norm_to_comp.keys(),
        n=1,
        cutoff=threshold,
    )

    if matches:
        c = norm_to_comp[matches[0]]
        print(f"[parse_option2_output] repaired '{predicted_label}' -> '{c.name}'")
        return c.id, c.name

    return None, None


def parse_option2_output(
    raw_text: str,
    ra_components: List[RAComponent],
) -> Tuple[str, float]:
    if "}]}}" in raw_text:
        raw_text = raw_text.replace("}]}}", "}]}")

    cleaned = _extract_json_from_text(raw_text)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"[parse_option2_output] invalid JSON: {raw_text}")
        return INVALID_RA_ID, 0.0

    best_label = data.get("best")
    ranked = data.get("ranked", [])

    if not isinstance(best_label, str):
        print(f"[parse_option2_output] invalid best field: {data}")
        return INVALID_RA_ID, 0.0

    best_score = 1.0

    if isinstance(ranked, list):
        for item in ranked:
            if (
                isinstance(item, dict)
                and item.get("label") == best_label
                and isinstance(item.get("score"), (int, float))
            ):
                best_score = float(item["score"])
                break

    ra_id, matched_name = _repair_ra_label(best_label, ra_components)

    if ra_id:
        return ra_id, best_score

    print(f"[parse_option2_output] unknown RA label: {best_label}")
    return INVALID_RA_ID, best_score