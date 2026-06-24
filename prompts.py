from typing import List, Dict, Optional, Tuple, Set
from graph_utils import GraphNode, GraphEdge, RAComponent
from text_utils import normalize_name
import json

# OPTION 1 (map all nodes)
SYSTEM_PROMPT_OPTION1 = (
    "You are an expert software architect. "
    "Your task is to map nodes in an AADL architecture graph to components "
    "in a reference architecture. "
    "You MUST output ONLY a single valid JSON object, without any markdown "
    "formatting, code fences, or explanations."
)

# OPTION 2 (map each node)
SYSTEM_PROMPT_OPTION2 = (
    "You are an expert software architect and a careful classifier.\n"
    "You will receive:\n"
    "  (a) the name of a target node in an AADL graph,\n"
    "  (b) the names of its neighbor nodes (if any), and\n"
    "  (c) a finite set of reference architecture components as labels (as a list of names).\n"
    "Select EXACTLY ONE best label for the target node and also return a ranked list\n"
    "of labels with confidence scores in [0,1].\n"
    "All labels in the output MUST be exactly one of the provided component names.\n"
    "Return ONLY strict JSON with this schema:\n"
    '{"best":"<label>", "ranked":[{"label":"<label>","score":0.0}, ...]}'
)

_FEWSHOT_HEADER = """Few-shot examples (follow the SAME output schema strictly; labels must be chosen from the given list):"""

# FEWSHOT - OPTION 1
_FEWSHOT_OPTION1_IOT = """
Example 1
Graph id: smart_home_arch_RemoteServer_RemoteServer_impl_1.aaxl2

Nodes (names):
- system
- client controller
- router controller

Edges between node names (undirected view):
- system -- client controller
- system -- router controller

Reference Architecture components (labels):
- Application
- IoTIM
- Gateway
- Device
- Sensor
- Actuator

Output:
{
  "mappings": [
    {"node_name": "system", "ra_component_name": "IoTIM"},
    {"node_name": "client controller", "ra_component_name": "Application"},
    {"node_name": "router controller", "ra_component_name": "Gateway"}
  ]
}

Example 2
Graph id: DHsystem_Devices_humidity_impl_1.aaxl2

Nodes (names):
- system
- dehumidifier
- humidifier
- humidity sensor

Edges between node names (undirected view):
- humidity sensor -- dehumidifier
- humidity sensor -- humidifier

Reference Architecture components (labels):
- Application
- IoTIM
- Gateway
- Device
- Sensor
- Actuator

Constraint feedback (previous mapping violated RA rules):
1 constraint(s) have not been satisfied
constraint=relationType | component: Sequence {"humidifier", "dehumidifier"} does not respect the RA rules. Reasons: humidifier (mapped to Actuator) connects to humidity sensor (mapped to Sensor), but in the RA there is no allowed connection from Actuator to Sensor. || dehumidifier (mapped to Actuator) connects to humidity sensor (mapped to Sensor), but in the RA there is no allowed connection from Actuator to Sensor.
IMPORTANT: Keep other correct mappings if possible, but change the violating ones.

Output:
{
  "mappings": [
    {"node_name": "system", "ra_component_name": "Application"},
    {"node_name": "dehumidifier", "ra_component_name": "Device"},
    {"node_name": "humidifier", "ra_component_name": "Device"},
    {"node_name": "humidity sensor", "ra_component_name": "Sensor"}
  ]
}
""".strip()

_FEWSHOT_OPTION1_AUTONOMOUS_DRIVING_SYSTEM = """
Example:
Graph id: AADL-Self-driving-car_integration_integration_functional_3.aaxl2

Nodes (names):
- camera
- radar1
- radar2
- traffic lane sensor
- induction speed sensor
- optical speed sensor
- acceleration pedal
- brake
- accelerator
- panel
- data acquisition
- thr image analysis
- thr distance voter
- thr overtaking detection
- thr speed voter
- panel controller
- speed controller

Edges between node names (undirected view):
- camera -- thr image analysis
- radar1 -- thr distance voter
- radar2 -- thr distance voter
- traffic lane sensor -- thr overtaking detection
- induction speed sensor -- thr speed voter
- optical speed sensor -- thr speed voter
- acceleration pedal -- panel controller
- acceleration pedal -- speed controller
- thr image analysis -- speed controller
- thr distance voter -- speed controller
- thr overtaking detection -- speed controller
- thr speed voter -- speed controller
- thr speed voter -- panel controller
- panel controller -- panel
- speed controller -- brake
- speed controller -- accelerator

Reference Architecture components (labels):
- Sensors
- Processing_Unit
- V2X_Cloud_Comms
- Mobile_Platform_Actuators
- Internal_Networking_Interfaces
- Software_Frameworks_and_Standards
- ML_AI_DL_Algorithms
- Data_Collection
- UI_UX_and_Infotainment
- Real_time_and_Critical_Control_Software

Output:
{
  "mappings": [
    {"node_name": "camera", "ra_component_name": "Sensors"},
    {"node_name": "radar1", "ra_component_name": "Sensors"},
    {"node_name": "radar2", "ra_component_name": "Sensors"},
    {"node_name": "traffic lane sensor", "ra_component_name": "Sensors"},
    {"node_name": "induction speed sensor", "ra_component_name": "Sensors"},
    {"node_name": "optical speed sensor", "ra_component_name": "Sensors"},
    {"node_name": "acceleration pedal", "ra_component_name": "Sensors"},
    {"node_name": "brake", "ra_component_name": "Mobile_Platform_Actuators"},
    {"node_name": "accelerator", "ra_component_name": "Mobile_Platform_Actuators"},
    {"node_name": "panel", "ra_component_name": "UI_UX_and_Infotainment"},
    {"node_name": "data acquisition", "ra_component_name": "Data_Collection"},
    {"node_name": "thr image analysis", "ra_component_name": "ML_AI_DL_Algorithms"},
    {"node_name": "thr distance voter", "ra_component_name": "Real_time_and_Critical_Control_Software"},
    {"node_name": "thr overtaking detection", "ra_component_name": "ML_AI_DL_Algorithms"},
    {"node_name": "thr speed voter", "ra_component_name": "Real_time_and_Critical_Control_Software"},
    {"node_name": "panel controller", "ra_component_name": "UI_UX_and_Infotainment"},
    {"node_name": "speed controller", "ra_component_name": "Real_time_and_Critical_Control_Software"}
  ]
}
""".strip()

_FEWSHOT_OPTION1_SMARTPARKING = """
Example:
Graph id: AadlProjects_smartparking_personallocmanagement_impl_1.aaxl2

Nodes (names):
- system
- travnavmapdatabase
- driverloc

Edges between node names (undirected view):
- driverloc -- system

Reference Architecture components (labels):
- Presentation_Layer
- Business_Logic_Layer
- Data_Management_Layer
- Security

Output:
{
  "mappings": [
    {"node_name": "system", "ra_component_name": "Business_Logic_Layer"},
    {"node_name": "travnavmapdatabase", "ra_component_name": "Data_Management_Layer"},
    {"node_name": "driverloc", "ra_component_name": "Business_Logic_Layer"}
  ]
}
""".strip()

# FEWSHOT - OPTION 2
_FEWSHOT_OPTION2_IOT = """
Example 1
Graph id: smart_home_arch_RemoteServer_RemoteServer_impl_1.aaxl2

Target node name: system

Neighbor node names:
- client controller
- router controller

Reference Architecture components (labels):
- Application
- IoTIM
- Gateway
- Device
- Sensor
- Actuator

Output:
{"best":"IoTIM","ranked":[{"label":"IoTIM","score":0.92},{"label":"Application","score":0.07},{"label":"Device","score":0.01}]}

Example 2
Graph id: DHsystem_Devices_humidity_impl_1.aaxl2

Target node name: dehumidifier

Neighbor node names:
- humidity sensor

Reference Architecture components (labels):
- Application
- IoTIM
- Gateway
- Device
- Sensor
- Actuator

Constraint feedback (previous mapping violated RA rules):
1 constraint(s) have not been satisfied
constraint=relationType | component: Sequence {"humidifier", "dehumidifier"} does not respect the RA rules. Reasons: humidifier (mapped to Actuator) connects to humidity sensor (mapped to Sensor), but in the RA there is no allowed connection from Actuator to Sensor. || dehumidifier (mapped to Actuator) connects to humidity sensor (mapped to Sensor), but in the RA there is no allowed connection from Actuator to Sensor.

Output:
{"best":"Device","ranked":[{"label":"Device","score":0.90},{"label":"Application","score":0.07},{"label":"Actuator","score":0.03}]}
""".strip()

_FEWSHOT_OPTION2_AUTONOMOUS_DRIVING_SYSTEM = """
Example 1
Graph id: AADL-Self-driving-car_integration_integration_functional_3.aaxl2

Target node name: camera

Neighbor node names:
- thr image analysis

Reference Architecture components (labels):
- Sensors
- Processing_Unit
- V2X_Cloud_Comms
- Mobile_Platform_Actuators
- Internal_Networking_Interfaces
- Software_Frameworks_and_Standards
- ML_AI_DL_Algorithms
- Data_Collection
- UI_UX_and_Infotainment
- Real_time_and_Critical_Control_Software

Output:
{"best":"Sensors","ranked":[{"label":"Sensors","score":0.97},{"label":"Data_Collection","score":0.02},{"label":"ML_AI_DL_Algorithms","score":0.01}]}

Example 2
Graph id: AADL-Self-driving-car_integration_integration_functional_3.aaxl2

Target node name: thr image analysis

Neighbor node names:
- camera
- speed controller

Reference Architecture components (labels):
- Sensors
- Processing_Unit
- V2X_Cloud_Comms
- Mobile_Platform_Actuators
- Internal_Networking_Interfaces
- Software_Frameworks_and_Standards
- ML_AI_DL_Algorithms
- Data_Collection
- UI_UX_and_Infotainment
- Real_time_and_Critical_Control_Software

Output:
{"best":"ML_AI_DL_Algorithms","ranked":[{"label":"ML_AI_DL_Algorithms","score":0.93},{"label":"Data_Collection","score":0.04},{"label":"Real_time_and_Critical_Control_Software","score":0.03}]}
""".strip()

_FEWSHOT_OPTION2_SMARTPARKING = """
Example 1
Graph id: AadlProjects_smartparking_personallocmanagement_impl_1.aaxl2

Target node name: travnavmapdatabase

Neighbor node names:
- (no neighbors)

Reference Architecture components (labels):
- Presentation_Layer
- Business_Logic_Layer
- Data_Management_Layer
- Security

Output:
{"best":"Data_Management_Layer","ranked":[{"label":"Data_Management_Layer","score":0.91},{"label":"Business_Logic_Layer","score":0.08},{"label":"Security","score":0.01}]}

Example 2
Graph id: AadlProjects_smartparking_personallocmanagement_impl_1.aaxl2

Target node name: driverloc

Neighbor node names:
- system

Reference Architecture components (labels):
- Presentation_Layer
- Business_Logic_Layer
- Data_Management_Layer
- Security

Output:
{"best":"Business_Logic_Layer","ranked":[{"label":"Business_Logic_Layer","score":0.94},{"label":"Data_Management_Layer","score":0.05},{"label":"Presentation_Layer","score":0.01}]}
""".strip()


# DOMAIN -> FEWSHOT MAPPING
_FEWSHOT_OPTION1_BY_DOMAIN = {
    "iot": _FEWSHOT_OPTION1_IOT,
    "autonomous_driving_system": _FEWSHOT_OPTION1_AUTONOMOUS_DRIVING_SYSTEM,
    "smartparking": _FEWSHOT_OPTION1_SMARTPARKING,
}

_FEWSHOT_OPTION2_BY_DOMAIN = {
    "iot": _FEWSHOT_OPTION2_IOT,
    "autonomous_driving_system": _FEWSHOT_OPTION2_AUTONOMOUS_DRIVING_SYSTEM,
    "smartparking": _FEWSHOT_OPTION2_SMARTPARKING,
}


def _normalize_ra_domain(ra_domain: Optional[str]) -> str:
    if not ra_domain:
        return "iot"
    return ra_domain.strip().lower()


def _get_fewshot_option1_for_domain(ra_domain: Optional[str]) -> str:
    domain = _normalize_ra_domain(ra_domain)
    return _FEWSHOT_OPTION1_BY_DOMAIN.get(domain, _FEWSHOT_OPTION1_IOT)


def _get_fewshot_option2_for_domain(ra_domain: Optional[str]) -> str:
    domain = _normalize_ra_domain(ra_domain)
    return _FEWSHOT_OPTION2_BY_DOMAIN.get(domain, _FEWSHOT_OPTION2_IOT)

def format_ra_connections_by_name(
    ra_edges: Set[Tuple[str, str]],
    ra_components: List[RAComponent],
) -> List[str]:
    """
    Convert RA edges from component ids to component names.
    Remove duplicated undirected edges.
    """
    id_to_name = {c.id: c.name for c in ra_components}

    seen = set()
    lines = []

    for src_id, dst_id in ra_edges:
        src_name = id_to_name.get(src_id)
        dst_name = id_to_name.get(dst_id)

        if not src_name or not dst_name:
            continue

        edge_key = tuple(sorted([src_name, dst_name]))
        if edge_key in seen:
            continue

        seen.add(edge_key)
        lines.append(f"- {src_name} -- {dst_name}")

    return sorted(lines)


def build_option1_prompt(
    graph_id: str,
    nodes: List[GraphNode],
    edges: List[GraphEdge],
    ra_components: List[RAComponent],
    ra_edges: Set[Tuple[str, str]],
    use_ra_descriptions: bool,
    use_ra_structural_info: bool,
    ra_descriptions: Dict[str, str],
    ra_domain: Optional[str] = None,
    fewshot_learning: bool = False,
) -> str:
    """Prompt Option 1, node name + RA name (+ optional description + optional RA connections)."""

    id_to_name = {n.id: normalize_name(n.name) for n in nodes}

    lines = []
    lines.append(f"Graph id: {graph_id}")
    lines.append("")

    lines.append("Nodes (names):")
    for n in nodes:
        lines.append(f"- {id_to_name[n.id]}")
    lines.append("")

    lines.append("Edges between node names (undirected view):")
    for e in edges:
        src_name = id_to_name.get(e.source, e.source)
        dst_name = id_to_name.get(e.target, e.target)
        lines.append(f"- {src_name} -- {dst_name}")
    lines.append("")

    lines.append("Reference Architecture components (labels):")
    for c in ra_components:
        if use_ra_descriptions:
            desc = ra_descriptions.get(c.name, "")
            if desc:
                lines.append(f"- {c.name}: {desc}")
            else:
                lines.append(f"- {c.name}")
        else:
            lines.append(f"- {c.name}")
    lines.append("")

    if use_ra_structural_info:
        lines.append("Reference Architecture connections between components:")
        ra_connection_lines = format_ra_connections_by_name(
            ra_edges=ra_edges,
            ra_components=ra_components,
        )

        if ra_connection_lines:
            lines.extend(ra_connection_lines)
        else:
            lines.append("- (no RA connections)")
        lines.append("")

    if fewshot_learning:
        fewshot_text = _get_fewshot_option1_for_domain(ra_domain)
        if fewshot_text:
            lines.append(_FEWSHOT_HEADER)
            lines.append(fewshot_text)
            lines.append("")

    lines.append(
        "Task: For each node name above, choose exactly ONE reference architecture "
        "component (by its NAME) that best matches the role of the node."
    )
    lines.append(
        "Return a JSON object with this structure:\n"
        "{\n"
        '  "mappings": [\n'
        '    {"node_name": "<node name>", "ra_component_name": "<one of the component names above>"},\n'
        "    ...\n"
        "  ]\n"
        "}\n"
        "Ensure every node name from the list appears exactly once in the JSON."
    )

    return "\n".join(lines)


def build_option2_prompt_for_node(
    graph_id: str,
    node: GraphNode,
    neighbor_nodes: List[GraphNode],
    ra_components: List[RAComponent],
    ra_edges: Set[Tuple[str, str]],
    use_ra_descriptions: bool,
    use_ra_structural_info: bool,
    ra_descriptions: Dict[str, str],
    ra_domain: Optional[str] = None,
    fewshot_learning: bool = False,
) -> str:
    """Prompt Option 2, dùng node name + RA name (+ optional description + optional RA connections)."""

    target_name = normalize_name(node.name)
    neighbor_names = sorted({normalize_name(n.name) for n in neighbor_nodes})

    candidate_ra_ids = {c.id for c in ra_components}
    filtered_ra_edges = {
        (src, dst)
        for src, dst in ra_edges
        if src in candidate_ra_ids and dst in candidate_ra_ids
    }

    lines = []
    lines.append(f"Graph id: {graph_id}")
    lines.append("")
    lines.append(f"Target node name: {target_name}")
    lines.append("")

    lines.append("Neighbor node names:")
    if neighbor_names:
        for nn in neighbor_names:
            lines.append(f"- {nn}")
    else:
        lines.append("- (no neighbors)")
    lines.append("")

    lines.append("Reference Architecture component labels:")
    if use_ra_descriptions:
        lines.append(
            "The label name is the text before ':' on each line; the text after ':' "
            "is its description."
        )
        for c in ra_components:
            desc = ra_descriptions.get(c.name, "")
            if desc:
                lines.append(f"- {c.name}: {desc}")
            else:
                lines.append(f"- {c.name}")
    else:
        for c in ra_components:
            lines.append(f"- {c.name}")
    lines.append("")

    if use_ra_structural_info:
        lines.append("Reference Architecture connections involving the candidate labels:")
        ra_connection_lines = format_ra_connections_by_name(
            ra_edges=filtered_ra_edges,
            ra_components=ra_components,
        )

        if ra_connection_lines:
            lines.extend(ra_connection_lines)
        else:
            lines.append("- (no RA connections among the candidate labels)")
        lines.append("")

    if fewshot_learning:
        fewshot_text = _get_fewshot_option2_for_domain(ra_domain)
        if fewshot_text:
            lines.append(_FEWSHOT_HEADER)
            lines.append(fewshot_text)
            lines.append("")

    lines.append(
        "Task: Select EXACTLY ONE best label (component NAME) for the target node.\n"
        "Also provide a ranked list of labels with confidence scores in [0,1].\n"
        "All labels MUST be chosen from the component names above.\n"
        'Return ONLY strict JSON with this schema:\n'
        '{"best":"<label>", "ranked":[{"label":"<label>","score":0.0}, ...]}'
    )

    return "\n".join(lines)



SYSTEM_PROMPT_RECREATE_MAPPING = (
    "You are an expert software architect. "
    "Your task is to select or recreate the best mapping between AADL nodes "
    "and reference architecture components. "
    "You MUST output ONLY a single valid JSON object, without markdown, "
    "code fences, or explanations."
)


def build_recreate_mapping_prompt(
    graph_id: str,
    recent_mappings: List[Dict[str, Optional[str]]],
    nodes: List[GraphNode],
    ra_components: List[RAComponent],
    constraint_feedback: str = "",
) -> str:
    node_items = [
        {
            "id": n.id,
            "name": n.name,
            "normalized_name": normalize_name(n.name),
        }
        for n in nodes
    ]

    ra_items = [
        {
            "id": c.id,
            "name": c.name,
        }
        for c in ra_components
    ]

    payload = {
        "graph_id": graph_id,
        "nodes": node_items,
        "reference_architecture_components": ra_items,
        "recent_mappings": [
            {
                "mapping_index": i + 1,
                "mapping": m,
            }
            for i, m in enumerate(recent_mappings)
        ],
        "latest_constraint_feedback": constraint_feedback or "",
    }

    return (
        "You are given several recent candidate mappings from an AADL model "
        "to a reference architecture.\n\n"
        "Your task is to produce ONE final mapping.\n\n"
        "You may either:\n"
        "1. select the best mapping among the recent mappings, or\n"
        "2. create a better mapping by combining or improving them.\n\n"
        "CRITICAL RULES:\n"
        "- Do NOT rename, rewrite, normalize, translate, or modify any key.\n"
        "- Every key MUST be copied exactly from the provided node ids.\n"
        "- Every non-null value MUST be copied exactly from the provided RA component ids.\n"
        "- Do NOT invent new node ids.\n"
        "- Do NOT invent new RA component ids.\n"
        "- Use null only when no suitable RA component exists.\n"
        "- The output mapping MUST contain all node ids exactly once.\n\n"
        "Return ONLY strict JSON with this schema:\n"
        '{\n'
        '  "mapping": {\n'
        '    "<node_id>": "<ra_component_id_or_null>"\n'
        "  }\n"
        "}\n\n"
        "Input data:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )