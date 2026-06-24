# graph_utils.py
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class RAComponent:
    id: str
    name: str


@dataclass
class GraphNode:
    id: str
    name: str


@dataclass
class GraphEdge:
    source: str
    target: str


def load_mapping_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_ra_components(mapping_data: Dict[str, Any]) -> List[RAComponent]:
    comps = []
    for c in mapping_data["ra_reference"]["components"]:
        comps.append(RAComponent(id=c["id"], name=c["name"]))
    return comps

def get_ra_parent_map(mapping_data: Dict[str, Any]) -> Dict[str, Optional[str]]:
    parent_map = {}
    for c in mapping_data["ra_reference"]["components"]:
        cid = c["id"]
        parent_map[cid] = c.get("parent_id")
    return parent_map


def get_ra_edge_set(mapping_data: Dict[str, Any]) -> set:
    """Build undirected edge set between RA components (by id)."""
    edges = set()
    for conn in mapping_data["ra_reference"]["connectors"]:
        s = conn["source"]
        t = conn["target"]
        edges.add((s, t))
        edges.add((t, s))
    return edges


def get_graph_nodes(mapping_data: Dict[str, Any]) -> List[GraphNode]:
    nodes = []
    for n in mapping_data["nodes"]:
        nodes.append(GraphNode(id=n["id"], name=n["name"]))
    return nodes


def get_graph_edges(mapping_data: Dict[str, Any]) -> List[GraphEdge]:
    edges = []
    for e in mapping_data["edges"]:
        edges.append(GraphEdge(source=e["source"], target=e["target"]))
    return edges


def build_neighbors(edges: List[GraphEdge]) -> Dict[str, List[str]]:
    """Neighbors for each node id (undirected)."""
    nbrs: Dict[str, List[str]] = {}
    for e in edges:
        nbrs.setdefault(e.source, []).append(e.target)
        nbrs.setdefault(e.target, []).append(e.source)
    return nbrs
