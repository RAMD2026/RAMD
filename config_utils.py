# config_utils.py
import yaml
import os
from typing import Dict, Any

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_ra_descriptions(path: str) -> Dict[str, str]:
    """
    Read file TXT describe RA Components.
    Format: '<RA Name>: <description>'.
    """
    desc_map: Dict[str, str] = {}
    if not path:
        return desc_map
    if not os.path.exists(path):
        print(f"[WARN] RA description file not found: {path}")
        return desc_map

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ":" not in line:
                continue
            name, desc = line.split(":", 1)
            name = name.strip()
            desc = desc.strip()
            if name:
                desc_map[name] = desc

    return desc_map
