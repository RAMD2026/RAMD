"""
Build "mismatched" AADL model inputs for the RA-incompatibility experiment.

This script does NOT touch any file of the main RAMD pipeline. It only reads
existing JSON model files from `mapping_data/` and writes NEW json files
(model graph unchanged, `ra_reference` swapped with a different domain's RA)
into `experiments/ra_incompatibility_test/mismatched_inputs/`.

Scope (narrowed after exploration): earlier rounds tested several domain
pairs (Smart Parking / Autonomous Driving forced onto the IoT RA, and onto a
tightened Smart Parking RA). Only ONE pairing actually produced genuine,
substantive RA-constraint violations under the current (fixed) Java
validator and (tightened) smartparking.ecore: 3 specific IoT models forced
onto the Smart Parking RA. The experiment is now scoped to exactly those 3
models, which is what this script builds.
"""

import copy
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXPERIMENT_DIR / "mismatched_inputs"

PAIR_NAME = "iot_vs_smart_parking"
SOURCE_DOMAIN_FOLDER = "iot_mapping_files"
TARGET_DOMAIN_FOLDER = "SMART_PARKING_mapping_files"

# The 3 IoT models that were found to genuinely violate the (tightened)
# Smart Parking RA's structural/mandatory constraints.
SOURCE_FILES = [
    "mapping_An_AADL_Model_of_Smart_Home_System_MainControlRouter_MainControlRouter_impl_1_aaxl2.json",
    "mapping_An_AADL_Model_of_Smart_Home_System_smart_home_system_smart_home_impl_1_aaxl2.json",
    "mapping_DHsystem_Accounts_DHaccounts_impl_1_aaxl2.json",
]

DOMAIN_META = {
    "iot_mapping_files": {"domain": "iot", "ecore": "iot.ecore", "desc_file": "RA/iot_ra_components.txt"},
    "SMART_PARKING_mapping_files": {"domain": "smart_parking", "ecore": "smartparking.ecore", "desc_file": "RA/smartparking_ra_components.txt"},
    "SELF_DRIVING_CAR_mapping_files": {"domain": "autonomous_driving", "ecore": "autonomous_driving_system.ecore", "desc_file": "RA/autonomous_driving_system_ra_components.txt"},
}


def load_domain_ra_reference(domain_folder: str) -> Dict[str, Any]:
    """Load the (domain-wide, model-independent) ra_reference block once,
    from the first available JSON file of that domain."""
    folder = REPO_ROOT / "mapping_data" / domain_folder
    files = sorted(glob.glob(str(folder / "*.json")))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {folder}")
    with open(files[0], "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["ra_reference"]


def build_pair() -> List[str]:
    source_files = [
        REPO_ROOT / "mapping_data" / SOURCE_DOMAIN_FOLDER / name for name in SOURCE_FILES
    ]
    missing = [str(p) for p in source_files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected source file(s): {missing}")

    target_ra_reference = load_domain_ra_reference(TARGET_DOMAIN_FOLDER)
    target_meta = DOMAIN_META[TARGET_DOMAIN_FOLDER]
    source_meta = DOMAIN_META[SOURCE_DOMAIN_FOLDER]

    pair_out_dir = OUTPUT_DIR / PAIR_NAME
    pair_out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for src_path in source_files:
        with open(src_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        mismatched = copy.deepcopy(data)
        # Keep nodes/edges/graph_id/meta from the ORIGINAL model untouched.
        # Only the RA side is swapped to a different, incompatible domain.
        mismatched["ra_reference"] = copy.deepcopy(target_ra_reference)
        mismatched["_mismatch_meta"] = {
            "purpose": "RA-incompatibility test",
            "source_domain": source_meta["domain"],
            "source_file": os.path.basename(src_path),
            "injected_ra_domain": target_meta["domain"],
            "injected_ra_ecore_file": target_meta["ecore"],
            "injected_ra_description_file": target_meta["desc_file"],
            "note": (
                "The 'ra_mappings' fields inside 'nodes' still refer to the "
                "ORIGINAL (correct) RA and are kept only for traceability; "
                "they are NOT valid ground truth for the injected RA and "
                "must not be used for accuracy scoring in this experiment."
            ),
        }

        out_name = os.path.basename(src_path)
        out_path = pair_out_dir / out_name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(mismatched, f, indent=2, ensure_ascii=False)
        written.append(str(out_path))

    return written


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = build_pair()
    manifest = {
        PAIR_NAME: {
            "source_domain": DOMAIN_META[SOURCE_DOMAIN_FOLDER]["domain"],
            "injected_ra_domain": DOMAIN_META[TARGET_DOMAIN_FOLDER]["domain"],
            "injected_ra_ecore_file": DOMAIN_META[TARGET_DOMAIN_FOLDER]["ecore"],
            "injected_ra_description_file": DOMAIN_META[TARGET_DOMAIN_FOLDER]["desc_file"],
            "files": [os.path.basename(p) for p in written],
        }
    }
    print(f"[{PAIR_NAME}] wrote {len(written)} mismatched files -> {OUTPUT_DIR / PAIR_NAME}")

    with open(OUTPUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Manifest written to {OUTPUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
