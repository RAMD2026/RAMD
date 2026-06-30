# RAMD: Reference Architecture Mapping Discovery

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20829399.svg)](https://doi.org/10.5281/zenodo.20829399)

Replication package for the paper:

> **Discovering the Relation Between Architectural Models and Reference Architectures**  

---

## Overview

RAMD is an LLM-based pipeline that automatically maps components of an AADL architectural model to components in a Reference Architecture (RA). It combines:

- **Model preprocessing** — strips noise from AADL models and retains semantically meaningful elements.
- **Retrieval-Augmented Generation (RAG)** — embeds RA component descriptions and historical mappings into a vector index to provide relevant context to the LLM.
- **Structured LLM prompting** — two prompting strategies: map all nodes at once (Option 1) or one node at a time (Option 2).
- **RA compliance-guided iterative refinement** — an external checker validates the candidate mapping against RA structural constraints and returns violations to the LLM as feedback for the next iteration.

We evaluate RAMD on **64 open-source AADL models** spanning three Reference Architectures from the IoT, Autonomous Driving, and Smart Parking domains. RAMD achieves **85.8% mapping accuracy**, outperforming the strongest baseline by **44.6 percentage points**.

---

## Repository Structure

```
RAComponentMapping/
├── main.py                      # Entry point — orchestrates the mapping pipeline
├── mapping_core.py              # Core logic for Option 1 and Option 2 mapping loops
├── prompts.py                   # Prompt builders and few-shot examples
├── llm_clients.py               # LLM client wrappers (OpenAI, etc.)
├── graph_utils.py               # AADL model and RA graph loading utilities
├── semantic_utils.py            # RAG: semantic indexing and retrieval
├── scoring.py                   # Accuracy computation
├── config_utils.py              # Configuration loading
├── text_utils.py                # Text normalization utilities
├── LLM_mapping_RA.yml           # Main configuration file (edit this to run experiments)
│
├── RA/                          # Reference Architecture definitions
│   ├── iot.ecore                # RA structural model — IoT Device System
│   ├── iot_ra_components.txt    # RA textual specification — IoT Device System
│   ├── smartparking.ecore       # RA structural model — Smart Parking System
│   ├── smartparking_ra_components.txt
│   ├── autonomous_driving_system.ecore
│   └── autonomous_driving_system_ra_components.txt
│
├── mapping_data/                # Input data and RAG knowledge bases
│   ├── iot_mapping_files/           # 25 IoT AADL models (JSON)
│   ├── SMART_PARKING_mapping_files/ # 29 Smart Parking AADL models (JSON)
│   ├── SELF_DRIVING_CAR_mapping_files/ # 10 Autonomous Driving AADL models (JSON)
│   ├── iot_ra_component_mapping_labels.csv        # Historical mappings for RAG (IoT)
│   ├── all_SMART_PARKING_connection_components_with_ra_component.csv
│   └── all_SELF_DRIVING_CAR_connection_components_with_ra_component.csv
│
├── baselines/                   # Baseline implementations
│   ├── tfidf_lexical_matching.py    # TF-IDF lexical baseline
│   ├── sentence_embedding_matching.py # Sentence embedding baseline
│   └── strutural_matching.py        # Graph-structural baseline
│
├── results/                     # Output directory (created at runtime)
│   ├── compute_cliffs_delta.py  # Statistical analysis script
│   └── plot/                    # Plot generation scripts (RQ2, RQ4)
│
├── java/                        # RA compliance checker (Epsilon/EMF)
│   ├── bin/                     # Compiled Java classes and metamodels
│   └── lib/                     # JAR dependencies (EMF, Epsilon)
│
├── PROMPT_ALL (EXAMPLE)         # Example prompt for Option 1 (map all nodes)
└── PROMPT_COMP (EXAMPLE)        # Example prompt for Option 2 (map one node)
```

---

## Requirements

### Python

- Python 3.9+
- Install dependencies:

```bash
pip install openai sentence-transformers scikit-learn numpy pyyaml
```

### Java (RA Compliance Checker)

The compliance checker uses Eclipse Epsilon and EMF. The required JARs are bundled in `java/lib/`.

Compile the Java sources:

```bash
cd java
javac -cp "lib/*" -d bin src/it/uni/sim/architecturemodeling/launcher/**/*.java
```

Requires Java 11+.

---

## Configuration

All experiment settings are controlled by [`LLM_mapping_RA.yml`](LLM_mapping_RA.yml). Key options:

| Parameter | Description |
|---|---|
| `ra_ecore_file` | Path to the RA structural model (`.ecore`) |
| `input_folder` | Folder containing input AADL models (`.json`) |
| `ra_description_file` | Path to the RA textual specification |
| `option` | `1` = map all nodes at once; `2` = map one node at a time |
| `check_RA_constraints` | Enable RA compliance-guided iterative refinement |
| `fewshot_learning` | Include few-shot examples in prompts |
| `use_ra_descriptions` | Append RA component descriptions to prompts |
| `use_semantic_filter` | Enable RAG (requires `mapping_knowledge_csv_path`) |
| `mapping_knowledge_csv_path` | CSV of historical mappings for RAG retrieval |
| `max_iterations` | Maximum refinement iterations per model |
| `models[].provider` | LLM provider (`openai`) |
| `models[].model` | Model identifier (e.g., `gpt-4o`, `gpt-4o-mini`) |

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY=<your-key>
```

---

## Running RAMD

### Full pipeline (RAMD — best configuration)

Edit `LLM_mapping_RA.yml` to select the target domain, then run:

```bash
python main.py
```

Results are written to `results/<run_folder>/`.

### Example: IoT domain with Option 2 + RAG + Constraints

```yaml
data:
  ra_ecore_file: "RA/iot.ecore"
  input_folder: "mapping_data/iot_mapping_files"
  ra_description_file: "RA/iot_ra_components.txt"
  output_folder: "results"

mapping:
  option: 2
  check_RA_constraints: true
  fewshot_learning: true
  use_ra_descriptions: true
  use_semantic_filter: true
  mapping_knowledge_csv_path: "mapping_data/iot_ra_component_mapping_labels.csv"
  max_iterations: 30

models:
  - name: "chatgpt"
    provider: "openai"
    model: "gpt-4o-mini"
    temperature: 0
```

### Running Baselines

```bash
python baselines/tfidf_lexical_matching.py
python baselines/sentence_embedding_matching.py
python baselines/strutural_matching.py
```

---

## Dataset

The AADL models used in our experiments are drawn from a curated, publicly available dataset:

> **Mining Architectural Models in the Wild: A Curated Dataset of AADL Models**  
> GitHub: [https://github.com/dinhtranthi/aadl-architectural-models-dataset](https://github.com/dinhtranthi/aadl-architectural-models-dataset)

The pre-processed JSON representations of these models — with ground truth mappings between AADL components and RA components — are included in `mapping_data/`.

---

## Data Format

Each input file in `mapping_data/*/` is a JSON file representing one AADL model graph together with its ground truth mappings to RA components, with the following structure:

```json
{
  "graph_id": "<model_filename>",
  "nodes": [
    { "id": "<node_id>", "name": "<component_name>", "category": "<aadl_type>" }
  ],
  "edges": [
    { "source": "<node_id>", "target": "<node_id>" }
  ],
  "ra_components": [
    { "id": "<ra_id>", "name": "<ra_component_name>" }
  ],
  "ra_edges": [["<ra_id>", "<ra_id>"], ...],
  "ground_truth": { "<node_id>": "<ra_component_id>" }
}
```

---

## Output Format

Each model result is saved as a JSON file in `results/<run_folder>/thread1/`:

```json
{
  "graph_id": "...",
  "predicted_mapping": { "<node_name>": "<ra_component_name>" },
  "total_score": 0.85,
  "avg_score": 0.85,
  "per_node_scores": { "<node_name>": 1.0 },
  "check_constraints_log": "...",
  "execution_time_seconds": 12.34
}
```

A `summary_all_threads.json` with aggregate micro/macro accuracy scores is written to the run folder.

---

## Prompt Examples

- [`PROMPT_ALL (EXAMPLE)`](PROMPT_ALL%20(EXAMPLE)) — illustrates the prompt sent to the LLM in Option 1 (all nodes mapped in one call), including graph structure, RA component descriptions, few-shot examples, and constraint feedback.
- [`PROMPT_COMP (EXAMPLE)`](PROMPT_COMP%20(EXAMPLE)) — illustrates the prompt sent to the LLM in Option 2 (one target node mapped per call).

---

## Reproducing Paper Results

To reproduce Table II (RQ3 — overall accuracy):

1. Run RAMD with `option: 2`, `check_RA_constraints: true`, `fewshot_learning: true`, `use_ra_descriptions: true`, `use_semantic_filter: true` for each of the three domains.
2. Run each baseline script for the same input folders.
3. Compare `avg_macro_score` values in the generated `summary_all_threads.json` files.

Statistical tests (Cliff's delta) can be reproduced with:

```bash
python results/compute_cliffs_delta.py
```

---

## License

This replication package is released for research purposes.
