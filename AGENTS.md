# Docling AI Agents and Advanced Skills Configuration

This document outlines the setup and orchestration of Docling-driven Document Intelligence Agents within this project. Following production constraints, this configuration **does not run locally**; it is designed strictly for execution on the High-Performance Computing (HPC) cluster utilizing isolated project directories and localized weights.

---

## 1. Architectural Strategy & Design Principles

- **No Local Compute / Isolated Environment:** All execution relies entirely on remote cluster nodes (`gpu-preempt` partitions). No external API hits or local machine compute.
- **Strict Project Locality:** All model weights, configurations, datasets, and generated artifacts are strictly contained within the project root folder.
- **Decoupled Conversion & Orchestration:** The heavy structural extraction (GPU-bound VLM processing) is kept isolated from downstream multi-agent text chunking and analysis (CPU-bound workflows).

---

## 2. Environment Setup & Tool Configurations

To prevent workers from attempting to fetch assets or weights from external repositories at runtime, all paths are hardcoded to relative or absolute project directory paths.

### Project Directory Layout
```text
superfund-docling/
├── AGENTS.md
├── main.py
├── src/
│   └── docling.py
│   └── init.py
├── data/
└── .venv/
```

### Dependency Alignment
Ensure your `.venv` contains the necessary advanced intelligence packages without pulling global layers:
```bash
uv pip install docling[vlm] docling-core
```

---

## 3. Evaluation and Performance Verification

To enforce strict quality boundaries on cluster tasks without external dependencies, execute verification sweeps using the local scripts directory:

```bash
# Run structural quality verification on isolated test cases
uv run python3 scripts/docling-evaluate.py     ./data/ground_truth/sample.json     --markdown ./data/output_markdown/sample.md
```

---

## 4. Execution Controls

When launching tasks via Slurm, verify that model downloads do not trigger by forcing offline execution mode within your submission block:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```