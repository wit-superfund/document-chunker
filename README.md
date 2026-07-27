# Docling PDF Chunker

High-performance PDF document conversion and chunking pipeline built on [Docling](https://github.com/DS4SD/docling). Converts PDF documents into contextualized Markdown chunks (`.md`) optimized for RAG workflows.

## Features

- **Hybrid Chunking**: Uses Docling's `HybridChunker` combined with `sentence-transformers/all-MiniLM-L6-v2` tokenizer for structure-aware text chunking.
- **Multithreaded Processing**: Concurrent PDF conversion and chunking with configurable worker threads.
- **HPC & Slurm Integration**: Batch processing scripts tailored for cluster execution (`gpu-preempt` partitions with offline model execution support).
- **Evaluation Tools**: Structural quality scoring (`scripts/docling-evaluate.py`) and comparative evaluation workflows (`scripts/compare_evaluations.py`).

## Setup & Installation

Requires Python 3.14+ and `uv`.

```bash
uv sync
```

## Usage

### 1. Local Execution

Configure input/output directories in `.env` (defaults to `./data`):

```env
INPUT_DIR=./data
OUTPUT_DIR=./data
```

Run the chunking process:

```bash
uv run main.py
```

### 2. Slurm / HPC Jobs

Submit jobs to the cluster:

```bash
# Main chunking pipeline
sbatch run_document_chunker.slurm
```

## Project Layout

```text
document-chunker/
├── main.py                  # Entry point for chunking pipeline
├── src/
│   ├── docling_tools.py     # Core DocChunker wrapper & PDF text checks
│   └── workers.py           # Parallel worker thread management
└── run_document_chunker.slurm
```
