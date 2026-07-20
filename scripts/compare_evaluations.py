import sys
import json
import time
import difflib
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.docling import init_docling
import gc
import torch


def extract_text_from_json(data):
    """Recursively extract all text values from a JSON structure."""
    if isinstance(data, str):
        return [data]
    elif isinstance(data, list):
        texts = []
        for item in data:
            texts.extend(extract_text_from_json(item))
        return texts
    elif isinstance(data, dict):
        if "text" in data and isinstance(data["text"], str):
            return [data["text"]]
        texts = []
        for val in data.values():
            texts.extend(extract_text_from_json(val))
        return texts
    return []


def compute_metrics(gt_text, md_text):
    """Calculate SequenceMatcher similarity and word-level overlap F1 score."""
    ratio = difflib.SequenceMatcher(None, gt_text, md_text).ratio()
    gt_words = set(gt_text.lower().split())
    md_words = set(md_text.lower().split())

    intersection = gt_words.intersection(md_words)
    precision = len(intersection) / len(md_words) if md_words else 0
    recall = len(intersection) / len(gt_words) if gt_words else 0
    f1 = 2 * precision * recall / \
        (precision + recall) if (precision + recall) else 0

    return {
        "similarity_ratio": ratio,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 scripts/compare_evaluations.py <path_to_pdf> <ground_truth_json>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    gt_path = Path(sys.argv[2])

    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    if not gt_path.exists():
        print(f"Error: Ground truth JSON not found: {gt_path}")
        sys.exit(1)

    # Load and extract Ground Truth text
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    gt_texts = extract_text_from_json(gt_data)
    gt_text = "\n".join([t.strip() for t in gt_texts if t.strip()])

    # 1. Evaluate with Pictures OFF
    print("Converting with Pictures OFF...")
    conv_off, _, _ = init_docling(pictures=False)
    start_off = time.perf_counter()
    res_off = conv_off.convert(pdf_path)
    md_off = res_off.document.export_to_markdown()
    elapsed_off = time.perf_counter() - start_off
    metrics_off = compute_metrics(gt_text, md_off)

    del conv_off, res_off
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Evaluate with Pictures ON
    print("Converting with Pictures ON...")
    conv_on, _, _ = init_docling(pictures=True)
    start_on = time.perf_counter()
    res_on = conv_on.convert(pdf_path)
    md_on = res_on.document.export_to_markdown()
    elapsed_on = time.perf_counter() - start_on
    metrics_on = compute_metrics(gt_text, md_on)

    # Print comparative report
    print("\n" + "="*50)
    print("=== PICTUES ON vs. PICTURES OFF COMPARISON ===")
    print("="*50)
    print(f"PDF Document: {pdf_path.name}")
    print(f"Ground Truth: {gt_path.name}")
    print("-"*50)
    print(f"Metric                  | Pictures OFF | Pictures ON")
    print(f"--------------------------------------------------")
    print(
        f"Elapsed Time (s)        | {elapsed_off:12.2f} | {elapsed_on:11.2f}")
    print(
        f"Similarity Ratio        | {metrics_off['similarity_ratio']:12.4f} | {metrics_on['similarity_ratio']:11.4f}")
    print(
        f"Word-Level Precision    | {metrics_off['precision']:12.4f} | {metrics_on['precision']:11.4f}")
    print(
        f"Word-Level Recall       | {metrics_off['recall']:12.4f} | {metrics_on['recall']:11.4f}")
    print(
        f"Word-Level F1 Score     | {metrics_off['f1']:12.4f} | {metrics_on['f1']:11.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
