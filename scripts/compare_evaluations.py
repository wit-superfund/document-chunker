import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.docling import init_docling
import json
import time
import re
import gc
from collections import Counter

def extract_text_from_json(data):
    """Extract only document text content from DoclingDocument JSON."""
    if isinstance(data, dict) and "texts" in data:
        return [item["text"] for item in data["texts"] if "text" in item]
    # fallback for other JSON structures
    if isinstance(data, list):
        return [item["text"] for item in data if isinstance(item, dict) and "text" in item]
    return []



def clean_text(text: str) -> str:
    """Strip markdown formatting and normalize text for accurate comparison."""
    # Remove markdown links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove headers, bold, italics, code backticks, table dividers
    text = re.sub(r'[#*_`|~]', ' ', text)
    # Replace multiple whitespaces/newlines with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def compute_metrics(gt_text, md_text):
    """Calculate normalized word-level overlap and metrics using Counter."""
    gt_clean = clean_text(gt_text)
    md_clean = clean_text(md_text)

    gt_words = re.findall(r'\w+', gt_clean)
    md_words = re.findall(r'\w+', md_clean)

    gt_counter = Counter(gt_words)
    md_counter = Counter(md_words)

    # Multi-set intersection (accounts for word frequency)
    intersection_counter = gt_counter & md_counter
    intersection_count = sum(intersection_counter.values())

    precision = intersection_count / len(md_words) if md_words else 0
    recall = intersection_count / len(gt_words) if gt_words else 0
    f1 = 2 * precision * recall / \
        (precision + recall) if (precision + recall) else 0

    # Fast Jaccard similarity ratio on word multisets
    union_count = sum((gt_counter | md_counter).values())
    similarity_ratio = intersection_count / union_count if union_count else 0

    return {
        "similarity_ratio": similarity_ratio,
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
    print("=== PICTURES ON vs. PICTURES OFF COMPARISON ===")
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
