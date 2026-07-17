import sys
import json
import difflib
from pathlib import Path

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

def main():
    if len(sys.argv) < 4 or sys.argv[2] != "--markdown":
        print("Usage: python3 scripts/docling-evaluate.py <ground_truth_json> --markdown <markdown_file>")
        sys.exit(1)
        
    gt_path = Path(sys.argv[1])
    md_path = Path(sys.argv[3])
    
    if not gt_path.exists():
        print(f"Error: Ground truth JSON not found: {gt_path}")
        sys.exit(1)
    if not md_path.exists():
        print(f"Error: Markdown file not found: {md_path}")
        sys.exit(1)
        
    # Load JSON and extract text
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    gt_texts = extract_text_from_json(gt_data)
    gt_text = "\n".join([t.strip() for t in gt_texts if t.strip()])
    
    # Load Markdown
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
        
    # Metrics
    ratio = difflib.SequenceMatcher(None, gt_text, md_text).ratio()
    
    # Word-level overlap metrics
    gt_words = set(gt_text.lower().split())
    md_words = set(md_text.lower().split())
    
    intersection = gt_words.intersection(md_words)
    precision = len(intersection) / len(md_words) if md_words else 0
    recall = len(intersection) / len(gt_words) if gt_words else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    
    print(f"=== Quality Evaluation Report ===")
    print(f"Ground Truth JSON: {gt_path.name}")
    print(f"Markdown Output:   {md_path.name}")
    print(f"---------------------------------")
    print(f"Sequence Matcher Similarity Ratio: {ratio:.4f}")
    print(f"Word-level Precision:              {precision:.4f}")
    print(f"Word-level Recall:                 {recall:.4f}")
    print(f"Word-level F1 Score:               {f1:.4f}")
    print(f"=================================")

if __name__ == "__main__":
    main()
