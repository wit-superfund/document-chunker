import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from docling_core.transforms.chunker import BaseChunk
from dotenv import load_dotenv

from src.docling_tools import DocChunker

load_dotenv()


# Ensure directories exist
IN_DIR = Path(os.getenv("INPUT_DIR", "./data"))
OUT_DIR = Path(os.getenv("OUTPUT_DIR", "./data"))
IN_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def worker(
    doc_chunker: DocChunker,
    document: Path,
) -> dict[str, Path | float]:
    out_path: Path = OUT_DIR / document.relative_to(IN_DIR).with_suffix(".md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    """ Instructions for fututre worker sent to convert .pdf's"""

    start_timer = time.perf_counter()

    doc_chunker.save_chunks_streaming(out_path, document)

    elapsed = time.perf_counter()
    elapsed_document = elapsed - start_timer

    return {
        "path": document,
        "output": out_path,
        "total_time": elapsed_document,
    }


def run_converters(
    in_dir: Path, do_ocr: bool = False, do_pictures: bool = False
) -> list[dict[str, Path | float]]:
    """ Creates list of documents in a directory, sends workers to convert found documents to .md """
    future_res: list[dict[str, Path | float]] = []
    n_workers = 16
    # Create DocChunker instances for amount n_workers
    chunkers = [DocChunker(pictures=do_pictures, ocr=do_ocr)
                for _ in range(n_workers)]
    files = list(in_dir.rglob("*.pdf"))

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # currently, chunkers[i % n_workers] may slow down program if there are numerous large documents that have the same (i % n_workers)
        futures = {
            executor.submit(worker, chunkers[i % n_workers], f) 
            for i, f in enumerate(files)
        }
        for future in as_completed(futures):
            # Save result of each worker and combine to single dict
            result: dict[str, Path | float] = future.result()
            future_res.append(result)

    return future_res


if __name__ == "__main__":
    results: list[dict[str, Path | float]] = run_converters(IN_DIR)

    df = pd.DataFrame(results)

    print(df)
