import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from docling_core.transforms.chunker import BaseChunk
from dotenv import load_dotenv

from src.docling_tools import DocChunker

IN_DIR = Path(os.getenv("INPUT_DIR", "./data"))
OUT_DIR = Path(os.getenv("OUTPUT_DIR", "./data"))
IN_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


load_dotenv()


def worker(
    doc_chunker: DocChunker,
    document: Path,
) -> dict[str, Path | float]:
    out_path: Path = OUT_DIR / document.relative_to(IN_DIR).with_suffix(".md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start_timer = time.perf_counter()

    chunks: list[BaseChunk] = doc_chunker.chunk_document(document)

    chunker_end = time.perf_counter()

    doc_chunker.save_chunks(out_path, chunks)

    elapsed = time.perf_counter()
    elapsed_document = elapsed - start_timer
    elapsed_convert = chunker_end - start_timer
    elapsed_save = elapsed - chunker_end

    return {
        "path": document,
        "output": out_path,
        "total_time": elapsed_document,
        "convert_time": elapsed_convert,
        "save_time": elapsed_save,
    }


def run_converters(
    in_dir: Path, do_ocr: bool = False, do_pictures: bool = False
) -> list[dict[str, Path | float]]:
    future_res: list[dict[str, Path | float]] = []
    n_workers = 4
    chunkers = [DocChunker(pictures=do_pictures, ocr=do_ocr) for _ in range(n_workers)]
    files = list(in_dir.rglob("*.pdf"))

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(worker, chunkers[i % n_workers], f)
            for i, f in enumerate(files)
        }
        for future in as_completed(futures):
            result: dict[str, Path | float] = future.result()
            future_res.append(result)

    return future_res


if __name__ == "__main__":
    results: list[dict[str, Path | float]] = run_converters(IN_DIR)

    df = pd.DataFrame(results)

    print(df)
