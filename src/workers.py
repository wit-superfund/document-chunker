from concurrent.futures import ThreadPoolExecutor, as_completed
from docling_tools import init_docling, chunk_document, save_chunks
import time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import os
import threading

FILE_LOCK = threading.Lock()
SOURCE_LOCK = threading.Lock()
IN_DIR = Path(os.getenv("INPUT_DIR", "./data"))
OUT_DIR = Path(os.getenv("OUTPUT_DIR", "./data"))
IN_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


load_dotenv()

# TODO: Move initialization to before workers get deployed, ensure files cannot be accessed by multiple workers.
# Create .slurm file.
# Scale up # of files.


def worker(document: Path, pictures_on: bool = False, ocr_on: bool = False):
    # Initialize docling based on inputs
    # Convert and export to .md
    # Repeat for rest of the directory
    out_path = OUT_DIR / f"{document.name.split('.')[0]}.md"

    start_timer = time.perf_counter()

    conv, chunker = init_docling(pictures=pictures_on, ocr=ocr_on)

    init_end = time.perf_counter()

    chunks = chunk_document(document, conv, chunker)

    chunker_end = time.perf_counter()

    save_chunks(chunks, chunker, out_path)

    elapsed = time.perf_counter()
    elapsed_document = elapsed - start_timer
    elapsed_init = init_end - start_timer
    elapsed_convert = chunker_end - init_end
    elapsed_save = elapsed - chunker_end

    return {
        "path": document,
        "output": out_path,
        "total_time": elapsed_document,
        "init_time": elapsed_init,
        "convert_time": elapsed_convert,
        "save_time": elapsed_save,
    }


def run_converters(in_dir: Path):
    future_res = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(worker, path, pictures_on=True, ocr_on=False)
            for path in in_dir.rglob("*.pdf")
        }
    for future in as_completed(futures):
        result = future.result()
        future_res.append(result)

    return future_res


if __name__ == "__main__":
    results = run_converters(IN_DIR)

    df = pd.DataFrame(results)

    print(df)
