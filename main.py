import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from rich.console import Console

from src.workers import run_converters
import time

load_dotenv()

IN_DIR = Path(os.getenv("INPUT_DIR", "./data"))
OUT_DIR = Path(os.getenv("OUTPUT_DIR", "./data"))
CSV_DIR: Path = OUT_DIR / "results"
IN_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)


def main():
    start_timer = time.perf_counter()
    results: list[dict[str, Path | float]] = run_converters(IN_DIR)

    elapsed_time = (time.perf_counter() - start_timer) / 60

    df = pd.DataFrame(results)

    df.to_csv(CSV_DIR / "results.csv")

    Console().print(df, justify="center")

    Console().print(
        f"=== Elapsed time: {elapsed_time:.2f} minutes", justify="center")


if __name__ == "__main__":
    main()
