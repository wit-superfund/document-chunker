import os
import time
from pathlib import Path
from src.docling import chunk_document, save_chunks, init_docling
from src.init import initialize
from dotenv import load_dotenv

load_dotenv()


def main(console):
    in_dir = Path(os.getenv("INPUT_DIR", "./data"))
    
    # 1. Pictures OFF
    console.print("\n[bold yellow]--- Running with Pictures OFF ---[/bold yellow]", justify='center')
    start_off = time.perf_counter()
    converter, chunker, tokenizer = init_docling(pictures=False)
    for path in in_dir.glob("*.pdf"):
        out_dir = Path(f"./data/without_picture_descr/{path.name.split('.')[0]}_off.md")
        chunks = chunk_document(path, converter, chunker, console=console)
        save_chunks(chunks, chunker, out_dir, console=console)
    elapsed_off = time.perf_counter() - start_off

    # 2. Pictures ON
    console.print("\n[bold yellow]--- Running with Pictures ON ---[/bold yellow]", justify='center')
    start_on = time.perf_counter()
    converter, chunker, tokenizer = init_docling(pictures=True)
    for path in in_dir.glob("*.pdf"):
        out_dir = Path(f"./data/with_picture_descr/{path.name.split('.')[0]}_on.md")
        chunks = chunk_document(path, converter, chunker, console=console)
        save_chunks(chunks, chunker, out_dir, console=console)
    elapsed_on = time.perf_counter() - start_on

    # Results Summary
    console.print("\n[bold green]=== Performance Comparison ===[/bold green]", justify='center')
    console.print(f"Pictures OFF: [bold cyan]{elapsed_off:.2f} seconds[/bold cyan]", justify='center')
    console.print(f"Pictures ON : [bold cyan]{elapsed_on:.2f} seconds[/bold cyan]", justify='center')
    console.print(f"Difference  : [bold red]{elapsed_on - elapsed_off:+.2f} seconds[/bold red]", justify='center')


if __name__ == "__main__":
    console = initialize()
    main(console)
