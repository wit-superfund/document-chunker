import os
import time
from pathlib import Path
from src.docling import chunk_document, save_chunks, init_docling
from src.init import initialize
from dotenv import load_dotenv

load_dotenv()

def main(console):
    # Load document
    in_dir = Path(os.getenv("INPUT_DIR", "./data"))
    
    converter, chunker, tokenizer = init_docling()
    
    for path in in_dir.glob("*.pdf"):
        out_dir = Path(f"./data/{path.name.split('.')[0]}.md")
        chunks = chunk_document(
            path, converter, chunker, console=console)

        # Save chunks
        save_chunks(chunks, chunker, out_dir, console=console)


if __name__ == "__main__":
    console = initialize()
    start_time = time.perf_counter()
    main(console)
    elapsed = time.perf_counter() - start_time
    console.print(f'[bold red]Document chunked in {elapsed:.2f} seconds[/bold red]', justify='center')

