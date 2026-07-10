import os
from pathlib import Path
from src.docling import chunk_document, save_chunks, analyze_chunks, __init__
from src.init import initialize
from dotenv import load_dotenv
from timeit import timeit
load_dotenv()


def main():
    # Load document
    in_dir = Path(os.getenv("INPUT_DIR", "./data"))
    
    converter, chunker, tokenizer = __init__()
    
    for path in in_dir.glob("*.pdf"):
        out_dir = Path(f"./data/{path.name.split('.')[0]}.md")
        chunks = chunk_document(
            path, converter, chunker, console=console)

        # Save chunks
        save_chunks(chunks, chunker, out_dir, console=console)


if __name__ == "__main__":
    console = initialize()
    console.print(f'[bold red]Document chunked in {timeit(main, number=1):.2f} seconds: [/bold red]', justify='center')    
