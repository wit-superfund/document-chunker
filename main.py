from pathlib import Path
from src.docling import chunk_document, save_chunks, analyze_chunks
from src.init import initialize
from dotenv import load_dotenv
from timeit import timeit
load_dotenv()


def main():
    # Load document
    in_dir = Path("./data/10691613.pdf")
    out_dir = Path("./data/10691613_chunks.md")
    chunks, tokenizer, chunker = chunk_document(
        in_dir, console=console)

    # analyze_chunks(chunks, tokenizer)

    # Save chunks
    save_chunks(chunks, chunker, out_dir, console=console)


if __name__ == "__main__":
    console = initialize()
    console.print(f'[bold red]Document chunked in {timeit(main, number=1):.2f} seconds: [/bold red]', justify='center')    
