from src.docling import chunk_document, save_chunks, analyze_chunks
from pathlib import Path


def main():
    # Load document
    chunks, tokenizer, chunker = chunk_document(
        Path("./data/10691613.pdf"))

    # analyze_chunks(chunks, tokenizer)

    # Save chunks
    save_chunks(chunks, chunker, Path("./data/10691613_chunks.md"))


if __name__ == "__main__":
    main()
