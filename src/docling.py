from transformers.models.auto.tokenization_auto import SentencePieceBackend
from docling.document_converter import PdfFormatOption
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from transformers import AutoTokenizer
from pathlib import Path
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, smolvlm_picture_description
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from rich.console import Console


def __init__(max_tokens: int = 512):

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_picture_description = True
    pipeline_options.picture_description_options = smolvlm_picture_description
    pipeline_options.picture_description_options.generation_config["repetition_penalty"] = 1.2

    # GPU accelerator
    accelerator_options = AcceleratorOptions(device=AcceleratorDevice.AUTO)
    pipeline_options.accelerator_options = accelerator_options

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=8192)

    chunker = HybridChunker(tokenizer=tokenizer,
                            max_tokens=max_tokens, merge_peers=True)

    return converter, chunker, tokenizer


def chunk_document(file_path: Path, converter, chunker, console: Console = Console()):
    """Convert a document into chunks for embedder to read"""

    console.print(f'Processing: {file_path.name}',
                  justify='center', style='bold magenta')

    console.print("\n\nConverting Document...\n",
                  style="bold blue", justify='center')

    result = converter.convert(file_path)
    doc = result.document

    console.print("Generating Chunks...\n",
                  style="bold blue", justify='center')
    chunk_iter = chunker.chunk(dl_doc=doc)
    chunks = list(chunk_iter)

    return chunks


def analyze_chunks(chunks, tokenizer):
    """Analyze chunks and return embeddings"""
    print("\n" + "=" * 60)
    print("CHUNK ANALYSIS")
    print("=" * 60)

    total_tokens = 0
    chunk_sizes = []

    for i, chunk in enumerate(chunks):
        # Get text content
        text = chunk.text
        tokens = tokenizer.encode(text)
        token_count = len(tokens)

        total_tokens += token_count
        chunk_sizes.append(token_count)

        # Display first 3 chunks in detail
        if i < 3:
            print(f"\n--- Chunk {i} ---")
            print(f"Tokens: {token_count}")
            print(f"Characters: {len(text)}")
            print(f"Preview: {text[:150]}...")

            # Show metadata if available
            if hasattr(chunk, 'meta') and chunk.meta:
                print(f"Metadata: {chunk.meta}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total chunks: {len(chunks)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Average tokens per chunk: {total_tokens / len(chunks):.1f}")
    print(f"Min tokens: {min(chunk_sizes)}")
    print(f"Max tokens: {max(chunk_sizes)}")

    # Token distribution
    print(f"\nToken distribution:")
    ranges = [(0, 128), (128, 256), (256, 384), (384, 512)]
    for start, end in ranges:
        count = sum(1 for size in chunk_sizes if start <= size < end)
        print(f"  {start}-{end} tokens: {count} chunks")


def save_chunks(chunks, chunker, output_path: Path, console: Console = Console()):
    """Save chunks to a file"""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"## Chunk {i+1}\n\n")
            contextualized_text = chunker.serialize(chunk=chunk)
            f.write(contextualized_text)
            f.write("\n\n")
    console.print(f'\n\nChunks saved to: {output_path}', style='bold green', justify='center')
    return None
