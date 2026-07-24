from collections.abc import Iterator
from pathlib import Path

import pymupdf
from docling.chunking import HybridChunker
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    smolvlm_picture_description,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker import BaseChunk
from rich.console import Console
from transformers import AutoTokenizer


def pdf_needs_ocr(pdf_path: Path, min_words: int = 10) -> bool:
    """Return True if the PDF lacks an embedded text layer."""
    doc = pymupdf.open(pdf_path)
    text = "".join(
        str(page.get_text("text")) for page in doc[:3]
    )  # sample first 3 pages
    doc.close()
    return len(text.split()) < min_words


def init_docling(
    max_tokens: int = 512, pictures: bool = True, ocr: bool = False
) -> tuple[DocumentConverter, HybridChunker]:

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = ocr
    if not pictures:
        pipeline_options.do_picture_description = False
        pipeline_options.do_table_structure = False
    else:
        pipeline_options.do_picture_description = True
    pipeline_options.picture_description_options = smolvlm_picture_description
    pipeline_options.picture_description_options.generation_config[
        "repetition_penalty"
    ] = 1.2

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

    chunker = HybridChunker(
        tokenizer=tokenizer, max_tokens=max_tokens, merge_peers=True
    )

    return converter, chunker


def chunk_document(
    file_path: Path,
    converter: DocumentConverter,
    chunker: HybridChunker,
    console: Console = Console(),
) -> list[BaseChunk]:
    """Convert a document into chunks for embedder to read"""

    console.print(f"Processing: {file_path.name}", justify="center")

    console.print("\n\nConverting Document...\n", justify="center")

    result: ConversionResult = converter.convert(file_path)
    doc = result.document

    console.print("Generating Chunks...\n", justify="center")
    chunk_iter: Iterator[BaseChunk] = chunker.chunk(dl_doc=doc)
    chunks: list[BaseChunk] = list(chunk_iter)

    return chunks


def save_chunks(
    chunks: list[BaseChunk],
    chunker: HybridChunker,
    output_path: Path,
    console: Console = Console(),
) -> None:
    """Save chunks to a file"""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"## Chunk {i + 1}\n\n")
            contextualized_text = chunker.contextualize(chunk=chunk)
            f.write(contextualized_text)
            f.write("\n\n")
    console.print(f"\n\nChunks saved to: {output_path}", justify="center")
