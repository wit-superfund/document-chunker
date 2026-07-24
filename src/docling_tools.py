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


class DocChunker:
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    console = Console()

    def __init__(
        self, max_tokens: int = 512, ocr: bool = False, pictures: bool = False
    ) -> None:
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr = ocr
        if not pictures:
            self.pipeline_options.do_picture_description = False
            self.pipeline_options.do_table_structure = False
        else:
            self.pipeline_options.do_picture_description = True
            self.pipeline_options.picture_description_options = (
                smolvlm_picture_description
            )
            self.pipeline_options.picture_description_options.generation_config[
                "repetition_penalty"
            ] = 1.2

        self.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.AUTO)
        self.pipeline_options.accelerator_options = self.accelerator_options

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
            }
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, model_max_length=8192
        )

        self.chunker = HybridChunker(
            tokenizer=self.tokenizer, max_tokens=max_tokens, merge_peers=True
        )

    def chunk_document(self, file_path: Path) -> list[BaseChunk]:
        self.console.print(f"Processing: {file_path.name}", justify="center")

        self.console.print("\n\nConverting Document...\n", justify="center")

        result: ConversionResult = self.converter.convert(file_path)
        doc = result.document

        chunk_iter: Iterator[BaseChunk] = self.chunker.chunk(dl_doc=doc)
        chunks: list[BaseChunk] = list(chunk_iter)

        return chunks

    def save_chunks(self, output_path: Path, chunks):
        self.console.print(f"Saving chunks to: {output_path}", justify="center")
        with open(output_path, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                f.write(f"## Chunk{i + 1} \n\n")
                contextualized_text: str = self.chunker.contextualize(chunk=chunk)
                f.write(contextualized_text)
                f.write("\n\n")
        self.console.print(f"Chunks saved to: {output_path}", justify="center")


def pdf_needs_ocr(pdf_path: Path, min_words: int = 10) -> bool:
    """Return True if the PDF lacks an embedded text layer."""
    doc = pymupdf.open(pdf_path)
    text = "".join(
        str(page.get_text("text")) for page in doc[:3]
    )  # sample first 3 pages
    doc.close()
    return len(text.split()) < min_words
