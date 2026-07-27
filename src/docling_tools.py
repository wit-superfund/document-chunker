from collections.abc import Generator
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
        self.pipeline_options.do_table_structure = False
        if not pictures:
            self.pipeline_options.do_picture_description = False
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

    PAGE_BATCH_SIZE = 100  # pages per conversion batch for large docs

    def _iter_chunks_fast(self, file_path: Path) -> Generator[str, None, None]:
        """Fast path: extract text via pymupdf and chunk by token count. No ML models."""
        doc = pymupdf.open(file_path)
        buf: list[str] = []
        buf_tokens = 0
        for page in doc:
            raw_text = page.get_text("text")
            text = (raw_text if isinstance(raw_text, str) else " ".join(map(str, raw_text)) if isinstance(raw_text, list) else raw_text.get("text", "") if isinstance(raw_text, dict) else str(raw_text)).strip()
            if not text:
                continue
            tokens = len(self.tokenizer.encode(text, add_special_tokens=False)) if self.tokenizer else len(text.split())
            if buf_tokens + tokens > self.chunker.max_tokens and buf:
                yield " ".join(buf)
                buf, buf_tokens = [text], tokens
            else:
                buf.append(text)
                buf_tokens += tokens
        if buf:
            yield " ".join(buf)
        doc.close()

    def _iter_chunks(self, file_path: Path) -> Generator[BaseChunk, None, None]:
        """Yield chunks: fast pymupdf path for text-layer PDFs, Docling for OCR."""
        if not pdf_needs_ocr(file_path):
            self.console.print(f"  [fast] text-layer PDF: {file_path.name}", justify="center")
            yield from self._iter_chunks_fast(file_path)  # type: ignore[misc]
            return
        total_pages = pymupdf.open(file_path).page_count
        for start in range(0, total_pages, self.PAGE_BATCH_SIZE):
            end = min(start + self.PAGE_BATCH_SIZE - 1, total_pages - 1)
            self.console.print(
                f"  [docling] Converting pages {start + 1}–{end + 1} / {total_pages}",
                justify="center",
            )
            result: ConversionResult = self.converter.convert(
                file_path, page_range=(start + 1, end + 1)
            )
            yield from self.chunker.chunk(dl_doc=result.document)

    def chunk_document(self, file_path: Path) -> list[BaseChunk]:
        self.console.print(f"Processing: {file_path.name}", justify="center")
        self.console.print("\n\nConverting Document...\n", justify="center")
        return list(self._iter_chunks(file_path))

    def save_chunks_streaming(self, output_path: Path, file_path: Path) -> int:
        """Convert, chunk, and write in one streaming pass — no full chunk list in RAM."""
        self.console.print(f"Saving chunks to: {output_path}", justify="center")
        i = -1
        with open(output_path, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(self._iter_chunks(file_path)):
                text = chunk if isinstance(chunk, str) else self.chunker.contextualize(chunk=chunk)
                f.write(f"## Chunk{i + 1} \n\n{text}\n\n")
        self.console.print(f"Chunks saved to: {output_path}", justify="center")
        return i + 1

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
