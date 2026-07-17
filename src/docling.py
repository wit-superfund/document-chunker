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


def init_docling(max_tokens: int = 512):
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False 
    pipeline_options.do_picture_description = False  
    pipeline_options.do_table_structure = False     
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
