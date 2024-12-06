from rag_demo.preprocessing import (
    convert_pdf_to_text,
    load_to_vector_db,
    chunk_and_embed,
)
from loguru import logger


def process_pdf(file_path: str):
    convert = convert_pdf_to_text([file_path])
    embedded_chunks = chunk_and_embed([convert])
    load_to_vector_db(embedded_chunks)
    return True
