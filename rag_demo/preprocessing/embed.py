from typing_extensions import Annotated
from typing import Generator
from .base import Chunk
from .base import EmbeddedChunk
from .chunking import chunk_text
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv
from uuid import uuid4
from loguru import logger

load_dotenv()


def batch(list_: list, size: int) -> Generator[list, None, None]:
    yield from (list_[i : i + size] for i in range(0, len(list_), size))


def embed_chunks(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    api = InferenceClient(
        model="intfloat/multilingual-e5-large-instruct",
        token=os.getenv("HF_API_TOKEN"),
    )
    logger.info(f"Embedding {len(chunks)} chunks")
    embedded_chunks = []
    for chunk in chunks:
        try:
            embedded_chunks.append(
                EmbeddedChunk(
                    id=uuid4(),
                    content=chunk.content,
                    embedding=api.feature_extraction(chunk.content),
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    metadata=chunk.metadata,
                    similarity=None,
                )
            )
        except Exception as e:
            logger.error(f"Error embedding chunk: {e}")
    logger.info(f"{len(embedded_chunks)} chunks embedded successfully")

    return embedded_chunks


def chunk_and_embed(
    cleaned_documents: Annotated[list, "cleaned_documents"],
) -> Annotated[list, "embedded_documents"]:
    embedded_chunks = []
    for document in cleaned_documents:
        chunks = chunk_text(document)

        for batched_chunks in batch(chunks, 10):
            batched_embedded_chunks = embed_chunks(batched_chunks)
            embedded_chunks.extend(batched_embedded_chunks)
    logger.info(f"{len(embedded_chunks)} chunks embedded successfully")
    return embedded_chunks
