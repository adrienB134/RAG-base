from abc import ABC
from typing import Optional

from pydantic import UUID4, Field

from rag_demo.preprocessing.base.vectordb import VectorBaseDocument


class Chunk(VectorBaseDocument, ABC):
    content: str
    document_id: UUID4
    chunk_id: UUID4
    metadata: dict = Field(default_factory=dict)
