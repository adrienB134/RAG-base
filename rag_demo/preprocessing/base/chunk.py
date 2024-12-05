from abc import ABC
from typing import Optional

from pydantic import UUID4, Field

from rag_demo.preprocessing.base.vectordb import VectorBaseDocument


class Chunk(VectorBaseDocument, ABC):
    content: str
    platform: str
    document_id: UUID4
    author_id: UUID4
    author_full_name: str
    metadata: dict = Field(default_factory=dict)
