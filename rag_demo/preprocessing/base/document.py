from abc import ABC
from typing import Optional

from pydantic import UUID4, BaseModel

from .vectordb import VectorBaseDocument


class CleanedDocument(VectorBaseDocument, ABC):
    content: str
    doc_id: UUID4
    doc_title: str
    # doc_url: str


class Document(BaseModel):
    text: str
    document_id: UUID4
    metadata: dict
