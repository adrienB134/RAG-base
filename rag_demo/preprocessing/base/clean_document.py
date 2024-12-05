from abc import ABC
from typing import Optional

from pydantic import UUID4

from .base import VectorBaseDocument


class CleanedDocument(VectorBaseDocument, ABC):
    content: str
    doc_id: UUID4
    doc_title: str
    # doc_url: str
