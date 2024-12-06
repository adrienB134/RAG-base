from abc import ABC

from pydantic import UUID4, Field


from .vectordb import VectorBaseDocument


class EmbeddedChunk(VectorBaseDocument, ABC):
    content: str
    embedding: list[float] | None
    document_id: UUID4
    chunk_id: UUID4
    metadata: dict = Field(default_factory=dict)
    similarity: float | None

    @classmethod
    def to_context(cls, chunks: list["EmbeddedChunk"]) -> str:
        context = ""
        for i, chunk in enumerate(chunks):
            context += f"""
            Chunk {i + 1}:
            Type: {chunk.__class__.__name__}
            Document ID: {chunk.document_id}
            Chunk ID: {chunk.chunk_id}
            Content: {chunk.content}\n
            """

        return context

    class Config:
        name = "embedded_documents"
        category = "Document"
        use_vector_index = True
