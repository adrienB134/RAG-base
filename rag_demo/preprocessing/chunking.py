from uuid import uuid4

from langchain.text_splitter import MarkdownTextSplitter
from rag_demo.preprocessing.base import Chunk
from rag_demo.preprocessing.base import Document


def chunk_text(
    document: Document, chunk_size: int = 500, chunk_overlap: int = 50
) -> list[Chunk]:
    text_splitter = MarkdownTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(document.text)
    result = []
    for chunk in chunks:
        result.append(
            Chunk(
                content=chunk,
                document_id=document.document_id,
                chunk_id=uuid4(),
                metadata=document.metadata,
            )
        )

    return result
