from loguru import logger
from typing_extensions import Annotated
from typing import Generator

from .base import VectorBaseDocument


def batch(list_: list, size: int) -> Generator[list, None, None]:
    yield from (list_[i : i + size] for i in range(0, len(list_), size))


def load_to_vector_db(
    documents: Annotated[list, "documents"],
) -> Annotated[bool, "successful"]:
    logger.info(f"Loading {len(documents)} documents into the vector database.")

    grouped_documents = VectorBaseDocument.group_by_class(documents)
    for document_class, documents in grouped_documents.items():
        logger.info(f"Loading documents into {document_class.get_collection_name()}")
        for documents_batch in batch(documents, size=4):
            try:
                document_class.bulk_insert(documents_batch)
            except Exception as e:
                logger.error(
                    f"Failed to insert documents into {document_class.get_collection_name()}: {e}"
                )

                return False

    return True
