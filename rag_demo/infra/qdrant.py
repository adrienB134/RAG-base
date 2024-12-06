from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse


class QdrantDatabaseConnector:
    _instance: QdrantClient | None = None

    def __new__(cls, *args, **kwargs) -> QdrantClient:
        if cls._instance is None:
            try:
                cls._instance = QdrantClient(":memory:")

                logger.info(f"Connection to Qdrant DB with URI successful")
            except:
                logger.exception(
                    "Couldn't connect to Qdrant.",
                )

                raise

        return cls._instance


connection = QdrantDatabaseConnector()
