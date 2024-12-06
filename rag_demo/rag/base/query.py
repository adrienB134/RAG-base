from pydantic import UUID4, Field

from rag_demo.preprocessing.base import VectorBaseDocument


class Query(VectorBaseDocument):
    content: str
    metadata: dict = Field(default_factory=dict)

    class Config:
        category = "query"

    @classmethod
    def from_str(cls, query: str) -> "Query":
        return Query(content=query.strip("\n "))

    def replace_content(self, new_content: str) -> "Query":
        return Query(
            id=self.id,
            content=new_content,
            metadata=self.metadata,
        )


class EmbeddedQuery(Query):
    embedding: list[float]

    class Config:
        category = "query"
