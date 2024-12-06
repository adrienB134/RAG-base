import os

from huggingface_hub import InferenceClient

from rag_demo.rag.base.query import Query
from rag_demo.rag.base.template_factory import RAGStep
from rag_demo.preprocessing.embed import EmbeddedChunk


class Reranker(RAGStep):
    def generate(
        self, query: Query, chunks: list[EmbeddedChunk], keep_top_k: int
    ) -> list[EmbeddedChunk]:
        api = InferenceClient(
            model="intfloat/multilingual-e5-large-instruct",
            token=os.getenv("HF_API_TOKEN"),
        )
        similarity = api.sentence_similarity(
            query.content, [chunk.content for chunk in chunks]
        )
        for chunk, sim in zip(chunks, similarity):
            chunk.similarity = sim

        return sorted(chunks, key=lambda x: x.similarity, reverse=True)[:keep_top_k]
