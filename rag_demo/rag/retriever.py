import concurrent.futures
import os

from loguru import logger
from qdrant_client.models import FieldCondition, Filter, MatchValue
from huggingface_hub import InferenceClient

from rag_demo.preprocessing.base import (
    EmbeddedChunk,
)
from rag_demo.rag.base.query import EmbeddedQuery, Query

from .query_expansion import QueryExpansion
from .reranker import Reranker
from .prompt_templates import AnswerGenerationTemplate

from dotenv import load_dotenv

load_dotenv()


def flatten(nested_list: list) -> list:
    """Flatten a list of lists into a single list."""

    return [item for sublist in nested_list for item in sublist]


class RAGPipeline:
    def __init__(self, mock: bool = False) -> None:
        self._query_expander = QueryExpansion(mock=mock)
        self._reranker = Reranker(mock=mock)

    def search(
        self,
        query: str,
        k: int = 3,
        expand_to_n_queries: int = 3,
    ) -> list:
        query_model = Query.from_str(query)

        n_generated_queries = self._query_expander.generate(
            query_model, expand_to_n=expand_to_n_queries
        )
        logger.info(
            f"Successfully generated {len(n_generated_queries)} search queries.",
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            search_tasks = [
                executor.submit(self._search, _query_model, k)
                for _query_model in n_generated_queries
            ]

            n_k_documents = [
                task.result() for task in concurrent.futures.as_completed(search_tasks)
            ]
            n_k_documents = flatten(n_k_documents)
            n_k_documents = list(set(n_k_documents))

        logger.info(f"{len(n_k_documents)} documents retrieved successfully")

        if len(n_k_documents) > 0:
            k_documents = self.rerank(query, chunks=n_k_documents, keep_top_k=k)
        else:
            k_documents = []

        return k_documents

    def _search(self, query: Query, k: int = 3) -> list[EmbeddedChunk]:
        assert k >= 3, "k should be >= 3"

        def _search_data(
            data_category_odm: type[EmbeddedChunk], embedded_query: EmbeddedQuery
        ) -> list[EmbeddedChunk]:
            return data_category_odm.search(
                query_vector=embedded_query.embedding,
                limit=k,
            )

        api = InferenceClient(
            model="intfloat/multilingual-e5-large-instruct",
            token=os.getenv("HF_API_TOKEN"),
        )
        embedded_query: EmbeddedQuery = EmbeddedQuery(
            embedding=api.feature_extraction(query.content),
            id=query.id,
            content=query.content,
        )

        retrieved_chunks = _search_data(EmbeddedChunk, embedded_query)
        logger.info(f"{len(retrieved_chunks)} documents retrieved successfully")

        return retrieved_chunks

    def rerank(
        self, query: str | Query, chunks: list[EmbeddedChunk], keep_top_k: int
    ) -> list[EmbeddedChunk]:
        if isinstance(query, str):
            query = Query.from_str(query)

        reranked_documents = self._reranker.generate(
            query=query, chunks=chunks, keep_top_k=keep_top_k
        )

        logger.info(f"{len(reranked_documents)} documents reranked successfully.")

        return reranked_documents

    def generate_answer(self, query: str, reranked_chunks: list[EmbeddedChunk]) -> str:
        context = ""
        for chunk in reranked_chunks:
            context += "\n Document: "
            context += chunk.content
        api = InferenceClient(
            model="meta-llama/Llama-3.1-8B-Instruct",
            token=os.getenv("HF_API_TOKEN"),
        )
        answer_generation_template = AnswerGenerationTemplate()
        prompt = answer_generation_template.create_template(context, query)
        logger.info(prompt)
        response = api.chat_completion(
            [{"role": "user", "content": prompt}],
            max_tokens=8192,
        )
        return response.choices[0].message.content

    def rag(self, query: str) -> tuple[str, list[str]]:
        docs = self.search(query, k=10)
        reranked_docs = self.rerank(query, docs, keep_top_k=10)
        return (
            self.generate_answer(query, reranked_docs),
            [doc.metadata["filename"].split(".pdf")[0] for doc in reranked_docs],
        )
