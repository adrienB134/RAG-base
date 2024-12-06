import os
from typing import Any

from huggingface_hub import InferenceClient

from rag_demo.rag.base.query import Query
from rag_demo.rag.base.template_factory import RAGStep
from rag_demo.rag.prompt_templates import QueryExpansionTemplate


class QueryExpansion(RAGStep):
    def generate(self, query: Query, expand_to_n: int) -> Any:
        api = InferenceClient(
            model="Qwen/Qwen2.5-72B-Instruct",
            token=os.getenv("HF_API_TOKEN"),
        )
        query_expansion_template = QueryExpansionTemplate()
        prompt = query_expansion_template.create_template(expand_to_n - 1)
        response = api.chat_completion(
            [
                {
                    "role": "user",
                    "content": prompt.template.format(
                        question=query.content,
                        expand_to_n=expand_to_n,
                        separator=query_expansion_template.separator,
                    ),
                }
            ]
        )
        result = response.choices[0].message.content
        queries_content = result.split(query_expansion_template.separator)
        queries = [query]
        queries += [
            query.replace_content(stripped_content)
            for content in queries_content
            if (stripped_content := content.strip())
        ]
        return queries
