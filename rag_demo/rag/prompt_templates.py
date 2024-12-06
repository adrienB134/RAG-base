from langchain.prompts import PromptTemplate

from .base import PromptTemplateFactory


class QueryExpansionTemplate(PromptTemplateFactory):
    prompt: str = """You are an AI language model assistant. Your task is to generate {expand_to_n}
    different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions seperated by '{separator}'.
    Original question: {question}"""

    @property
    def separator(self) -> str:
        return "#next-question#"

    def create_template(self, expand_to_n: int) -> PromptTemplate:
        return PromptTemplate(
            template=self.prompt,
            input_variables=["question"],
            partial_variables={
                "separator": self.separator,
                "expand_to_n": expand_to_n,
            },
        )


class AnswerGenerationTemplate(PromptTemplateFactory):
    prompt: str = """You are an AI language model assistant. Your task is to generate an answer to the given user question based on the provided context.
    Context: {context}
    Question: {question}

    Give your answer in markdown format.
    Give only your answer, do not include any other text like 'Certainly! Here is the answer:' or 'The answer is:' or anything similar."""

    def create_template(self, context: str, question: str) -> str:
        return self.prompt.format(context=context, question=question)
