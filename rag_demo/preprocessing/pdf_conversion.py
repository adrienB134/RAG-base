from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from uuid import uuid4
from .base import Document
from loguru import logger

from dotenv import load_dotenv

load_dotenv()


# set up parser
parser = LlamaParse(
    api_key="llx-TN6YSXvZdpG0qhJ7rVx9QFg5Zq298RXr7Id7XzXb5Wr4Rnpt",
    result_type="markdown",  # "markdown" and "text" are available
)


def convert_pdf_to_text(filepaths: list[str]) -> Document:
    file_extractor = {".pdf": parser}
    # use SimpleDirectoryReader to parse our file

    documents = SimpleDirectoryReader(
        input_files=filepaths, file_extractor=file_extractor
    ).load_data()

    logger.info("Converted 1 documents")

    return Document(
        document_id=uuid4(),
        text=" ".join(document.text for document in documents),
        metadata={"filename": filepaths[0].split("/")[-1]},
    )
