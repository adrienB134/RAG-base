from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered,

converter = PdfConverter(
    artifact_dict=create_model_dict(),
)


def convert_pdf_to_text(filepaths: list[str]) -> list[str]:
    """
    Convert a list of PDF filepaths to a list of text strings.
    """
    texts = []
    for filepath in filepaths:
        rendered = converter(filepath)
        text, _, _ = text_from_rendered(rendered)
        metadata = rendered.metadata
        texts.append((text, metadata))
    return texts



