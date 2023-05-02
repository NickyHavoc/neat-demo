from pathlib import Path

from documents import DocumentHandler


documents_path = Path(__file__).parent / "example_docs"
doc_handler = DocumentHandler()
doc_handler.instantiate_database(
    documents_path=documents_path
)
print("")
