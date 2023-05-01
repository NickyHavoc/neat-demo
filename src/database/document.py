import hashlib
from http.client import responses
import json
import PyPDF2

from typing import Dict, List, Optional
from pathlib import Path
from aleph_alpha_client import Prompt, SemanticEmbeddingRequest, SemanticEmbeddingResponse, SemanticRepresentation

from utils import LLMWrapper


class Document:
    def __init__(
        self,
        path_to_doc: Path
    ):
        self.doc_path = path_to_doc
        self.doc_type = path_to_doc.suffix
        self.title = path_to_doc.stem
        if self.doc_type == ".pdf":
            self.content = self.read_pdf_content()
        elif self.doc_type == ".xlsx":
            self.content = self.read_excel_content()
        else:
            raise TypeError(f"Doc type not allowed.")
        self.ident = hashlib.sha256(f"{str(self.doc_path)}{''.join(self.content)}".encode()).hexdigest
        self.embeddings = None

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Document):
            return False
        if self.ident != __o.ident:
            return False
        return True

    def read_pdf_content(self):
        return []

    def read_excel_content(self):
        pages: List[str] = []
        with open('example.pdf', 'rb') as f:
            reader = PyPDF2.PdfFileReader(f)
            for page in reader.pages:
                pages.append(page.extract_text())
        return pages

    def add_embeddings(self, embeddings: List[float]):
        self.embeddings = embeddings

    def as_json(self):
        return {
            "content": self.content,
            "doc_path": self.doc_path,
            "doc_type": self.doc_type,
            "embedding": self.embeddings,
            "title": self.title
        }


class Embedder:
    def __init__(
        self,
        embeddings_json_path: Path
    ):
        self.llm_wrapper = LLMWrapper()
        if embeddings_json_path.is_file():
            with open(embeddings_json_path, 'r') as file:
                self.loaded_embeddings = json.load(file)

    def embed(
        self,
        document: Document,
        compare_versions: bool,
    ):
        """
        Will embed a document. By default, will instantiate a second document and compare the two to see if any changes were made.
        """
        if compare_versions:
            path_to_doc = document.doc_path
            sister_doc = Document(path_to_doc)
            if sister_doc == document:
                embeddings = self.loaded_embeddings[str(path_to_doc)]
        
        prompt_iter = (Prompt.from_text(c) for c in document.content)
        requests = [
            self.llm_wrapper.build_aleph_alpha_request(
                request_object=r,
                model="luminous-base"
            ) for r in (
                SemanticEmbeddingRequest(
                    prompt=p,
                    representation=SemanticRepresentation.Document,
                    compress_to_size=128
                ) for p in prompt_iter
            )
        ]
        responses: List[SemanticEmbeddingResponse] = self.llm_wrapper.aleph_alpha_batch_request(requests)
        embeddings = [r.embedding for r in responses]
        document.add_embeddings(embeddings)
        return (document.doc_path, embeddings)

    def embed_documents(
        self,
        documents: List[Document],
        compare_versions: bool=True
    ):
        pass
