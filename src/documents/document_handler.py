import json

from hashlib import sha256
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional
from aleph_alpha_client import Prompt, SemanticEmbeddingRequest, SemanticEmbeddingResponse, SemanticRepresentation
from tqdm import tqdm

from utils import LLMWrapper

from .parser import Parser


@dataclass
class Chunk:
    content: str
    hash_id: str
    embedding: Optional[List[float]] = None

    def add_embedding(self, embedding: List[float]) -> None:
        self.embedding = embedding


@dataclass
class Document:
    chunks: List[Chunk]
    path: Path
    hash_id: str

    @classmethod
    def from_json(cls, json: dict):
        return cls(
            chunks=[Chunk(**c) for c in json["chunks"]],
            path=Path(json["path"]),
            hash_id=json["hash_id"]
        )
    
    def serialize(self):
        return {
            self.hash_id: {
                "chunks": [asdict(c) for c in self.chunks],
                "path": str(self.path)
            }
        }

    def add_embedding_for_chunk(
        self,
        chunk_idx: int,
        embedding: List[float]
    ) -> None:
        self.chunks[chunk_idx].add_embedding(embedding)


class DocumentHandler:
    def __init__(
        self,
        database_path: Optional[Path] = None
    ) -> None:
        self.llm_wrapper = LLMWrapper()
        self.parser = Parser()

        if database_path is None:
            self.database_path = Path.cwd() / "neat_database"
        else:
            self.database_path = database_path

        self.database_path.mkdir(parents=True, exist_ok=True)
        self.documents: Optional[List[Document]] = None

    @staticmethod
    def collect_file_paths(folder_path: Path) -> List[Path]:
        all_file_paths = []
        supported_extensions = ['.pdf', '.xlsx']
        for item in folder_path.glob('**/*'):
            if item.is_file() and item.suffix.lower() in supported_extensions:
                all_file_paths.append(item)
        return all_file_paths

    @staticmethod
    def generate_hash_id(string: str) -> str:
        return sha256(string.encode('utf-8')).hexdigest()

    def create_document_object_from_file(self, file_path: Path) -> Document:
        raw_chunks = self.parser.parse_file(file_path)
        unique_str = "".join(raw_chunks) + str(file_path)
        doc_hash_id = self.generate_hash_id(string=unique_str)
        return Document(
            chunks=[
                Chunk(
                    content=c,
                    hash_id=self.generate_hash_id(string=c)
                ) for c in raw_chunks
            ],
            hash_id=doc_hash_id,
            path=file_path
        )

    def add_document_embeddings(self, document: Document):
        request_dicts = [
            self.llm_wrapper.build_aleph_alpha_request(
                request_object=SemanticEmbeddingRequest(
                    prompt=Prompt.from_text(
                        text=c.content
                    ),
                    representation=SemanticRepresentation.Document,
                    compress_to_size=128
                ),
                model="luminous-base"
            ) for c in document.chunks
        ]
        responses: List[SemanticEmbeddingResponse] = self.llm_wrapper.aleph_alpha_batch_request(
            requests=request_dicts,
        )
        for idx, r in enumerate(responses):
            document.add_embedding_for_chunk(
                chunk_idx=idx,
                embedding=r.embedding
            )

    def load_raw_documents(self) -> Dict[str, dict]:
        documents_file = self.database_path / "documents.json"
        if documents_file.exists():
            with documents_file.open("r") as file:
                return json.load(file)
        return {}

    def compare_documents(self, documents_path: Path):
        file_paths = self.collect_file_paths(documents_path)
        saved_documents = self.load_raw_documents()
        documents = []
        for file_path in tqdm(file_paths, desc="Parsing and embedding..."):
            # implement here check to see if file changed (with timestamp)
            doc = self.create_document_object_from_file(file_path)
            if doc.hash_id not in saved_documents:
                self.add_document_embeddings(doc)
                documents.append(doc)
            else:
                raw_saved_document = saved_documents[doc.hash_id]
                saved_document = Document(
                    chunks=[
                        Chunk(**c) for c in raw_saved_document["chunks"]
                    ],
                    hash_id=doc.hash_id,
                    path=raw_saved_document["path"]
                )
                documents.append(saved_document)
        return documents

    def save_documents(self, documents: List[Document]) -> None:
        serialized_docs = {}
        for d in documents:
            serialized_docs |= d.serialize()
        save_path = self.database_path / "documents.json"
        with save_path.open("w") as file:
            json.dump(serialized_docs, file, indent=4)

    def instantiate_database(self, documents_path: Path):
        documents = self.compare_documents(documents_path)
        self.documents = documents
        self.save_documents(documents)
