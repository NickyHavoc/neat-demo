import json

from hashlib import sha256
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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


@dataclass
class DocumentSearchResult:
    document: Document
    chunk_scores: List[float]
    best_chunks: Optional[List[Tuple[Chunk, float]]] = None

    def add_best_chunks(self, threshold: float):
        best_chunks = []
        for idx, score in enumerate(self.chunk_scores):
            if score > threshold:
                best_chunks.append(
                    (self.document.chunks[idx], score)
                )
        self.best_chunks = sorted(best_chunks, key=lambda chunk_score: chunk_score[1], reverse=True)


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
    def _collect_file_paths(folder_path: Path) -> List[Path]:
        all_file_paths = []
        supported_extensions = ['.pdf', '.xlsx']
        for item in folder_path.glob('**/*'):
            if item.is_file() and item.suffix.lower() in supported_extensions:
                all_file_paths.append(item)
        return all_file_paths

    @staticmethod
    def _generate_hash_id(string: str) -> str:
        return sha256(string.encode('utf-8')).hexdigest()

    def _create_document_object_from_file(self, file_path: Path) -> Document:
        raw_chunks = self.parser.parse_file(file_path)
        unique_str = "".join(raw_chunks) + str(file_path)
        doc_hash_id = self._generate_hash_id(string=unique_str)
        return Document(
            chunks=[
                Chunk(
                    content=c,
                    hash_id=self._generate_hash_id(string=c)
                ) for c in raw_chunks
            ],
            hash_id=doc_hash_id,
            path=file_path
        )

    def _add_document_embeddings(self, document: Document):
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

    def _load_raw_documents(self) -> Dict[str, dict]:
        documents_file = self.database_path / "documents.json"
        if documents_file.exists():
            with documents_file.open("r") as file:
                return json.load(file)
        return {}

    def _document_from_raw(self, document: dict, id: str) -> List[Document]:
        return Document(
            chunks=[
                Chunk(**c) for c in document["chunks"]
            ],
            hash_id=id,
            path=document["path"]
        )

    def _compare_documents(self, documents_path: Path):
        file_paths = self._collect_file_paths(documents_path)
        saved_documents = self._load_raw_documents()
        documents = []
        for file_path in tqdm(file_paths, desc="Parsing and embedding..."):
            doc = self._create_document_object_from_file(file_path)
            if doc.hash_id not in saved_documents:
                self._add_document_embeddings(doc)
                documents.append(doc)
            else:
                raw_saved_document = saved_documents[doc.hash_id]
                saved_document = self._document_from_raw(
                    document=raw_saved_document,
                    id=doc.hash_id
                )
                documents.append(saved_document)
        return documents

    def _save_documents(self, documents: List[Document]) -> None:
        serialized_docs = {}
        for d in documents:
            serialized_docs |= d.serialize()
        save_path = self.database_path / "documents.json"
        with save_path.open("w") as file:
            json.dump(serialized_docs, file, indent=4)

    def instantiate_database(self, documents_path: Path, update: bool=True):
        if update:
            documents = self._compare_documents(documents_path)
            self._save_documents(documents)
        else:
            raw_documents = self._load_raw_documents()
            documents = [self._document_from_raw(d, key) for key, d in raw_documents.items()]
        self.documents = documents
        
    def score_chunks(self, embedded_query: List[float], threshold: float) -> List[DocumentSearchResult]:
        results = []
        for d in self.documents:
            chunk_scores = [self.llm_wrapper.compute_cosine_similarity(
                embedding_1=embedded_query,
                embedding_2=c.embedding
            ) for c in d.chunks]
            res = DocumentSearchResult(
                document=d,
                chunk_scores=chunk_scores
            )
            res.add_best_chunks(threshold=threshold)
            if bool(res.best_chunks):
                results.append(res)
        return sorted(results, key=lambda r: max(r.chunk_scores), reverse=True)

    def search(self, question: str, threshold: float = 0.7) -> List[DocumentSearchResult]:
        request_dict = self.llm_wrapper.build_aleph_alpha_request(
            request_object=SemanticEmbeddingRequest(
                prompt=Prompt.from_text(
                    text=question
                ),
                representation=SemanticRepresentation.Query,
                compress_to_size=128
            ),
            model="luminous-base"
        )
        response: SemanticEmbeddingResponse = self.llm_wrapper.aleph_alpha_request(
            request=request_dict,
        )
        results = self.score_chunks(
            embedded_query=response.embedding,
            threshold=threshold
        )
        return results
