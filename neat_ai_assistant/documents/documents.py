import json

from hashlib import sha256
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Optional, Tuple
from aleph_alpha_client import Prompt, SemanticEmbeddingRequest, SemanticEmbeddingResponse, SemanticRepresentation
from tqdm import tqdm

from ..llm import LLMWrapper, ChatRequest

from .parser import Parser


# ToDos:
# - Integrate Qdrant (or other)
# - Include additional search params
# - Use pydantic for automatic serialization


@dataclass
class Chunk:
    content: str
    hash_id: str
    embedding: Optional[Sequence[float]] = None

    def add_embedding(self, embedding: Sequence[float]) -> None:
        self.embedding = embedding


@dataclass
class Document:
    chunks: Sequence[Chunk]
    path: Path
    hash_id: str

    @classmethod
    def from_json(cls, json: dict) -> "Document":
        return cls(
            chunks=[Chunk(**c) for c in json["chunks"]],
            path=Path(json["path"]),
            hash_id=json["hash_id"]
        )

    def get_excerpt(self) -> str:
        excerpt = ""
        for c in self.chunks[:5]:
            excerpt += " ".join(c.content.split()[:20]) + "[...]"
        return excerpt

    def serialize(self) -> dict[str, dict[str, Any]]:
        return {
            self.hash_id: {
                "chunks": [asdict(c) for c in self.chunks],
                "path": str(self.path)
            }
        }

    def add_embedding_for_chunk(
        self,
        chunk_idx: int,
        embedding: Sequence[float]
    ) -> None:
        self.chunks[chunk_idx].add_embedding(embedding)


@dataclass
class DocumentSearchResult:
    document: Document
    chunk_scores: Sequence[float]
    best_chunks: Optional[Sequence[Tuple[Chunk, float]]] = None

    def add_best_chunks(self, threshold: float) -> None:
        best_chunks = []
        for idx, score in enumerate(self.chunk_scores):
            if score > threshold:
                best_chunks.append(
                    (self.document.chunks[idx], score)
                )
        self.best_chunks = sorted(
            best_chunks,
            key=lambda chunk_score: chunk_score[1],
            reverse=True)


class DocumentMinion:
    def __init__(
        self,
        llm_wrapper: LLMWrapper,
        documents_path: Path
    ) -> None:
        self.llm_wrapper = llm_wrapper
        self.parser = Parser()

        self.documents_path = documents_path
        self.database_path = (self.documents_path).parent / "neat_database"

        self.database_path.mkdir(parents=True, exist_ok=True)
        self.documents: Optional[Sequence[Document]] = None
        self.name: Optional[str] = None
        self.description: Optional[str] = None

    def get_description(
        self
    ) -> None:
        document_text_excerpts = "\n".join(
            d.get_excerpt() for d in self.documents[:10]
        )
        response = self.llm_wrapper.open_ai_chat_complete(
            request=ChatRequest.from_json(
                json_object={
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {
                            "role": "system",
                            "content": """You are the head of a document library.
You are presented with a list of texts from your library and seek to find out what the topic of your library is."""
                        },
                        {
                            "role": "user",
                            "content": f"""Hey! I found books with these texts in your library:
{document_text_excerpts}

Can you write a short summary of what your library is about?
Please use the following format:
```
<Database Title>
<Database Summary>
```"""
                        }
                    ],
                    "temperature": 0
                }
            )
        )
        result: str = response.completions[0].message.content
        res_tuple = result.strip().split("\n")
        if len(res_tuple) != 2:
            res_tuple = ("Database", result)
        self.name, self.description = res_tuple

    @staticmethod
    def _collect_file_paths(folder_path: Path) -> Sequence[Path]:
        all_file_paths = []
        supported_extensions = ['.pdf', '.xlsx']
        for item in folder_path.glob('**/*'):
            if item.is_file() and item.suffix.lower() in supported_extensions:
                all_file_paths.append(item)
        return all_file_paths

    @staticmethod
    def _generate_hash_id(
            file_path: Optional[Path] = None,
            string: Optional[str] = None) -> str:
        if bool(file_path) == bool(string):
            raise TypeError("Must provide only one of file_path, string")
        sha256_hash = sha256()

        if bool(file_path):
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
        else:
            byte_string = string.encode()
            sha256_hash.update(byte_string)

        return sha256_hash.hexdigest()

    def _create_document_object_from_file(
            self, file_path: Path, doc_hash_id: str) -> Document:
        raw_chunks = self.parser.parse_file(file_path)
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

    def _add_document_embeddings(self, document: Document) -> None:
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
        responses: Sequence[SemanticEmbeddingResponse] = self.llm_wrapper.aleph_alpha_batch_request(
            requests=request_dicts, )
        for idx, r in enumerate(responses):
            document.add_embedding_for_chunk(
                chunk_idx=idx,
                embedding=r.embedding
            )

    def _load_raw_documents(self) -> dict[str, dict]:
        all_documents = {}
        for doc_file in self.database_path.glob("documents*.json"):
            with doc_file.open("r") as file:
                all_documents |= json.load(file)
        return all_documents

    def _document_from_raw(self, document: dict, id: str) -> Document:
        return Document(
            chunks=[
                Chunk(**c) for c in document["chunks"]
            ],
            hash_id=id,
            path=document["path"]
        )

    def _compare_documents(self, documents_path: Path) -> list[Document]:
        file_paths = self._collect_file_paths(documents_path)
        saved_documents = self._load_raw_documents()
        documents: list[Document] = []
        for file_path in tqdm(file_paths, desc="Parsing and embedding..."):
            doc_hash_id = self._generate_hash_id(file_path=file_path)
            if doc_hash_id not in saved_documents:
                doc = self._create_document_object_from_file(
                    file_path, doc_hash_id)
                self._add_document_embeddings(doc)
                documents.append(doc)
            else:
                raw_saved_document = saved_documents[doc_hash_id]
                saved_document = self._document_from_raw(
                    document=raw_saved_document,
                    id=doc_hash_id
                )
                documents.append(saved_document)
        return documents

    def _save_documents(
            self,
            documents: Sequence[Document],
            max_documents_per_file: int = 50) -> None:
        for i in range(0, len(documents), max_documents_per_file):
            batch = documents[i: i + max_documents_per_file]
            serialized_docs = {}
            for d in batch:
                serialized_docs |= d.serialize()
            save_path = self.database_path / \
                f"documents{i//max_documents_per_file}.json"
            with save_path.open("w") as file:
                json.dump(serialized_docs, file, indent=4)

    def instantiate_database(self, update: bool = True) -> None:
        if update:
            documents = self._compare_documents(self.documents_path)
            self._save_documents(documents)
        else:
            raw_documents = self._load_raw_documents()
            documents = [
                self._document_from_raw(
                    d,
                    key) for key,
                d in raw_documents.items()]
        self.documents = documents
        self.get_description()

    def score_chunks(
            self,
            embedded_query: Sequence[float],
            threshold: float) -> Sequence[DocumentSearchResult]:
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

    def search(
            self,
            question: str,
            threshold: float = 0.7) -> Sequence[DocumentSearchResult]:
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
            request=request_dict, )
        results = self.score_chunks(
            embedded_query=response.embedding,
            threshold=threshold
        )
        return results
