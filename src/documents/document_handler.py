import uuid
import json

from qdrant_client import QdrantClient, models
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from pathlib import Path
from aleph_alpha_client import Prompt, SemanticEmbeddingRequest, SemanticEmbeddingResponse, SemanticRepresentation

from utils import LLMWrapper

from .parser import Parser


@dataclass
class Chunk:
    content: str
    embedding: Optional[List[float]] = None

    def add_embedding(self, embedding: List[float]):
        self.embedding = embedding


@dataclass
class Document:
    chunks: List[Chunk]
    uuid: str
    path: Path

    def get_payload_by_chunk(self, chunk_idx: int):
        return {
            "content": self.chunks[chunk_idx],
            "path": self.path
        }
    
    def add_embedding_for_chunk(
        self,
        chunk_idx: int,
        embedding: List[float]
    ):
        self.chunks[chunk_idx].add_embedding(embedding)


class DocumentHandler:
    def __init__(
        self,
        database_path: Optional[Path] = None
    ):
        self.llm_wrapper = LLMWrapper()
        self.parser = Parser()

        if database_path is None:
            self.database_path = Path.cwd() / "neat_database"
        else:
            self.database_path = database_path

        self.database_path.mkdir(parents=True, exist_ok=True)
        self.qdrant_client = QdrantClient(path=self.database_path / "qdrant")

    @staticmethod
    def collect_file_paths(folder_path: Path):
        all_file_paths = []
        supported_extensions = ['.pdf', '.xlsx']
        for item in folder_path.glob('**/*'):
            if item.is_file() and item.suffix.lower() in supported_extensions:
                all_file_paths.append(item)
        return all_file_paths

    @staticmethod
    def generate_document_uuid(content: str, namespace: uuid.UUID = uuid.NAMESPACE_URL) -> uuid.UUID:
        return uuid.uuid5(namespace, content)

    def create_document(self, file_path: Path):
        raw_chunks = self.parser.parse_file(file_path)
        uuid_str = "".join(raw_chunks) + str(file_path)
        uuid = str(self.generate_document_uuid(content=uuid_str))
        return Document(
            chunks=[Chunk(content=c) for c in raw_chunks],
            uuid=uuid,
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

    def save_document_uuids(self, uuids: List[str]):
        uuid_file = self.database_path / "document_uuids.json"
        with uuid_file.open("w") as file:
            json.dump(uuids, file, indent=4)

    def load_document_uuids(self):
        uuid_file = self.database_path / "document_uuids.json"
        if uuid_file.exists():
            with uuid_file.open("r") as file:
                return json.load(file)
        return []

    def compare_documents(self, documents_path: Path):
        file_paths = self.collect_file_paths(documents_path)
        saved_ids = self.load_document_uuids()
        new_documents = []
        active_ids = []
        for file_path in tqdm(file_paths, desc="Parsing and embedding..."):
            doc = self.create_document(file_path)
            if doc.uuid not in saved_ids:
                self.add_document_embeddings(doc)
                new_documents.append(doc)
            active_ids.append(doc.uuid)
        deprecated_ids = [id for id in saved_ids if id not in active_ids]
        return new_documents, active_ids, deprecated_ids

    def instantiate_database(self, documents_path: Path):
        new_documents, active_ids, deprecated_ids = self.compare_documents(documents_path)
        self.save_document_uuids(active_ids)
        self.recreate_qdrant_collection()
        self.delete_qdrant_points(deprecated_ids)
        self.upload_qdrant_points(new_documents)

    def recreate_qdrant_collection(self):
        self.qdrant_client.recreate_collection(
            collection_name="documents",
            vectors_config=models.VectorParams(
                size=128,
                distance=models.Distance.COSINE
            )
        )

    def delete_qdrant_points(self, point_ids: List[str]):
        self.qdrant_client.clear_payload(
            collection_name="documents",
            points_selector=models.PointIdsList(
                points=point_ids
            ),
        )

    def upload_qdrant_points(self, documents: List[Document]):
        self.qdrant_client.upload_records(
            collection_name="documents",
            # Can one document have multiple ids?
            # Will probably wanna switch to recording ids on a document and chunk level...
            records=[
                models.Record(
                    id=document.uuid,
                    vector=chunk.embedding,
                    payload=document.get_payload_by_chunk(idx)
                ) for document in documents for idx, chunk in enumerate(document.chunks)
            ]
        )
