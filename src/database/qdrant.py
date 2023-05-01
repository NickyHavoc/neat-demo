from pathlib import Path
from typing import List, Literal, Optional
from qdrant_client import models, QdrantClient

from .document import Document


def collect_file_paths(folder_path: Path):
    all_file_paths = []
    for item in folder_path.glob('**/*'):
        if item.is_file():
            all_file_paths.append(item)
    return all_file_paths


def collect_documents(file_paths: List[Path]):
    return [Document(file_path) for file_path in file_paths]


def get_embeddings(documents: List[Document]):
    pass


def instantiate_database(
    documents_path: Path,
    database_path: Optional[Path]=None,
):
    file_paths = collect_file_paths(documents_path)
    documents = collect_documents(file_paths=file_paths)
    if database_path == None:
        database_path = documents_path.parent / "neat_database"

    qdrant = QdrantClient(path=database_path / "qdrant")
    qdrant.recreate_collection(
    collection_name=documents_path.name,
    vectors_config=models.VectorParams(
        size=128,
        distance=models.Distance.COSINE
    )
)

