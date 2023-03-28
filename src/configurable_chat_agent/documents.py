import hashlib
import json
import re

from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from tqdm.asyncio import tqdm
from aleph_alpha_client import (
    Prompt,
    SemanticEmbeddingRequest,
    SemanticEmbeddingResponse,
    SemanticRepresentation,
)

from .api_wrapper import LLMWrapper


class Chunk:
    def __init__(
        self,
        text: str,
        chunk_idx: int,
        parent_document_ident: str,
        ident: Optional[str] = None,
        embedding: Optional[Sequence[float]] = None,
    ):

        self.text = text
        self.chunk_idx = chunk_idx
        self.parent_document_ident = parent_document_ident

        if not ident:
            self.ident = hashlib.sha3_256(
                f"{self.text}{str(self.chunk_idx)}{self.parent_document_ident}".encode("utf-8")
            ).hexdigest()
        else:
            self.ident = ident

        self.embedding = embedding

    def __eq__(self, __obj: object):
        if not isinstance(__obj, Chunk):
            return False
        if not self.ident == __obj.ident:
            return False
        return True

    @classmethod
    def from_json(cls, json_chunk: dict):
        return cls(**json_chunk)

    def set_embedding(self, embedding: Sequence[float]):
        self.embedding = embedding

    def as_json(self):
        return {
            "ident": self.ident,
            "text": self.text,
            "chunk_idx": self.chunk_idx,
            "parent_document_ident": self.parent_document_ident,
            "embedding": self.embedding,
        }


class Document:
    def __init__(
        self,
        text: str,
        metadata: dict,
        ident: Optional[str] = None,
        max_chunk_length: Optional[int] = None,
        chunk_steps: Optional[List[List[str]]] = None,
        chunks: Optional[List[Chunk]] = None,
    ):
        self.text = text
        self.metadata = metadata

        if not ident:
            metadata_str = json.dumps(self.metadata)
            self.ident = hashlib.sha3_256(
                f"{self.text}{metadata_str}".encode("utf-8")
            ).hexdigest()
        else:
            self.ident = ident

        assert (bool(max_chunk_length) and bool(chunk_steps)) != bool(
            chunks
        ), "Must provide exactly one of max_chunk_length and chunk steps or chunks."

        if chunks:
            self.chunks = [Chunk.from_json(c) for c in chunks]

        else:
            raw_chunks = self.chunk_text(
                text=self.text,
                max_length=max_chunk_length,
                chunk_steps=chunk_steps
            )
            self.chunks = [Chunk(
                text=text,
                chunk_idx=idx,
                parent_document_ident=self.ident
            ) for idx, text in enumerate(raw_chunks)]

    @classmethod
    def from_json(cls, json_document: dict):
        return cls(
            **json_document,
        )

    @staticmethod
    def split_at(text: str, chunk_steps: List[List[str]], step_idx: int):
        delimiters = chunk_steps[step_idx]
        if delimiters is not None:
            regex = "([{}])".format("".join(delimiters))
            pattern = re.compile(regex)
            return pattern.split(text)
        return [c for c in text]

    def chunk_text(
        self,
        text: str,
        max_length: int,
        chunk_steps: List[List[str]],
        chunk_step: int = 0,
    ):

        chunks = []
        proposed_chunking = self.split_at(
            text=text, chunk_steps=chunk_steps, step_idx=chunk_step
        )

        for proposed_chunk in proposed_chunking:
            if len(proposed_chunk) < max_length:
                if chunks == []:
                    chunks.append(proposed_chunk)
                elif len(chunks[-1] + proposed_chunk) < max_length:
                    chunks[-1] += proposed_chunk
                else:
                    chunks.append(proposed_chunk)
            else:
                next_level_chunking = self.chunk_text(
                    text=proposed_chunk,
                    max_length=max_length,
                    chunk_steps=chunk_steps,
                    chunk_step=chunk_step + 1,
                )
                chunks += next_level_chunking
        return chunks

    def add_embeddings(self, embeddings: List[Sequence[float]]):
        assert len(embeddings) == len(
            self.chunks
        ), "There must be an equal number of chunks and embeddings."
        for idx, chunk in enumerate(self.chunks):
            chunk.set_embedding(embeddings[idx])

    def as_json(self):
        return {
            "ident": self.ident,
            "text": self.text,
            "metadata": self.metadata,
            "chunks": [c.as_json() for c in self.chunks],
        }


class DocumentSearchResult:
    def __init__(self):
        self.results: List[Tuple[float, Chunk]] = []

    def __bool__(self):
        return bool(self.results)

    def sort(self):
        self.results.sort(key=lambda x: x[0], reverse=True)

    def add_result(self, result: Tuple[float, Chunk]):
        self.results.append(result)

    # gets the top k chunks, irrespective of the document they are in
    def get_top_k_chunks(self, top_k: int):
        self.sort()
        top_k_res = self.results[:top_k]
        return [res[1] for res in top_k_res]


class DocumentBase:
    CHUNK_STEPS = [["\n\n"], ["\n"], [".", "!", "?"], [",", ":", "-"], [" "], None]

    def __init__(
        self,
        wrapper: LLMWrapper,
        path: Path,
        max_chunk_length: int = 1000,
        search_cutoff: float = 0.65,
    ):
        self.wrapper = wrapper

        self.dir = path

        self.documents_file = self.dir / "documents.json"
        self.raw_documents = json.load(open(self.documents_file, "r", encoding="UTF-8"))
        self.max_chunk_length = max_chunk_length
        self.search_cutoff = search_cutoff

        new_documents = [
            Document(
                **d,
                max_chunk_length=self.max_chunk_length,
                chunk_steps=self.CHUNK_STEPS,
            )
            for d in self.raw_documents
        ]
        self.cache_file = self.dir / "documents_cache.json"
        self.documents: List[Document] = self._get_documents_and_save_cache(
            new_documents=new_documents
        )

    @classmethod
    def from_json(
        cls,
        wrapper: LLMWrapper,
        path: Path,
        max_chunk_length: int,
        search_cutoff: float,
    ):
        return cls(
            wrapper=wrapper,
            path=path,
            max_chunk_length=max_chunk_length,
            search_cutoff=search_cutoff,
        )

    def find_document_by_ident(self, ident: str):
        for d in self.documents:
            if d.ident == ident:
                return d

    def _embed_query(self, text: str):
        request = self.wrapper.build_aleph_alpha_request(
            request_object=SemanticEmbeddingRequest(
                prompt=Prompt.from_text(text),
                representation=SemanticRepresentation.Query,
                compress_to_size=128,
            ),
            model="luminous-base",
        )
        response: SemanticEmbeddingResponse = self.wrapper.aleph_alpha_request(
            request=request
        )
        return response.embedding

    # Set of functions to load exiting document embeddings and see if anything changed.
    def _embed_documents(self, texts: List[str]):
        requests = [
            self.wrapper.build_aleph_alpha_request(
                request_object=SemanticEmbeddingRequest(
                    prompt=Prompt.from_text(text),
                    representation=SemanticRepresentation.Document,
                    compress_to_size=128,
                ),
                model="luminous-base",
            )
            for text in texts
        ]
        responses: List[SemanticEmbeddingResponse] = self.wrapper.aleph_alpha_batch_request(
            requests=requests
        )
        return [response.embedding for response in responses]

    def _get_documents_and_save_cache(self, new_documents: List[Document]):
        if self.cache_file.exists():
            documents = self._load_documents_with_embeddings(
                new_documents=new_documents
            )
        else:
            documents = self._get_new_documents_with_embeddings(new_documents)

        with open(self.cache_file, "w", encoding="UTF-8") as f:
            json.dump([d.as_json() for d in documents], f, indent=4)

        return documents

    def _add_embeddings_for_document(self, document: Document):
        chunk_texts = [c.text for c in document.chunks]

        embeddings = self._embed_documents(texts=chunk_texts)
        document.add_embeddings(embeddings=embeddings)

    def _get_new_documents_with_embeddings(self, new_documents: List[Document]):
        documents_with_embeddings: List[Document] = []

        for new_doc in tqdm(
            new_documents, desc="Adding and enriching new documents..."
        ):
            self._add_embeddings_for_document(new_doc)
            documents_with_embeddings.append(new_doc)

        return documents_with_embeddings

    def _load_documents_with_embeddings(self, new_documents: List[Document]):
        with open(self.cache_file, encoding="UTF-8") as f:
            cache_json = json.load(f)
        cached_document_set = set(
            Document.from_json(document_json) for document_json in cache_json
        )
        new_document_hashes = set(d.ident for d in new_documents)

        updated_docs = []
        for old_doc in tqdm(
            cached_document_set, "Checking if old documents are still valid..."
        ):
            old_doc_hash = old_doc.ident
            if old_doc_hash in new_document_hashes:
                new_document_hashes.remove(old_doc_hash)
                updated_docs.append(old_doc)

        documents_yet_to_be_embedded = [
            d for d in new_documents if d.ident in new_document_hashes
        ]
        if documents_yet_to_be_embedded:
            additional_documents_with_embeddings = (
                self._get_new_documents_with_embeddings(
                    new_documents=documents_yet_to_be_embedded
                )
            )
            updated_docs += additional_documents_with_embeddings

        return updated_docs

    def score_query(
        self,
        query: Optional[str] = None,
        query_embedding: Optional[Sequence[float]] = None,
        sort_descending: bool = True,
    ):
        assert bool(query) != bool(query_embedding), "Must provide exactly one of query, query_embedding."
        if not query_embedding:
            query_embedding = self._embed_query(query)

        results = DocumentSearchResult()
        for document in self.documents:

            for chunk in document.chunks:
                score = (
                    self.wrapper.compute_cosine_similarity(
                        query_embedding, chunk.embedding
                    )
                )
                if score >= self.search_cutoff:
                    results.add_result(result=(score, chunk))
        results.sort() if sort_descending else None
        return results
