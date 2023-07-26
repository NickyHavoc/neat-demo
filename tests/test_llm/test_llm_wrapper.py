from typing import List
from aleph_alpha_client import CompletionRequest, CompletionResponse, Prompt, SemanticEmbeddingRequest, SemanticEmbeddingResponse, SemanticRepresentation

from neat_ai_assistant.llm import ChatResponse

from tests import build_llm_wrapper, build_chat_request


def test_aleph_alpha_completion():
    llm_wrapper = build_llm_wrapper()

    completion_request = llm_wrapper.build_aleph_alpha_request(
        model="luminous-base",
        request_object=CompletionRequest(
            prompt=Prompt.from_text("An apple a day"),
            maximum_tokens=64
        )
    )
    response: CompletionResponse = llm_wrapper.aleph_alpha_request(completion_request)
    assert isinstance(
        response, CompletionResponse), f"Expected response of type aleph_alpha_client.CompletionResponse but got {type(response)}."
    completion = response.completions[0].completion
    assert isinstance(
        completion, str), f"Expected completion of type str but got {type(response)}."
    contains_str = "keeps the doctor away"
    assert contains_str in completion, f"Expected \"{contains_str}\" to be in completion, but got completion: {completion}"


def test_aleph_alpha_symmetric_semantic_embedding():
    llm_wrapper = build_llm_wrapper()

    sym_embedding_request_1 = llm_wrapper.build_aleph_alpha_request(
        model="luminous-base",
        request_object=SemanticEmbeddingRequest(
            prompt=Prompt.from_text("Mount Everest is 8848 meters high."),
            representation=SemanticRepresentation.Symmetric,
            compress_to_size=128
        )
    )
    sym_embedding_request_2 = llm_wrapper.build_aleph_alpha_request(
        model="luminous-base",
        request_object=SemanticEmbeddingRequest(
            prompt=Prompt.from_text("Der Mount Everest ist 8848 Meter hoch."),
            representation=SemanticRepresentation.Symmetric,
            compress_to_size=128
        )
    )
    responses: List[SemanticEmbeddingResponse] = llm_wrapper.aleph_alpha_batch_request(
        requests=[sym_embedding_request_1, sym_embedding_request_2]
    )
    assert all(isinstance(
        r, SemanticEmbeddingResponse) for r in responses), "Expected response of type aleph_alpha_client.SemanticEmbeddingResponse."
    embeddings = [r.embedding for r in responses]
    assert all(isinstance(
        e, list) for e in embeddings), f"Expected embeddings of type list."
    similarity = llm_wrapper.compute_cosine_similarity(embedding_1=embeddings[0], embedding_2=embeddings[1])
    similarity_threshold = 0.8
    assert similarity > similarity_threshold, f"Expected high score (> {str(similarity_threshold)}) for similar texts, but got score: {round(similarity, 2)}"


def test_open_ai_chat():
    llm_wrapper = build_llm_wrapper()
    chat_request, contains_str = build_chat_request(get_contains_str=True)
    response: ChatResponse = llm_wrapper.open_ai_chat_complete(chat_request)
    assert isinstance(
        response, ChatResponse), f"Expected response of type ChatResponse but got {type(response)}."
    completion = response.completions[0].message.content
    assert isinstance(
        completion, str), f"Expected completion of type str but got {type(response)}."
    assert contains_str in completion, f"Expected \"{contains_str}\" to be in completion, but got completion: {completion}"

