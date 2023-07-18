from aleph_alpha_client import CompletionRequest, CompletionResponse, Prompt, SemanticEmbeddingRequest, SemanticRepresentation

from tests import build_llm_wrapper


def test_aleph_alpha_completion_request():
    llm_wrapper = build_llm_wrapper()

    completion_request = llm_wrapper.build_aleph_alpha_request(
        model="luminous-base",
        request_object=CompletionRequest(
            prompt=Prompt.from_text("An apple a day"),
            maximum_tokens=64
        )
    )
    response = llm_wrapper.aleph_alpha_request(completion_request)
    assert isinstance(
        response, CompletionResponse), f"Expected response of type aleph_alpha_client.CompletionResponse but got {type(response)}."
    completion = response.completions[0].completion
    assert isinstance(
        completion, str), f"Expected completion of type str but got {type(response)}."
    contains_str = "keeps the doctor away"
    assert contains_str in completion, f"Expected \"keeps the doctor away\" to be in completion, but got completion: {completion}"
