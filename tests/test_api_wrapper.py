from typing import List
from aleph_alpha_client import CompletionRequest, CompletionResponse, Prompt

from neat_demo import LLMWrapper


WRAPPER = LLMWrapper()


def test_aleph_alpha_request():
    requests = WRAPPER.build_aleph_alpha_request(
        request_object=CompletionRequest(
            prompt=Prompt.from_text("An apple a day"),
            temperature=0,
            maximum_tokens=10,
        ),
        model="luminous-base",
    )

    response: CompletionResponse = WRAPPER.aleph_alpha_request(requests)
    assert isinstance(
        response, CompletionResponse
    ), "Expected object of type CompletionResponse."
    assert (
        "keeps the doctor away" in response.completions[0].completion.lower()
    ), "Very simple prompt was not completed correctly."


def test_aleph_alpha_batch_request():
    requests = [
        WRAPPER.build_aleph_alpha_request(
            request_object=CompletionRequest(
                prompt=Prompt.from_text("An apple a day"),
                temperature=0,
                maximum_tokens=10,
            ),
            model="luminous-base",
        )
        for _ in range(2)
    ]

    responses: List[CompletionResponse] = WRAPPER.aleph_alpha_batch_request(requests)
    assert all(isinstance(r, CompletionResponse) for r in responses)

    response_1, response_2 = responses  
    assert (
        response_1.completions[0].completion == response_2.completions[0].completion
    ), "Completions should be the same but aren't."
    assert (
        "keeps the doctor away" in response_1.completions[0].completion.lower()
    ), "Very simple prompt was not completed correctly."


def test_open_ai_chat_complete():
    messages = [
        {
            "role": "system",
            "content": "You like pizza."
        },
        {
            "role": "user",
            "content": "Repeat after me. \"I like pizza.\"."
        }
    ]
    response = WRAPPER.open_ai_chat_complete(
        params={
            "model": "gpt-3.5-turbo",
            "messages": messages,
            "temperature": 0
        }
    )
    completion = response["choices"][0]["message"]["content"]
    assert "pizza" in completion, "Very simple prompt was not completed correctly."
