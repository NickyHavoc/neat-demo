import json
from typing import Any, Mapping, Sequence

from pytest import fixture

from neat_ai_assistant import Message, Model, OpenaiWrapper


@fixture
def example_messages() -> Sequence[Message]:
    return [
        Message(role="system", content="You answer user questions concisely."),
        Message(role="user", content="What's the capital of France?"),
    ]


@fixture
def example_function_key() -> str:
    return "city"


@fixture
def example_functions(example_function_key: str) -> Sequence[Mapping[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "city_index",
                "description": "Find information about any city, including capitals.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        example_function_key: {
                            "type": "string",
                            "description": "The city name to look for.",
                        }
                    },
                    "required": [example_function_key],
                },
            },
        },
    ]


def test_openai_wrapper_returns_valid_chat_completion_with_content(
    openai_model: Model,
    openai_wrapper: OpenaiWrapper,
    example_messages: Sequence[Message],
) -> None:
    response = openai_wrapper.chat_complete(
        messages=example_messages, model=openai_model, temperature=0
    )

    message = response.choices[0].message
    assert message.content is not None
    assert "paris" in message.content.lower()
    assert message.tool_calls is None


def test_openai_wrapper_returns_valid_chat_completion_with_tool_call(
    openai_model: Model,
    openai_wrapper: OpenaiWrapper,
    example_messages: Sequence[Message],
    example_function_key: str,
    example_functions: Sequence[Mapping[str, Any]],
) -> None:
    response = openai_wrapper.chat_complete_with_tools(
        messages=example_messages,
        model=openai_model,
        tools=example_functions,
        temperature=0,
    )

    assert response.choices[0].message.content is None
    assert response.choices[0].message.tool_calls is not None
    assert len(response.choices[0].message.tool_calls) == 1

    parsed: Mapping[str, str] = json.loads(
        response.choices[0].message.tool_calls[0].function.arguments
    )
    assert "Paris" in parsed.get(example_function_key, "")
