import os
from typing import Tuple, Union

from dotenv import load_dotenv

from neat_ai_assistant.llm import LLMWrapper, ChatRequest


load_dotenv()
ALEPH_ALPHA_TOKEN = os.getenv("ALEPH_ALPHA_TOKEN")
OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
OPEN_WEATHER_MAP_API_KEY = os.getenv("OPEN_WEATHER_MAP_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")


def build_llm_wrapper():
    return LLMWrapper(
        aleph_alpha_token=ALEPH_ALPHA_TOKEN,
        open_ai_key=OPEN_AI_KEY
    )


def build_chat_request(
    get_contains_str: bool = False
) -> Union[ChatRequest, Tuple[ChatRequest, str]]:
    contains_str = "I am an AI"
    request = ChatRequest.from_json(
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You do exactly what the user is saying."
                },
                {
                    "role": "user",
                    "content": f"Speak after me: \"{contains_str}\"."
                }
            ],
            "model": "gpt-3.5-turbo",
            "temperature": 0
        }
    )
    if get_contains_str:
        return request, contains_str
    return request
