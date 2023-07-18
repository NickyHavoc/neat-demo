import os

from dotenv import load_dotenv

from neat_ai_assistant.llm import LLMWrapper


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
