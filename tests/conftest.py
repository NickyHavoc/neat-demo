from dotenv import load_dotenv
from pytest import fixture

from neat_ai_assistant import Model, OpenaiWrapper

load_dotenv


@fixture
def openai_model() -> Model:
    return Model.GPT_3_5


@fixture(scope="session")
def openai_wrapper() -> OpenaiWrapper:
    return OpenaiWrapper()
