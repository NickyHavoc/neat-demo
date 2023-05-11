from langchain import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.agents.react.base import DocstoreExplorer


class Brain:
    def __init__(
        self
    ):
        self.tools = None

    def process(self, user_message: str):
        pass