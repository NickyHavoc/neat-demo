import os
from pathlib import Path

from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool

from documents import DocumentHandler
from brain import DocumentSearchTool


documents_path = Path(__file__).parent / "example_docs"
doc_handler = DocumentHandler()
doc_handler.instantiate_database(
    documents_path=documents_path,
    update=False
)

llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPEN_AI_KEY"))
search_tool = DocumentSearchTool(document_handler=doc_handler)
agent = initialize_agent([search_tool], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)


while True:
    message = input()
    answer = agent.run(message)
    print(answer)
