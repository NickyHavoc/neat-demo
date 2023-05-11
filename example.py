import os
from pathlib import Path

from langchain.tools import DuckDuckGoSearchRun

from documents import DocumentHandler
from brain import Brain, DocumentSearchTool


documents_path = Path(__file__).parent / "example_docs"
doc_handler = DocumentHandler()
doc_handler.instantiate_database(
    documents_path=documents_path,
    update=False
)
tools = [
    DuckDuckGoSearchRun(),
    DocumentSearchTool(
        document_handler=doc_handler,
        name="EON Search",
        description="useful for retrieving information about EON."
    )
]
brain = Brain(tools=tools)

while True:
    user_message = input()
    answer = brain.reply_to(user_message=user_message)
    print(answer)
