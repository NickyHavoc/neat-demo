import os
from pathlib import Path

from langchain.tools import DuckDuckGoSearchRun

from documents import DocumentMinion
from brain import Brain, DocumentSearchTool


documents_path = Path(__file__).parent / "example_docs"
doc_minion = DocumentMinion()
doc_minion.instantiate_database(
    documents_path=documents_path,
    update=False
)
tools = [
    DuckDuckGoSearchRun(),
    DocumentSearchTool.from_document_minion(
        document_minion=doc_minion
    )
]
brain = Brain(tools=tools)

while True:
    user_message = input()
    answer = brain.reply_to(user_message=user_message)
    print(answer)
