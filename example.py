import os
from pathlib import Path

from documents import DocumentHandler
from brain import Brain, DocumentSearchTool


documents_path = Path(__file__).parent / "example_docs"
doc_handler = DocumentHandler()
doc_handler.instantiate_database(
    documents_path=documents_path,
    update=False
)
search_tool = DocumentSearchTool(
    document_handler=doc_handler,
    name="Search",
    description="useful for retrieving information about EON."
)
brain = Brain(tools=[search_tool])

while True:
    user_message = input()
    answer = brain.reply_to(user_message=user_message)
    print(answer)
