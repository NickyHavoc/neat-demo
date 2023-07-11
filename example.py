from pathlib import Path

from langchain.tools import DuckDuckGoSearchRun

from neat_ai_assistant.documents.documents import DocumentMinion
from neat_ai_assistant.brain.brain import Brain
from neat_ai_assistant.brain.tools import DocumentSearchTool


documents_path = Path(__file__).parent / "example_docs"
doc_minion = DocumentMinion()
doc_minion.instantiate_database(
    documents_path=documents_path,
    update=True
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
