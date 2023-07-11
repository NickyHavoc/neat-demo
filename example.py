from pathlib import Path

from neat_ai_assistant.documents.documents import DocumentMinion
from neat_ai_assistant.brain.agent import NeatAgent
from neat_ai_assistant.brain.tools import DocumentSearchTool, DuckDuckGoSearchTool


documents_path = Path(__file__).parent / "example_docs"
doc_minion = DocumentMinion()
doc_minion.instantiate_database(
    documents_path=documents_path,
    update=True
)
tools = [
    DuckDuckGoSearchTool(),
    DocumentSearchTool.from_document_minion(
        document_minion=doc_minion
    )
]
brain = NeatAgent(tools=tools)

while True:
    user_message = input()
    answer = brain.reply_to(user_message=user_message)
    print(answer)
