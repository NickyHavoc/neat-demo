import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette import EventSourceResponse

from neat_ai_assistant.documents.documents import DocumentMinion
from neat_ai_assistant.agent.agent import NeatAgent
from neat_ai_assistant.agent.conversation_history import ConversationHistory
from neat_ai_assistant.agent.tools import DocumentSearchTool, DuckDuckGoSearchTool, RetrieveConversationHistoryTool


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

documents_path = Path(__file__).parent / "example_docs"
doc_minion = DocumentMinion()
doc_minion.instantiate_database(
    documents_path=documents_path,
    update=False
)
history = ConversationHistory()
tools = [
    DuckDuckGoSearchTool(),
    DocumentSearchTool.from_document_minion(
        document_minion=doc_minion
    ),
    RetrieveConversationHistoryTool(
        history=history
    )
]
agent = NeatAgent(
    tools=tools,
    history=history,
    model="gpt-3.5-turbo"
)


@app.get("/bot")
async def chat(user_message: str):
    async def event_generator():
        for message in agent.reply_to(user_message):
            yield json.dumps(message.model_dump())
    return EventSourceResponse(event_generator())
