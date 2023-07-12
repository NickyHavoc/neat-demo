from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from neat_ai_assistant.documents.documents import DocumentMinion
from neat_ai_assistant.agent.agent import NeatAgent
from neat_ai_assistant.agent.tools import DocumentSearchTool, DuckDuckGoSearchTool


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
tools = [
    DocumentSearchTool.from_document_minion(
        document_minion=doc_minion
    ),
    DuckDuckGoSearchTool(),
]
brain = NeatAgent(tools=tools)


class IncomingMessage(BaseModel):
    message: str


@app.post("/send-message")
async def send_message(message: IncomingMessage):
    agent_message = brain.reply_to(message_string=message)
    return {"message": agent_message}
