import json
import os

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette import EventSourceResponse
from dotenv import load_dotenv

from neat_ai_assistant.agent import (
    NeatAgent,
    ConversationHistory,
    DocumentSearchTool,
    DuckDuckGoSearchTool,
    QueryConversationHistoryTool,
    WeatherRetrievalTool
)
from neat_ai_assistant.documents import DocumentMinion
from neat_ai_assistant.llm import LLMWrapper


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
ALEPH_ALPHA_TOKEN = os.getenv("ALEPH_ALPHA_TOKEN")
OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
OPEN_WEATHER_MAP_API_KEY = os.getenv("OPEN_WEATHER_MAP_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

llm_wrapper = LLMWrapper(
    aleph_alpha_token=ALEPH_ALPHA_TOKEN,
    open_ai_key=OPEN_AI_KEY
)
documents_path = Path(__file__).parent / "car_subsidy_documents" / "documents"
doc_minion = DocumentMinion(
    llm_wrapper,
    documents_path
)
doc_minion.instantiate_database(
    update=False
)
history = ConversationHistory()
tools = [
    DuckDuckGoSearchTool(),
    DocumentSearchTool.from_document_minion(
        document_minion=doc_minion
    ),
    QueryConversationHistoryTool(
        history=history
    ),
    WeatherRetrievalTool(
        open_weather_map_api_key=OPEN_WEATHER_MAP_API_KEY
    )
]
agent = NeatAgent(
    tools=tools,
    history=history,
    llm_wrapper=llm_wrapper,
    model="gpt-4"
)


@app.get("/chat")
async def chat(user_message: str):
    async def event_generator():
        for message in agent.reply_to(user_message):
            yield json.dumps(message.model_dump())
    return EventSourceResponse(event_generator())
