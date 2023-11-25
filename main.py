import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette import EventSourceResponse

from neat_ai_assistant.agent import (
    ConversationHistory,
    DuckDuckGoSearchTool,
    NeatAgent,
    QueryConversationHistoryTool,
    WebpageRetrievalTool,
)
from neat_ai_assistant.llm import OpenaiWrapper

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

llm_wrapper = OpenaiWrapper(
    aleph_alpha_token=ALEPH_ALPHA_TOKEN, open_ai_key=OPEN_AI_KEY
)
history = ConversationHistory()
tools = [
    WebpageRetrievalTool(),
    DuckDuckGoSearchTool(),
    QueryConversationHistoryTool(history=history),
]
agent = NeatAgent(
    tools=tools, history=history, openai_wrapper=llm_wrapper, model="gpt-4"
)


@app.get("/chat")
async def chat(user_message: str):
    async def event_generator():
        for message in agent.reply_to(user_message):
            yield json.dumps(message.model_dump())

    return EventSourceResponse(event_generator())
