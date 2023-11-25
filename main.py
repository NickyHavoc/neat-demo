import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette import EventSourceResponse

from neat_ai_assistant import (
    ConversationHistory,
    DuckDuckGoSearchTool,
    Model,
    NeatAgent,
    OpenaiWrapper,
    QueryConversationHistoryTool,
    WebpageRetrievalTool,
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


openai_wrapper = OpenaiWrapper()
history = ConversationHistory()
tools = [
    WebpageRetrievalTool(),
    DuckDuckGoSearchTool(),
    QueryConversationHistoryTool(history=history),
]
agent = NeatAgent(
    openai_wrapper=openai_wrapper, tools=tools, history=history, model=Model.GPT_4
)


@app.get("/chat")
async def chat(user_message: str):
    async def event_generator():
        for message in agent.reply_to(user_message):
            yield json.dumps(message.model_dump())

    return EventSourceResponse(event_generator())
