from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from configurable_chat_agent import AgentHangout


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent_hangout = AgentHangout()


class IncomingMessage(BaseModel):
    message: str


@app.post("/send-message")
async def send_message(message: IncomingMessage):
    agent_message = agent_hangout.produce_answer()
    return {"message": agent_message}
