import os

from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool

class Brain:
    def __init__(
        self,
        tools: list
    ):
        assert all(isinstance(t, BaseTool) for t in tools)
        llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPEN_AI_KEY"))
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            #memory=memory
        )

    def reply_to(self, user_message: str):
        try:
            answer = self.agent.run(user_message)
        except Exception as e:
            answer = "I made an error contemplating your query."
        return answer
