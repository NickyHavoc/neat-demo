from typing import List
from ..llm.open_ai_abstractions import OpenAIMessage


class ConversationHistory:
    def __init__(self):
        self.history: List[OpenAIMessage] = []

    def add_message(self, message: OpenAIMessage):
        self.history.append(message)

    def get(self):
        return self.history

    def get_as_string_list(self, n: int):
        def build_message_string(m: OpenAIMessage):
            return f"{m.role}: {m.content}"
        return [build_message_string(m) for m in self.history[-n:]]
