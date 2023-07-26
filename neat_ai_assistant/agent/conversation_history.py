from typing import List

from ..llm import Message


class ConversationHistory:
    def __init__(self):
        self.history: List[Message] = []

    def add_message(self, message: Message):
        self.history.append(message)

    def get(self):
        return self.history

    def get_as_string_list(self, n: int):
        def build_message_string(m: Message):
            return f"{m.role}: {m.content}"
        return [build_message_string(m) for m in self.history[-n:]]
