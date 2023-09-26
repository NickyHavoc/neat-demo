from typing import Sequence

from ..llm import Message


class ConversationHistory:
    def __init__(self):
        self.history: Sequence[Message] = []

    def add_message(self, message: Message) -> None:
        self.history.append(message)

    def get(self) -> Sequence[Message]:
        return self.history

    def get_as_string_list(self, n: int) -> list[str]:
        def build_message_string(m: Message):
            return f"{m.role}: {m.content}"

        return [build_message_string(m) for m in self.history[-n:]]
