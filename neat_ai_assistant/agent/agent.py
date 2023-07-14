import json

from typing import List, Literal, Optional

from pydantic import BaseModel

from .conversation_history import ConversationHistory
from .tool import Tool, ToolResult
from ..llm.llm_wrapper import LLMWrapper
from ..llm.open_ai_abstractions import OpenAIChatCompletion, OpenAIChatRequest, OpenAIMessage, OpenAIChatCompletionFunctionCall


class NeatAgentOutput(BaseModel):
    type: Literal["thought", "function_call", "answer"]
    text: str

    def __repr__(self):
        return f"\n\nTYPE: {self.type}\n\nOUTPUT: {self.text}\n\n"


class NeatAgent:
    def __init__(
        self,
        tools: List[Tool],
        history: ConversationHistory,
        llm_wrapper: LLMWrapper,
        model: Literal["gpt-3.5-turbo", "gpt-4"] = "gpt-4",
        require_reasoning: bool = True
    ):
        if not all(isinstance(t, Tool) for t in tools):
            raise TypeError("All tools must be of type tool.")
        serialized_tool_names = [t.serialized_name for t in tools]
        if len(set(serialized_tool_names)) != len(serialized_tool_names):
            raise ValueError(
                "There is an overlap in tool names (after serializing).")
        self.tools = tools
        self.llm_wrapper = llm_wrapper
        self.model = model
        self.require_reasoning = require_reasoning
        self.reasoning_key = "reasoning"

        self.system_message = OpenAIMessage(
            role="system",
            content="""You want to find the best answer to a user question. Always try to break a question down into subquestions, for example:
"What's the age of Dua Lipa's boyfriend?", Subquestions: "Who is Dua Lipa's boyfriend?", "How old is [boyfriend_name]?".
Answer the question using the functions you have been provided with.
If you have a final answer, return this instead.""")

        self.history = history

    def _get_tool_by_name(self, name: str):
        for t in self.tools:
            if name in [
                t.name,
                t.serialized_name
            ]:
                return t

    def _build_request(
        self,
        messages: List[OpenAIMessage],
    ) -> dict:
        request = OpenAIChatRequest(
            model=self.model, messages=messages, functions=[
                t.get_as_request_for_function_call(
                    self.require_reasoning) for t in self.tools])
        ommited_messages = []
        count = self.llm_wrapper.open_ai_count_tokens(request)
        while count > 4096:
            ommited_messages.append(request.messages.pop(1))
            count = self.llm_wrapper.open_ai_count_tokens(request)
        return request, ommited_messages

    def reply_to(self, message_string: str):
        final_answer: Optional[str] = None
        messages = [self.system_message]
        tool_results: List[ToolResult] = []

        while not final_answer:
            message_content = f"""My question: {message_string}

Your turn!
Decide the next action to choose. Pick from the available tools.

If you think you gathered all necessary information, generate a final answer."""
            if bool(tool_results):
                message_content = (
                    "Here is the information from the last tool use:\n"
                    + tool_results[-1].get_as_string()
                    + "\n\n"
                    + message_content
                )

            message = OpenAIMessage(role="user", content=message_content)
            messages.append(message)
            request, omitted_messages = self._build_request(messages)

            completion = self.llm_wrapper.open_ai_chat_complete(request)
            choice = completion.choices[0]
            if choice.finish_reason == "function_call":
                messages.append(choice.message)
                function_call = choice.message.function_call

                yield NeatAgentOutput(
                    type="thought",
                    text=function_call.arguments[self.reasoning_key]
                )
                tool_to_use = self._get_tool_by_name(function_call.name)

                if bool(tool_to_use):
                    arguments = function_call.get_args_except(
                        [self.reasoning_key])
                    tool_result = tool_to_use.run(arguments)

                else:
                    tool_result = ToolResult(
                        results=[], source=function_call.name)

                tool_results.append(tool_result)
                yield NeatAgentOutput(
                    type="function_call",
                    text=f"QUERY:\n{json.dumps(arguments)}\n\n{tool_result.get_as_string()}"
                )

            else:
                final_answer = choice.message.content

        self.history.add_message(OpenAIMessage(
            role="user",
            content=message_string
        ))
        self.history.add_message(choice.message)
        yield NeatAgentOutput(
            type="answer",
            text=final_answer
        )
