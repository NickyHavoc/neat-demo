import json
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence, cast

from dotenv import load_dotenv
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from pydantic import BaseModel

from ..llm.openai_wrapper import Message, Model, OpenaiWrapper
from .conversation_history import ConversationHistory
from .tool import Tool, ToolResult

load_dotenv()


class NeatAgentOutput(BaseModel):
    type: Literal["thought", "function_call", "answer"]
    text: Optional[str]


class _Function:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments
        self.json = self._parse_arguments(arguments)

    @classmethod
    def _from_openai_object(
        cls, chat_completion_message_tool_call: ChatCompletionMessageToolCall
    ) -> "_Function":
        return cls(
            name=chat_completion_message_tool_call.function.name,
            arguments=chat_completion_message_tool_call.function.arguments,
        )

    @staticmethod
    def _parse_arguments(arguments: str) -> Mapping[str, Any]:
        try:
            return cast(Mapping[str, Any], json.loads(arguments))
        except json.JSONDecodeError:
            return {}

    def get_arguments_except(self, except_args: Sequence[str]) -> Mapping[str, Any]:
        return {k: v for k, v in self.json.items() if k not in except_args}


class _ChatCompletionMessage:
    def __init__(
        self,
        role: Literal["assistant"],
        content: Optional[str],
        functions: Sequence[_Function],
    ) -> None:
        self.role = role
        self.content = content
        self.functions = functions

    @classmethod
    def from_openai_object(
        cls, chat_completion_message: ChatCompletionMessage
    ) -> "_ChatCompletionMessage":
        return cls(
            role=chat_completion_message.role,
            content=chat_completion_message.content,
            functions=[
                _Function._from_openai_object(tool_call)
                for tool_call in (chat_completion_message.tool_calls or [])
            ],
        )

    def to_message(self) -> Message:
        return Message(
            role=self.role,
            content=self.content or "\n\n".join(f.arguments for f in self.functions),
        )


class _ReplyState(BaseModel):
    messages: list[Message] = []
    tool_results_list: list[Sequence[ToolResult]] = []
    final_answer: Optional[str] = None

    @classmethod
    def from_system_message(cls, system_message: Message) -> "_ReplyState":
        return cls(messages=[system_message])

    def get_last_tool_results(self) -> Sequence[ToolResult]:
        return self.tool_results_list[-1] if self.tool_results_list else []

    def add_message(self, message: Message) -> None:
        self.messages.append(message)

    def add_tool_results(self, tool_results: Sequence[ToolResult]) -> None:
        self.tool_results_list.append(tool_results)
        final_tool_result = next((t for t in tool_results if t.final), None)
        if final_tool_result:
            self.set_final_answer(final_tool_result.get_as_string())

    def set_final_answer(self, final_answer: str) -> None:
        self.final_answer = final_answer


class NeatAgent:
    REASONING_KEY = "reasoning"
    SYSTEM_MESSAGE = Message(
        role="system",
        content="""You want to find the best answer to a user question.
Answer the question using the functions you have been provided with.
As soon as you have the ability to deliver a final answer, STOP calling functions.""",
    )
    TOOL_RESPONSE_TEMPLATE = """Here is the information from the last tool use.
Remember, if this is enough information to answer the query, proceed to answer!

{tool_results}

"""
    REACT_TEMPLATE = """{tool_response}My question: {query}

Your turn!
Decide the next action to choose. Pick from the available tools.

If you think you gathered all necessary information, generate a final answer."""

    def __init__(
        self,
        openai_wrapper: OpenaiWrapper,
        tools: Sequence[Tool],
        history: ConversationHistory,
        model: Model = Model.GPT_4,
        require_reasoning: bool = True,
    ) -> None:
        self.openai_wrapper = openai_wrapper
        self.model = model

        serialized_tool_names = [t.serialized_name for t in tools]
        if len(set(serialized_tool_names)) != len(serialized_tool_names):
            raise ValueError("There is an overlap in tool names (after serializing).")
        self.tools = tools

        self.history = history
        self.require_reasoning = require_reasoning

    def reply_to(self, query: str) -> Iterable[NeatAgentOutput]:
        state = _ReplyState.from_system_message(self.SYSTEM_MESSAGE)

        while not state.final_answer:
            message = self._build_message(query, state.get_last_tool_results())
            state.add_message(message)
            self._count_message_tokens(state.messages)

            response = self.openai_wrapper.chat_complete_with_tools(
                state.messages, self.model, [t.serialize(True) for t in self.tools]
            )
            chat_completion_message = _ChatCompletionMessage.from_openai_object(
                response.choices[0].message
            )
            if chat_completion_message.functions:
                state.add_message(chat_completion_message.to_message())

                tool_results = []
                for tool_call in chat_completion_message.functions:
                    yield NeatAgentOutput(
                        type="thought", text=tool_call.json.get(self.REASONING_KEY)
                    )
                    tool_results.append(self._call_tool(tool_call))
                state.add_tool_results(tool_results)

                if not state.final_answer:
                    for tool_result in state.get_last_tool_results():
                        yield NeatAgentOutput(
                            type="function_call",
                            text=f"Query:\n{tool_call.arguments}\n\n{tool_result.get_as_string()}",
                        )

            elif chat_completion_message.content:
                state.set_final_answer(chat_completion_message.content)

            else:
                raise RuntimeError(f"Openai response could not be read.")

        self.history.add_message(Message(role="user", content=query))
        self.history.add_message(chat_completion_message.to_message())
        yield NeatAgentOutput(type="answer", text=state.final_answer)

    def _build_message(self, query: str, last_tools: Sequence[ToolResult]) -> Message:
        message_content = self.REACT_TEMPLATE.format(
            tool_response=self.TOOL_RESPONSE_TEMPLATE.format(
                tool_results="\n\n".join(tool.get_as_string() for tool in last_tools)
            ),
            query=query,
        )
        return Message(role="user", content=message_content)

    def _count_message_tokens(
        self,
        messages: list[Message],
    ) -> list[Message]:
        ommited_messages: list[Message] = []
        count = self.openai_wrapper.open_ai_count_tokens(messages, self.model)
        while count > 4096:
            ommited_messages.append(messages.pop(1))
            count = self.openai_wrapper.open_ai_count_tokens(messages, self.model)
        return ommited_messages

    def _call_tool(self, function_helper: _Function) -> ToolResult:
        tool_to_use = next(
            (
                t
                for t in self.tools
                if function_helper.name in [t.name, t.serialized_name]
            ),
            None,
        )
        if tool_to_use is not None:
            arguments = function_helper.get_arguments_except([self.REASONING_KEY])
            return tool_to_use.run(arguments)
        return ToolResult(results=[], source=function_helper.name)
