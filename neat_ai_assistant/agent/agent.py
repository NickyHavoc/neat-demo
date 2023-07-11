import json

from typing import Dict, List, Optional

from .tools import Tool, ToolResult
from ..utils.api_wrapper import LLMWrapper, OpenAIChatCompletion, OpenAIMessage


class NeatAgent:
    def __init__(
        self,
        tools: List[Tool],
    ):
        if not all(isinstance(t, Tool) for t in tools):
            raise TypeError("All tools must be of type tool.")
        serialized_tool_names = [t.serialized_name for t in tools]
        if len(set(serialized_tool_names)) != len(serialized_tool_names):
            raise ValueError(
                "There is an overlap in tool names (after serializing).")
        self.tools = tools
        self.llm_wrapper = LLMWrapper()

    def _get_tool_by_name(self, name: str):
        for t in self.tools:
            if name in [
                t.name,
                t.serialized_name
            ]:
                return t

    def _get_tools_for_prompt(self) -> str:
        return "\n".join(
            t.get_for_prompt() for t in self.tools
        )

    def _get_tools_for_function_call(self) -> List[dict]:
        function_call_jsons = [t.get_for_function_call() for t in self.tools]
        for fcj in function_call_jsons:
            fcj["parameters"]["properties"].update(
                {
                    "thought": {
                        "type": "string",
                        "description": "What do you think is a good next action to take? Describe your thoughts."
                    }
                }
            )
            fcj["parameters"]["required"].append("thought")
        return function_call_jsons

    def _build_request(
        self,
        messages: List[Dict[str, str]],
    ) -> dict:
        system_message = OpenAIMessage(role="system", content="Only use the functions you have been provided with.")
        messages = [system_message] + messages
        return {
            "model": "gpt-4",
            "messages": messages,
            "functions": self._get_tools_for_function_call()
        }

    def reply_to(self, user_message: str):
        final_answer: Optional[str] = None
        messages = []
        tool_results: List[ToolResult] = []

        while not final_answer:
            message_content = f"""My question: {user_message}

Your turn!
Decide the next action to choose. Pick from the available tools.

If you think you gathered all necessary information, generate a final answer"""
            if bool(tool_results):
                message_content = (
                    "Here is the information from the last tool use:\n"
                    + tool_results[-1].get_for_prompt()
                    + "\n\n"
                    + message_content
                )

            message = OpenAIMessage(role="user", content=message_content)
            messages.append(message)
            request = self._build_request(messages)

            completion = self.llm_wrapper.open_ai_chat_complete(params=request)
            choice = completion.choices[0]
            if choice.finish_reason == "function_call":
                messages.append(choice.message)
                function_call = choice.message.function_call
                thought_key = "thought"
                yield function_call.arguments[thought_key]

                tool_to_use = self._get_tool_by_name(function_call.name)

                if bool(tool_to_use):
                    arguments = function_call.get_args_except([thought_key])
                    tool_result = tool_to_use.run(arguments)

                else:
                    tool_result = ToolResult(results=[], source=function_call.name)

                tool_results.append(tool_result)
                yield tool_result.get_for_prompt()

            else:
                final_answer = choice.message.content

        return final_answer
