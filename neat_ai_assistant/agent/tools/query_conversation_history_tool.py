from typing import List
from ..tool import Tool, ToolParam, ToolResult
from ..conversation_history import ConversationHistory


TOOL_PARAM_N = ToolParam(
    name="n",
    type="integer",
    description="The number of last messages to retrieve. Default: 4.",
    required=True
)


class QueryConversationHistoryTool(Tool):
    def __init__(
        self,
        history: ConversationHistory,
        name: str = "Retrieve Conversation History",
        description: str = "If a question is lacking context, retrieve the prior conversation history to gather more information.",
        params: List[ToolParam] = [
            TOOL_PARAM_N
        ]
    ):
        super().__init__(name, description, params)
        self.history = history

    def run(self, json_query: dict) -> ToolResult:
        self.legal_params(json_query)
        # Idea: retrieve conversation history by embeddings of messages.
        # Disadvantage: may lose context between messages
        results = self.history.get_as_string_list(n=json_query["n"])
        return self._build_tool_result(results)
