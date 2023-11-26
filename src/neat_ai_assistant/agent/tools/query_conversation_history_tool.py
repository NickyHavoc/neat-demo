from typing import Any, Mapping, Sequence

from ..conversation_history import ConversationHistory
from ..tool import Tool, ToolParam, ToolResult

TOOL_PARAM_N = ToolParam(
    name="n",
    type="integer",
    description="The number of last messages to retrieve. Default: 4.",
    required=True,
)


class QueryConversationHistoryTool(Tool):
    """
    Injects prior chat history into the context as it is not included by default.
    Params:
    - n (int): the number of last messages to be retrieved
    """

    def __init__(
        self,
        history: ConversationHistory,
        name: str = "Retrieve Conversation History",
        description: str = "If a question is lacking context, retrieve the prior conversation history to gather more information.",
        params: Sequence[ToolParam] = [TOOL_PARAM_N],
    ) -> None:
        super().__init__(name, description, params)
        self.history = history

    def _run(self, json_query: Mapping[str, Any]) -> ToolResult:
        results = self.history.get_as_string_list(n=json_query["n"])
        return self.to_result(results)
