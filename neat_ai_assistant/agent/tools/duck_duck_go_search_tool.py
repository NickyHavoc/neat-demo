from typing import Dict, List
from duckduckgo_search import DDGS

from ..tool import Tool, ToolParam, ToolResult


TOOL_PARAM_N = ToolParam(
    name="n",
    type="integer",
    description="The number of pages to obtain. Set to higher value for greater hit rate. Default: 5.",
    required=True)
TOOL_PARAM_QUERY = ToolParam(
    name="query",
    type="string",
    description="A query to search the internet with.",
    required=True
)


class DuckDuckGoSearchTool(Tool):
    def __init__(
        self,
        name: str = "DuckDuckGo Search Engine",
        description: str = "Find information directly from the internet.",
        params: List[ToolParam] = [
            TOOL_PARAM_N,
            TOOL_PARAM_QUERY
        ]
    ):
        super().__init__(name, description, params)

    def run(self, json_query: dict) -> ToolResult:
        self.legal_params(json_query)

        def construct_result_string(r: Dict[str, str]) -> str:
            return "{title}\n{body}".format(
                title=r["title"],
                body=r["body"]
            )

        results = []
        with DDGS(timeout=5) as ddgs:
            for i, r in enumerate(ddgs.text(json_query["query"])):
                if i >= json_query["n"]:
                    break
                results.append(construct_result_string(r))

        return self._build_tool_result(results)
