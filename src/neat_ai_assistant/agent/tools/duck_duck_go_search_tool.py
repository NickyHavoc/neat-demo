from typing import Any, Dict, Mapping, Sequence

from duckduckgo_search import DDGS  # type: ignore

from ..tool import Tool, ToolParam, ToolResult

TOOL_PARAM_N = ToolParam(
    name="n",
    type="integer",
    description="The number of pages to obtain. Set to higher value for greater hit rate. Default: 5.",
    required=True,
)
TOOL_PARAM_QUERY = ToolParam(
    name="query",
    type="string",
    description="A query to search the internet with.",
    required=True,
)


class DuckDuckGoSearchTool(Tool):
    """
    Uses DuckDuckGo search engine to query the internet. Only retrieves descriptive texts for hits.
    Params:
    - n (int): number of pages
    - query (str): search query
    """

    def __init__(
        self,
        name: str = "DuckDuckGo Search Engine",
        description: str = "Find information directly from the internet.",
        params: Sequence[ToolParam] = [TOOL_PARAM_N, TOOL_PARAM_QUERY],
    ) -> None:
        super().__init__(name, description, params)

    def _run(self, json_query: Mapping[str, Any]) -> ToolResult:
        def construct_result_string(r: Dict[str, str]) -> str:
            return "{title}\n{body}".format(title=r["title"], body=r["body"])

        results = []
        with DDGS(timeout=5) as ddgs:
            for i, r in enumerate(ddgs.text(json_query["query"])):
                if i >= json_query["n"]:
                    break
                results.append(construct_result_string(r))

        return self.to_result(results)
