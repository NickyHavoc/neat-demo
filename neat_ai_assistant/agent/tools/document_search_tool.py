from typing import List

from ..tool import Tool, ToolParam, ToolResult
from ...documents.documents import DocumentMinion


TOOL_PARAM_N = ToolParam(
    name="n",
    type="integer",
    description="The number of search results to obtain. Set to higher value for greater hit rate. Default: 3.",
    required=True)
TOOL_PARAM_QUERY = ToolParam(
    name="query",
    type="string",
    description="The query to call the document database with. Formulate verbosely and precisely.",
    required=True)


class DocumentSearchTool(Tool):
    def __init__(
        self,
        document_minion: DocumentMinion,
        name: str = "Document Search",
        description: str = "Find documents from a private document base.",
        params: List[ToolParam] = [
            TOOL_PARAM_N,
            TOOL_PARAM_QUERY
        ]
    ):
        super().__init__(name, description, params)
        self.document_minion = document_minion

    @classmethod
    def from_document_minion(cls, document_minion: DocumentMinion):
        return cls(
            document_minion=document_minion,
            name=document_minion.name,
            description=document_minion.description
        )

    def run(self, json_query: dict) -> ToolResult:
        self.legal_params(json_query)
        results = self.document_minion.search(json_query["query"])
        formatted_results = [
            r.best_chunks[0][0].content for r in results[:json_query["n"]]] if bool(results) else []
        # Currently returns only best chunk per document -> desired?
        return self._build_tool_result(
            formatted_results
        )
