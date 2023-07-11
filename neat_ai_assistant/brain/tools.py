import re
from typing import Dict, List, Union
from duckduckgo_search import DDGS
from pydantic import BaseModel

from ..documents.documents import DocumentMinion


class ToolResult(BaseModel):
    results: List[str]
    source: str

    def get_for_prompt(self) -> str:
        return "Quelle: {source}\n\nResults:\n\n{results}".format(
            source=self.source,
            results="\n\n".join(self.results)
        )


class Tool:
    def __init__(
        self,
        name: str,
        description: str
    ):
        self.name = name
        self.serialized_name = self._get_serializable_function_name()
        self.description = description

    def run(self, json_query: dict) -> ToolResult:
        """
        Runs a json_query for the specific tool. Will return ToolResult
        Standard json_query:
        - query: str (the actual query)
        - n: int (the desired number of results)
        """

    def get_for_prompt(self) -> str:
        return f"{self.name} – {self.description}"

    def _get_serializable_function_name(self) -> str:
        transformed_string = self.name.replace(' ', '_')
        transformed_string = re.sub(r'[^\w-]', '', transformed_string)
        transformed_string = transformed_string[:64]
        return transformed_string.lower()

    def get_for_function_call(self) -> dict:
        """
        Returns a json object that can be used to call the OpenAI API and retrieve a json.
        """
        return {
            "name": self.serialized_name,
            "description": "{self.description}",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to call the tool with.",
                    },
                    "n": {
                        "type": "integer",
                        "description": "The number of (search) results to obtain. Set to higher value for greater hit rate. Default: 1."},
                },
                "required": [
                    "query",
                    "n"],
            },
        }

    def build_tool_result(self, results: List[str]):
        if not bool(results):
            results = ["This tool run did not yield a result."]
        return ToolResult(results=results, source=self.name)


class DocumentSearchTool(Tool):
    def __init__(
        self,
        document_minion: DocumentMinion,
        name: str = "Document Search",
        description: str = "Find documents from a private document base."
    ):
        super().__init__(name, description)
        self.document_minion = document_minion

    @classmethod
    def from_document_minion(cls, document_minion: DocumentMinion):
        return cls(
            document_minion=document_minion,
            name=document_minion.name,
            description=document_minion.description
        )

    def run(self, json_query: dict) -> str:
        results = self.document_minion.search(json_query["query"])
        formatted_results = [
            r.best_chunks[0][0].content for r in results[:json_query["n"]]] if bool(results) else []
        # Currently returns only best chunk per document -> desired?
        return self.build_tool_result(
            formatted_results
        )


class DuckDuckGoSearchTool(Tool):
    def __init__(
        self,
        name: str = "DuckDuckGo Search Engine",
        description: str = "Find information directly from the internet."
    ):
        super().__init__(name, description)

    def run(self, json_query: dict) -> str:

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

        return self.build_tool_result(results)
