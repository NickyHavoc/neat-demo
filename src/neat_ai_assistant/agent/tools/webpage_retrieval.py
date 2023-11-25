import re
from typing import Any, Mapping, Optional, Sequence

import requests
from bs4 import BeautifulSoup, Tag
from duckduckgo_search import DDGS  # type: ignore

from ..tool import Tool, ToolParam, ToolResult

TOOL_PARAM_N = ToolParam(
    name="n",
    type="integer",
    description="The number of detailed pages to obtain. Set to higher value for greater hit rate. Default: 5.",
    required=True,
)
TOOL_PARAM_QUERY = ToolParam(
    name="query",
    type="string",
    description="The query to get results for.",
    required=True,
)


class WebpageRetrievalTool(Tool):
    def __init__(
        self,
        name: str = "Webpage Retrieval Engine",
        description: str = "Search with a string to retrieve detailed webpages for this query.",
        params: Sequence[ToolParam] = [TOOL_PARAM_N, TOOL_PARAM_QUERY],
    ) -> None:
        super().__init__(name, description, params)

    @staticmethod
    def _scrape_body_text(url: str, maximum_length_char: int = 1000) -> Optional[str]:
        def clean_scraped_text(text: str) -> str:
            cleaned_text = text.replace("\xa0", " ")
            html_entities = re.compile(r"&[a-zA-Z]+;")
            cleaned_text = html_entities.sub(" ", cleaned_text)
            cleaned_text = re.sub(r"\n+", "\n", cleaned_text)
            return cleaned_text.strip()

        def cut_text(text: str) -> str:
            return text[:maximum_length_char] + "..."

        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        headers: Sequence[Tag] = soup.find_all(["h2", "h3", "h4", "h5", "h6"])
        if not bool(headers):
            return None

        sections: list[str] = []

        for header in headers:
            section_text: list[str] = [header.text]
            for sibling in header.find_all_next():
                if isinstance(sibling, Tag):
                    if sibling.name and sibling.name.startswith("h"):
                        break
                    elif sibling.name == "p":
                        section_text.append(sibling.text)

            if len(section_text) > 1:
                sections.append("\n".join(section_text))
        body_text: str = "\n\n".join(sections)

        return cut_text(clean_scraped_text(body_text))

    def _run(self, json_query: Mapping[str, Any]) -> ToolResult:
        self.legal_params(json_query)

        def construct_result_string(title: str, body: str) -> str:
            return "{title}\n\n{body}".format(title=title, body=body)

        prelim_results: list[Mapping[str, str | None]] = []
        with DDGS(timeout=5) as ddgs:
            for i, r in enumerate(ddgs.text(json_query["query"])):
                if i >= json_query["n"]:
                    break
                prelim_results.append(r)

        results = []
        for r in prelim_results:
            long_text = self._scrape_body_text(r.get("href"))
            if not bool(long_text):
                long_text = r.get("body")
            results.append(
                construct_result_string(title=r.get("title", ""), body=long_text or "")
            )

        return self.to_result(results)
