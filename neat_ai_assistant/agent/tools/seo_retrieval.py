import re
import requests

from typing import Dict, List, Optional
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup, Tag

from ..tool import Tool, ToolParam, ToolResult


TOOL_PARAM_N = ToolParam(
    name="n",
    type="integer",
    description="The number of pages to obtain. Set to higher value for greater hit rate. Default: 5.",
    required=True)
TOOL_PARAM_QUERY = ToolParam(
    name="query",
    type="string",
    description="Product to search for.",
    required=True
)


class SEORetrievalTool(Tool):
    def __init__(
        self,
        name: str = "SEO Retrieval Engine",
        description: str = "Search with a product to retrieve similar entries from the web about this product.",
        params: List[ToolParam] = [
            TOOL_PARAM_N,
            TOOL_PARAM_QUERY
        ],
    ):
        super().__init__(name, description, params)

    @staticmethod
    def _scrape_body_text(url: str, maximum_length_char: int = 1000):

        def clean_scraped_text(text: str) -> str:
            cleaned_text = text.replace("\xa0", " ")
            html_entities = re.compile(r'&[a-zA-Z]+;')
            cleaned_text = html_entities.sub(' ', cleaned_text)
            cleaned_text = re.sub(r'\n+', '\n', cleaned_text)
            return cleaned_text.strip()

        def cut_text(text: str) -> str:
            return text[:1000] + "..."

        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        headers: List[Tag] = soup.find_all(['h2', 'h3', 'h4', 'h5', 'h6'])
        if not bool(headers):
            return

        sections: List[str] = []

        for header in headers:
            section_text: List[str] = [header.text]
            for sibling in header.find_all_next():
                if sibling.name and sibling.name.startswith('h'):
                    break
                elif sibling.name == 'p':
                    section_text.append(sibling.text)

            if len(section_text) > 1:
                sections.append('\n'.join(section_text))
        body_text: str = '\n\n'.join(sections)

        return cut_text(clean_scraped_text(body_text))


    def run(self, json_query: dict) -> ToolResult:
        self.legal_params(json_query)

        def construct_result_string(title: str, body: str) -> str:
            return "{title}\n\n{body}".format(
                title=title,
                body=body
            )

        prelim_results = []
        with DDGS(timeout=5) as ddgs:
            for i, r in enumerate(ddgs.text(json_query["query"])):
                if i >= json_query["n"]:
                    break
                prelim_results.append(r)

        results = []
        for r in prelim_results:
            long_text = self._scrape_body_text(r["href"])
            if not bool(long_text):
                long_text = r["body"]
            results.append(construct_result_string(title=r["title"], body=long_text))


        return self._build_tool_result(results)
