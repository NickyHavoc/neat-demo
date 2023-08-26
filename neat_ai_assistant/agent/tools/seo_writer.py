from os import stat
from typing import Dict, List
from duckduckgo_search import DDGS

from neat_ai_assistant.llm.open_ai_abstractions import ChatRequest

from ..tool import Tool, ToolParam, ToolResult

from ...llm import LLMWrapper


TOOL_PARAM_PRODUCT = ToolParam(
    name="product",
    type="string",
    description="Product to write the article for.",
    required=True
)
TOOL_PARAM_KEYWORDS = ToolParam(
    name="keywords",
    type="string",
    description="A comma-separated list of keywords that are relevant for the article.",
    required=True
)
TOOL_PARAM_STRUCTURE = ToolParam(
    name="structure",
    type="string",
    description="Approximate length, structural requirements, language requirements, writing advice for the article.",
    required=True
)
TOOL_PARAM_EXAMPLES = ToolParam(
    name="examples",
    type="string",
    description="Examples of great sentences or short text excerpts that should be considered.",
    required=True
)


class SEOWriter(Tool):
    def __init__(
        self,
        llm_wrapper: LLMWrapper,
        company_name: str,
        company_description: str,
        name: str = "SEO Writer",
        description: str = "Write a SEO relevant article with some information.",
        params: List[ToolParam] = [
            TOOL_PARAM_KEYWORDS,
            TOOL_PARAM_PRODUCT,
            TOOL_PARAM_STRUCTURE,
            TOOL_PARAM_EXAMPLES
        ]
    ):
        super().__init__(name, description, params)
        self.llm_wrapper = llm_wrapper
        self.company_name = company_name
        self.company_desciption = company_description

    # @staticmethod
    # def _build_faq(faq_string: str):
    #     segments = faq_string.strip().split("\n\n")
    #     qa_list = []
    #     for segment in segments:
    #         lines = segment.split("\n")
    #         question = lines[0].replace("Question: ", "").strip()
    #         answer = lines[1].replace("Answer: ", "").strip()
    #         qa_list.append({"question": question, "answer": answer})

    def run(self, json_query: dict) -> ToolResult:
        self.legal_params(json_query)

        messages = [
            {
                "role": "system",
                "content": f"""You are a SEO writing engine. Based on some inputs, you write an article that advertises a given product and is readable and includes relevant keywords and more. Use markdown notation.
You work for {self.company_name}. {self.company_desciption}
Do not mention the competition."""
            },
            {
                "role": "user",
                "content": f"""Please write a SEO text about this:
Product: {json_query["product"]}
Relevant keywords: {json_query["keywords"]}
Other requirements: {json_query["structure"]}
Examples from other websites: {json_query["examples"]}

Please use this output format:
```
# Your headline
Your SEO article

# FAQ
**Question**: 'each question'
**Answer**: 'each answer'
```"""
            }
        ]
        response = self.llm_wrapper.open_ai_chat_complete(
            request=ChatRequest.from_json(
                {
                    "messages": messages,
                    "model": "gpt-3.5-turbo"
                }
            )
        )

        return self._build_tool_result([response.completions[0].message.content], final=True)