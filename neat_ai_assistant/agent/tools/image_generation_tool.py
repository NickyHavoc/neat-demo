from pydantic import BaseModel
import requests

from typing import List, Optional, Tuple
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy.location import Location

from ..tool import Tool, ToolParam, ToolResult
from ...llm.llm_wrapper import LLMWrapper


TOOL_PARAM_PROMPT = ToolParam(
    name="prompt",
    type="string",
    description="Thr prompt that should be used to generate an image. Make it as detailed as possible.",
    required=True
)


class ImageGenerationTool(Tool):
    def __init__(
        self,
        llm_wrapper: LLMWrapper,
        name: str = "Image Generation API",
        description: str = "API to generate an image given a prompt.",
        params: List[ToolParam] = [
            TOOL_PARAM_PROMPT
        ]
    ):
        super().__init__(name, description, params)
        self.llm_wrapper = llm_wrapper

    def run(self, json_query: dict) -> ToolResult:
        self.legal_params(json_query)

        results = []

        return self.build_tool_result(results)
