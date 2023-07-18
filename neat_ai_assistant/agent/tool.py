import re

from typing import List, Literal, Optional
from pydantic import BaseModel

from ..llm.open_ai_abstractions import ChatRequestFunctionCall


class ToolParam(BaseModel):
    name: str
    type: Literal["string", "number", "integer",
                  "object", "array", "boolean", "null"]
    description: str
    required: bool


class ToolResult(BaseModel):
    source: str
    results: Optional[List[str]]
    image: Optional[bytes] = None
    final: bool = False
    # add a bool "final" value that is False by default

    def get_as_string(self) -> str:
        if bool(self.results):
            return "Source: {source}\n\nResults:\n{results}".format(
                source=self.source,
                results="\n\n".join(self.results)
            )
        elif bool(self.image):
            return "Source: {source}\n\nResult is an image.".format(
                source=self.source,
            )
        raise ValueError("appears to be empty result")


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        params: List[ToolParam]
    ):
        self.name = name
        self.serialized_name = self._get_serializable_function_name()
        self.description = description
        self.params = params

    def legal_params(self, json_query: dict) -> None:
        received_params = set(json_query.keys())
        expected_params = set(p.name for p in self.params)
        if not expected_params.issubset(received_params):
            missing_params = expected_params - received_params
            raise ValueError(
                f"Missing parameters: {', '.join(missing_params)}. "
                f"Received parameters: {', '.join(received_params)}."
            )

    def run(self, json_query: dict) -> ToolResult:
        """
        Runs a json_query for the specific tool. Will return ToolResult.
        """
        # First, let's check if the params are legal...
        self.legal_params(json_query)
        # logic...
        return self._build_tool_result([])

    def _get_serializable_function_name(self) -> str:
        transformed_string = self.name.replace(' ', '_')
        transformed_string = re.sub(r'[^\w-]', '', transformed_string)
        transformed_string = transformed_string[:64]
        return transformed_string.lower()

    def _serialize_params_to_json(self, require_reasoning: bool):
        def build_param_json(type: str, description: str):
            return {"type": type, "description": description}

        return {
            "type": "object",
            "properties": {
                **{p.name: build_param_json(p.type, p.description) for p in self.params},
                **({"reasoning": build_param_json("string", "Why did you decide to take this action? Explain your thoughts.")} if require_reasoning else {})
            },
            "required": [p.name for p in self.params if p.required] + (["reasoning"] if require_reasoning else [])
        }

    def get_as_request_for_function_call(
            self, require_reasoning: bool) -> dict:
        """
        Returns a OpenAIChatRequestFunctionCall object that can be used to call the OpenAI API and retrieve a function call object.
        """
        return ChatRequestFunctionCall(
            name=self.serialized_name,
            description=self.description,
            parameters=self._serialize_params_to_json(require_reasoning)
        )

    def _build_tool_result(self, results: Optional[List[str]]) -> ToolResult:
        if not bool(results):
            results = ["This tool run did not yield a result."]
        return ToolResult(source=self.name, results=results)
    
    def _build_image_tool_result(self, image: Optional[bytes]) -> ToolResult:
        if not bool(image):
            results = ["This tool run did not yield a result."]
            final = False
        else:
            results = None
            final = True
        return ToolResult(source=self.name, results=results, image=image, final=final)
