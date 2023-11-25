import re
from abc import abstractmethod
from typing import Any, Literal, Mapping, Optional, Sequence, final

from pydantic import BaseModel


class ToolParam(BaseModel):
    name: str
    type: Literal["string", "number", "integer", "object", "array", "boolean", "null"]
    description: str
    required: bool
    enum: Optional[Sequence[str]] = None


class ToolResult(BaseModel):
    source: str
    results: Sequence[str]
    final: bool = False

    def get_as_string(self) -> str:
        return "Source: {source}\n\nResults:\n{results}".format(
            source=self.source, results="\n\n".join(self.results)
        )


class Tool:
    def __init__(self, name: str, description: str, params: Sequence[ToolParam]):
        self.name = name
        self.serialized_name = self._get_serializable_function_name()
        self.description = description
        self.params = params

    def _get_serializable_function_name(self) -> str:
        transformed_string = self.name.replace(" ", "_")
        transformed_string = re.sub(r"[^\w-]", "", transformed_string)
        transformed_string = transformed_string[:64]
        return transformed_string.lower()

    @abstractmethod
    def _run(self, json_query: Mapping[str, Any]) -> ToolResult:
        ...

    @final
    def run(self, json_query: Mapping[str, Any]) -> ToolResult:
        self.legal_params(json_query)
        return self._run(json_query)

    @final
    def legal_params(self, json_query: Mapping[str, Any]) -> None:
        received_params = set(json_query.keys())
        expected_params = set(p.name for p in self.params)
        if not expected_params.issubset(received_params):
            missing_params = expected_params - received_params
            raise ValueError(
                f"Missing parameters: {', '.join(missing_params)}. "
                f"Received parameters: {', '.join(received_params)}."
            )

    @final
    def serialize(self, require_reasoning: bool) -> Mapping[str, Any]:
        def build_param_json(
            type: str, description: str, enum: Optional[Sequence[str]] = None
        ) -> Mapping[str, Any]:
            return {"type": type, "description": description} | (
                {"enum": enum} if enum else {}
            )

        return {
            "type": "function",
            "function": {
                "name": self.serialized_name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        **{
                            p.name: build_param_json(p.type, p.description, p.enum)
                            for p in self.params
                        },
                        **(
                            {
                                "reasoning": build_param_json(
                                    "string",
                                    "Why did you decide to take this action? Explain your thoughts.",
                                )
                            }
                            if require_reasoning
                            else {}
                        ),
                    },
                    "required": [p.name for p in self.params if p.required]
                    + (["reasoning"] if require_reasoning else []),
                },
            },
        }

    @final
    def to_result(self, results: Sequence[str], final: bool = False) -> ToolResult:
        if not bool(results):
            results = ["This tool run did not yield a result."]
        return ToolResult(source=self.name, results=results, final=final)
