import re
import os
import requests

from typing import Dict, List, Literal
from duckduckgo_search import DDGS
from pydantic import BaseModel
from geopy.geocoders import Nominatim
from dotenv import load_dotenv

from .conversation_history import ConversationHistory
from ..documents.documents import DocumentMinion
from ..llm.open_ai_abstractions import OpenAIChatRequestFunctionCall


class ToolParam(BaseModel):
    name: str
    type: Literal["string", "number", "integer",
                  "object", "array", "boolean", "null"]
    description: str
    required: bool


# Defining certain standard params fro standard tools
tool_param_query = ToolParam(
    name="query",
    type="string",
    description="The query to call the tool with.",
    required=True
)
tool_param_n = ToolParam(
    name="n",
    type="integer",
    description="The number of (search) results to obtain. Set to higher value for greater hit rate. Default: 5.",
    required=True)
tool_param_n_history = ToolParam(
    name="n",
    type="integer",
    description="The number of last messages to retrieve. Default: 4.",
    required=True
)
tool_param_location_weather = ToolParam(
    name="location",
    type="string",
    description="The location to get the weather for.",
    required=True
)


class ToolResult(BaseModel):
    results: List[str]
    source: str

    def get_as_string(self) -> str:
        return "SOURCE: {source}\nRESULTS:\n{results}".format(
            source=self.source,
            results="\n\n".join(self.results)
        )


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
        return self.build_tool_result([])

    def get_as_str_for_prompt(self) -> str:
        return f"{self.name} – {self.description}"

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
        return OpenAIChatRequestFunctionCall(
            name=self.serialized_name,
            description=self.description,
            parameters=self._serialize_params_to_json(require_reasoning)
        )

    def build_tool_result(self, results: List[str]) -> ToolResult:
        if not bool(results):
            results = ["This tool run did not yield a result."]
        return ToolResult(results=results, source=self.name)


class DocumentSearchTool(Tool):
    def __init__(
        self,
        document_minion: DocumentMinion,
        name: str = "Document Search",
        description: str = "Find documents from a private document base.",
        params: List[ToolParam] = [
            tool_param_query,
            tool_param_n
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

    def run(self, json_query: dict) -> str:
        self.legal_params(json_query)
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
        description: str = "Find information directly from the internet.",
        params: List[ToolParam] = [
            tool_param_query,
            tool_param_n
        ]
    ):
        super().__init__(name, description, params)

    def run(self, json_query: dict) -> str:
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

        return self.build_tool_result(results)


class RetrieveConversationHistoryTool(Tool):
    def __init__(
        self,
        history: ConversationHistory,
        name: str = "Retrieve Conversation History",
        description: str = "If a question is lacking context, retrieve the prior conversation history to gather more information.",
        params: List[ToolParam] = [
            tool_param_n_history
        ]
    ):
        super().__init__(name, description, params)
        self.history = history

    def run(self, json_query: dict) -> str:
        self.legal_params(json_query)
        # Idea: retrieve conversation history by embeddings of messages.
        # Disadvantage: may lose context between messages
        results = self.history.get_as_string_list(n=json_query["n"])
        return self.build_tool_result(results)


class WeatherRetrievalTool(Tool):
    def __init__(
        self,
        name: str = "Weather Retrieval API",
        description: str = "Retrieve the current weather for a location.",
        params: List[ToolParam] = [
            tool_param_location_weather
        ]
    ):
        super().__init__(name, description, params)
        load_dotenv()
        self.api_key = os.getenv("WEATHER_API_KEY")
        self.url = "https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude={part}&appid={api_key}"

    @staticmethod
    def _get_coordinates(place_name: str):
        geolocator = Nominatim(user_agent="my-app")
        location = geolocator.geocode(place_name)

        if location:
            return location.latitude, location.longitude
        else:
            return None

    def _get_weather_for_coordinates(self, lat: float, lon: float):
        request_url = self.url.format(
            lat=str(lat),
            lon=str(lon),
            part="[]",
            api_key=self.api_key
        )
        response = requests.get(request_url)
        print("")

    def run(self, json_query: dict) -> str:
        self.legal_params(json_query)

        lat, lon = self._get_coordinates(json_query["location"])
        weather = self._get_weather_for_coordinates(lat, lon)

        # Idea: retrieve conversation history by embeddings of messages.
        # Disadvantage: may lose context between messages
        results = self.history.get_as_string_list(n=json_query["n"])
        return self.build_tool_result(results)
