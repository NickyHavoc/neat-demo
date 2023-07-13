import json
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class OpenAIChatCompletionFunctionCall(BaseModel):
    name: str
    arguments: Dict[str, Any]

    @classmethod
    def from_json(cls, json_object: Dict[str, Union[str, Dict[str, Any]]]):
        # OpenAI sometimes generates "functions." when returning this. No idea,
        # why.
        json_object["name"] = json_object["name"].replace("functions.", "")
        json_object["arguments"] = json.loads(json_object["arguments"])
        return cls(**json_object)

    def to_open_ai_api_format(self):
        return {
            "name": self.name,
            "arguments": json.dumps(self.arguments)
        }

    def to_string(self):
        return json.dumps(self.model_dump())

    def get_args_except(self, except_args: List[str]):
        return {k: v for k, v in self.arguments.items() if k not in except_args}


class OpenAIMessage(BaseModel):
    role: str
    content: Optional[str]
    function_call: Optional[OpenAIChatCompletionFunctionCall] = None

    @classmethod
    def from_json(cls, json_object: dict):
        json_object.setdefault('function_call', None)
        if json_object['function_call'] is not None:
            json_object['function_call'] = OpenAIChatCompletionFunctionCall.from_json(
                json_object['function_call'])
        return cls(**json_object)

    def to_json(self):
        if not bool(self.function_call):
            return {"role": self.role, "content": self.content}
        else:
            return {"role": self.role, "content": self.content,
                    "function_call": self.function_call.to_open_ai_api_format()}

    def get_for_tokenization(self):
        return {
            "role": self.role
        } | ({
            "content": self.content
        } if self.content else {
            "function_call": self.function_call.to_string()
        })


class OpenAIChatRequestFunctionCallParameters(BaseModel):
    type: str
    properties: Dict[str, Dict[str, str]]
    required: List[str]


class OpenAIChatRequestFunctionCall(BaseModel):
    name: str
    description: str
    parameters: OpenAIChatRequestFunctionCallParameters


class OpenAIChatRequest(BaseModel):
    messages: List[OpenAIMessage]
    model: str

    class Config:
        extra = "allow"

    @classmethod
    def from_json(cls, json_object: dict):
        json_object["messages"] = [
            OpenAIMessage.from_json(m) for m in json_object["messages"]]
        return cls(**json_object)

    def to_json(self):
        messages_json = [m.to_json() for m in self.messages]
        dict_representation = {"model": self.model, "messages": messages_json}

        for key, value in self.model_dump().items():
            if key not in dict_representation:
                dict_representation[key] = value

        return dict_representation


class OpenAIChatCompletionChoice(BaseModel):
    index: int
    message: OpenAIMessage
    finish_reason: str

    @classmethod
    def from_json(cls, json_object: dict):
        json_object['message'] = OpenAIMessage.from_json(
            json_object['message'])
        return cls(**json_object)


class OpenAIChatCompletion(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[OpenAIChatCompletionChoice]
    usage: dict

    @classmethod
    def from_json(cls, json_object: dict):
        json_object['choices'] = [OpenAIChatCompletionChoice.from_json(
            choice) for choice in json_object['choices']]
        return cls(**json_object)
