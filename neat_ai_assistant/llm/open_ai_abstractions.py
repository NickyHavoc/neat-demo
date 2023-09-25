import json
from typing import Any, Dict, Sequence, Optional, Union

from pydantic import BaseModel


class ChatCompletionFunctionCall(BaseModel):
    name: str
    arguments: Dict[str, Any]

    @classmethod
    def from_json(cls, json_object: Dict[str, Union[str, Dict[str, Any]]]):
        # OpenAI sometimes generates the str "functions.". No idea why.
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

    def get_args_except(self, except_args: Sequence[str]):
        return {k: v for k, v in self.arguments.items() if k not in except_args}


class Message(BaseModel):
    role: str
    content: Optional[str]
    function_call: Optional[ChatCompletionFunctionCall] = None

    @classmethod
    def from_json(cls, json_object: dict):
        json_object.setdefault('function_call', None)
        if json_object['function_call'] is not None:
            json_object['function_call'] = ChatCompletionFunctionCall.from_json(
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


class ChatRequestFunctionCall(BaseModel):
    name: str
    description: str
    parameters: dict


class ChatRequest(BaseModel):
    messages: Sequence[Message]
    model: str

    class Config:
        extra = "allow"

    @classmethod
    def from_json(cls, json_object: dict):
        json_object["messages"] = [
            Message.from_json(m) for m in json_object["messages"]]
        return cls(**json_object)

    def to_json(self):
        messages_json = [m.to_json() for m in self.messages]
        dict_representation = {"model": self.model, "messages": messages_json}

        for key, value in self.model_dump().items():
            if key not in dict_representation:
                dict_representation[key] = value

        return dict_representation


class ChatCompletion(BaseModel):
    index: int
    message: Message
    finish_reason: str

    @classmethod
    def from_json(cls, json_object: dict):
        json_object['message'] = Message.from_json(
            json_object['message'])
        return cls(**json_object)


class ChatResponse(BaseModel):
    metadata: dict
    completions: Sequence[ChatCompletion]

    @classmethod
    def from_json(cls, json_object: dict):
        completions = [ChatCompletion.from_json(
            choice) for choice in json_object.pop('choices')]
        json_object["usage"] = dict(json_object["usage"])
        return cls(metadata=dict(json_object), completions=completions)
