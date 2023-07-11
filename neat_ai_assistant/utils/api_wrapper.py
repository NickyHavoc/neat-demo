import json
import os
import openai

from math import sqrt
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Union
from pydantic import BaseModel
from aleph_alpha_client import (
    Client,
    CompletionRequest,
    DetokenizationRequest,
    EvaluationRequest,
    EmbeddingRequest,
    QaRequest,
    SemanticEmbeddingRequest,
    SummarizationRequest,
    TokenizationRequest,
)
from tqdm import tqdm


class OpenAIFunctionCall(BaseModel):
    name: str
    arguments: Dict[str, Any]

    @classmethod
    def from_string(cls, string: str):
        return cls(**json.loads(string))

    @classmethod
    def from_json(cls, json_object: dict):
        return cls(**json_object)

    def get_args_except(self, except_args: List[str]):
        return {k: v for k, v in self.arguments.items() if k not in except_args}


class OpenAIMessage(BaseModel):
    role: str
    content: Optional[str]
    function_call: Optional[OpenAIFunctionCall]

    @classmethod
    def from_json(cls, json_object: dict):
        json_object.setdefault('function_call', None)
        if json_object['function_call'] is not None:
            json_object['function_call'] = OpenAIFunctionCall.from_json(json_object['function_call'])
        return cls(**json_object)


class OpenAIChatCompletionChoice(BaseModel):
    index: int
    message: OpenAIMessage
    finish_reason: str

    @classmethod
    def from_json(cls, json_object: dict):
        json_object['message'] = OpenAIMessage.from_json(json_object['message'])
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
        json_object['choices'] = [OpenAIChatCompletionChoice.from_json(choice) for choice in json_object['choices']]
        return cls(**json_object)


class LLMWrapper:

    ALEPH_ALPHA_TOKEN = os.getenv("ALEPH_ALPHA_TOKEN")
    OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")

    def __init__(self):
        """
        Wrapper class for Aleph Alpha and Open AI APIs.
        Will read tokens for each on instatiation from environment, keys:
        - ALEPH_ALPHA_TOKEN
        - OPEN_AI_KEY
        """

        self.aleph_alpha_client = Client(self.ALEPH_ALPHA_TOKEN)
        openai.api_key = self.OPEN_AI_KEY

    @staticmethod
    def build_aleph_alpha_request(
        request_object: Union[
            CompletionRequest,
            DetokenizationRequest,
            EmbeddingRequest,
            EvaluationRequest,
            QaRequest,
            SemanticEmbeddingRequest,
            SummarizationRequest,
            TokenizationRequest,
        ],
        model: Optional[str] = None,
        checkpoint: Optional[str] = None,
        adapter: Optional[str] = None,
        beta: Optional[bool] = None,
    ):

        if bool(model) == bool(checkpoint):
            raise ValueError("Must provide exactly one of model, checkpoint.")

        return (
            {"request": request_object}
            | ({"model": model} if model else {"checkpoint": checkpoint})
            | ({"beta": beta} if beta else {})
            | ({"adapter": adapter} if adapter else {})
        )

    @staticmethod
    def _get_alpha_alpha_api_tasks(requests: List[dict]):
        """
        Finds the correct task for each request_dict.
        Useful if a large number of requests is send async and they require different API functions (e.g. complete, evaluate).
        """

        tasks = []

        for request in requests:
            request_type = type(request["request"])

            if request_type == CompletionRequest:
                tasks.append(Client.complete)

            elif request_type == DetokenizationRequest:
                tasks.append(Client.detokenize)

            elif request_type == EmbeddingRequest:
                tasks.append(Client.embed)

            elif request_type == EvaluationRequest:
                tasks.append(Client.evaluate)

            elif request_type == QaRequest:
                tasks.append(Client.qa)

            elif request_type == SemanticEmbeddingRequest:
                tasks.append(Client.semantic_embed)

            elif request_type == SummarizationRequest:
                tasks.append(Client.summarize)

            elif request_type == TokenizationRequest:
                tasks.append(Client.tokenize)

            else:
                raise ValueError(
                    f"Request type not allowed, got {request_type}.")

        return tasks

    def aleph_alpha_request(self, request: dict):
        task = self._get_alpha_alpha_api_tasks(requests=[request])[0]
        return task(self.aleph_alpha_client, **request)

    def aleph_alpha_batch_request(
        self,
        requests: List[dict],
        max_workers: int = 10,
        progress_bar: bool = False,
    ):
        tasks = self._get_alpha_alpha_api_tasks(requests=requests)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            requests_and_tasks = zip(requests, tasks)

            def call_task(params):
                request, task = params
                return task(self.aleph_alpha_client, **request)

            if progress_bar:
                responses = list(
                    tqdm(
                        executor.map(call_task, requests_and_tasks),
                        total=len(requests),
                    )
                )
            else:
                responses = list(executor.map(call_task, requests_and_tasks))

        return responses

    @staticmethod
    def compute_cosine_similarity(
        embedding_1: Sequence[float], embedding_2: Sequence[float]
    ):
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(embedding_1)):
            x = embedding_1[i]
            y = embedding_2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
        score = sumxy / sqrt(sumxx * sumyy)
        return (score + 1) / 2

    def open_ai_chat_complete(
        self,
        params: dict
    ) -> OpenAIChatCompletion:
        assert "model" in params and "messages" in params, "Must provide at least model and messages in params."
        response = openai.ChatCompletion.create(**params)
        return OpenAIChatCompletion.from_json(response)
