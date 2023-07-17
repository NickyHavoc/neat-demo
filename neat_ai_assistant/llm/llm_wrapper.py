import os
import openai
import tiktoken

from math import sqrt
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence, Union
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

from .open_ai_abstractions import OpenAIChatCompletion, OpenAIChatRequest, OpenAIImageRequest


class LLMWrapper:
    def __init__(
        self,
        aleph_alpha_token: str,
        open_ai_key: str
    ):
        """
        Wrapper class for Aleph Alpha and Open AI APIs.
        """

        self.aleph_alpha_client = Client(aleph_alpha_token)
        openai.api_key = open_ai_key

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
        request: OpenAIChatRequest
    ) -> OpenAIChatCompletion:
        if not isinstance(request, OpenAIChatRequest):
            raise TypeError("request must be of type OpenAIChatRequest.")
        response = openai.ChatCompletion.create(**request.to_json())
        return OpenAIChatCompletion.from_json(response)

    def open_ai_image_complete(
        self,
        request: OpenAIImageRequest,
    ):
        response = openai.Image.create(
        prompt=request.prompt,
        n=request.n,
        size=f"{str(request.size)}x{str(request.size)}"
        )
        return response['data'][0]['url']

    def open_ai_count_tokens(
        self,
        request: OpenAIChatRequest
    ):
        """Returns the number of tokens used by a list of messages."""
        if not isinstance(request, OpenAIChatRequest):
            raise TypeError("request must be of type OpenAIChatRequest.")
        try:
            encoding = tiktoken.encoding_for_model(request.model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = 0
        for message_obj in request.messages:
            message = message_obj.get_for_tokenization()
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += 4
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
