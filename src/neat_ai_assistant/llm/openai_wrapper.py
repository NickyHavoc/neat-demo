from enum import Enum
from functools import wraps
import time
from typing import Any, Callable, Generator, Literal, Mapping, Sequence, TypeVar, cast

import openai
from openai.types.chat import ChatCompletion
from pydantic import BaseModel
import tiktoken


class Model(Enum):
    GPT_3_5 = "gpt-3.5-turbo-1106"
    GPT_4 = "gpt-4-1106-preview"


class Message(BaseModel):
    role: Literal["assistant", "user", "system"]
    content: str


T = TypeVar("T")


class OpenaiWrapper:
    @staticmethod
    def retry_with_backoff(
        max_retries: int = 5,
        base_delay: float = 0.25,
        factor: float = 4,
        max_delay: float = 30,
    ) -> Callable[[Callable[..., T]], Callable[..., T],]:
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def wrapper(*args: tuple[T, ...], **kwargs: Mapping[str, T]) -> T:
                def exponential_backoff() -> Generator[float, None, None]:
                    delay = base_delay
                    while True:
                        yield delay
                        delay = min(delay * factor, max_delay)

                retry_delays = exponential_backoff()

                for _ in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except TimeoutError as e:
                        time.sleep(next(retry_delays))
                raise TimeoutError(
                    f"Function {func.__name__} failed after {max_retries} retries"
                )

            return wrapper

        return decorator

    @retry_with_backoff()
    def chat_complete_with_tools(
        self,
        messages: Sequence[Message],
        model: Model,
        tools: Sequence[Mapping[str, Any]],
        temperature: float = 0.0,
    ) -> ChatCompletion:
        result = openai.chat.completions.create(  # type: ignore
            messages=[m.model_dump() for m in messages],
            model=model.value,
            temperature=temperature,
            tools=tools,  # this for some reason causes mypy issues
        )
        return cast(ChatCompletion, result)

    @retry_with_backoff()
    def chat_complete(
        self, messages: Sequence[Message], model: Model, temperature: float = 0
    ) -> ChatCompletion:
        return openai.chat.completions.create(
            messages=[m.model_dump() for m in messages],  # type: ignore
            model=model.value,
            temperature=temperature,
        )

    def open_ai_count_tokens(self, messages: Sequence[Message], model: Model) -> int:
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(model.value)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = 0
        for message_obj in messages:
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += 4
            for key, value in message_obj.model_dump().items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
