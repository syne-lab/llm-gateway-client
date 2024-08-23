from __future__ import annotations
from typing import Iterator, List, Optional, Union, TYPE_CHECKING
from llm_gateway_client.llms import Message

if TYPE_CHECKING:
    from llm_gateway_client.llmabc import LLMAbstractBaseClass


def assertMessageList(messages: List[Message]):
    """@private"""
    assert isinstance(messages, list), "messages should be a list of Message"
    for message in messages:
        assert message.get("role") in (
            None,
            "user",
            "system",
            "assistant",
        ), "messages should be a list of Message"
        assert (
            message.get("content") is not None
        ), "messages should be a list of Message"


class LLMSession:
    def __init__(self, llm: LLMAbstractBaseClass):
        self._context: List[Message] = []
        self._logs: List[List[Message]] = [[]]
        self._current_context_id = 0
        self._llm = llm
        self.default()

    @property
    def log(self) -> List[List[Message]]:
        return self._logs

    @property
    def context(self) -> List[Message]:
        return self._context

    @context.setter
    def context(self, context: List[Message]):
        self._logs.append([message for message in context])
        self._current_context_id += 1
        self._context = context

    def clear_context(self):
        self._logs.append([])
        self._current_context_id += 1
        self._context = []

    def default(
        self,
        suffix: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.95,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        top_k: int = 40,
    ) -> LLMSession:
        self._suffix = suffix
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._logprobs = logprobs
        self._echo = echo
        self._stop = stop
        self._frequency_penalty = frequency_penalty
        self._presence_penalty = presence_penalty
        self._repeat_penalty = repeat_penalty
        self._top_k = top_k
        return self

    def suffix(self, suffix: Optional[str]) -> LLMSession:
        self._suffix = suffix
        return self

    def max_tokens(self, max_tokens: int) -> LLMSession:
        self._max_tokens = max_tokens
        return self

    def temperature(self, temperature: float) -> LLMSession:
        self._temperature = temperature
        return self

    def top_p(self, top_p: float) -> LLMSession:
        self._top_p = top_p
        return self

    def logprobs(self, logprobs: Optional[int]) -> LLMSession:
        self._logprobs = logprobs
        return self

    def echo(self, echo: bool) -> LLMSession:
        self._echo = echo
        return self

    def stop(self, stop: Optional[Union[str, List[str]]]) -> LLMSession:
        self._stop = stop
        return self

    def frequency_penalty(self, frequency_penalty: float) -> LLMSession:
        self._frequency_penalty = frequency_penalty
        return self

    def presence_penalty(self, presence_penalty: float) -> LLMSession:
        self._presence_penalty = presence_penalty
        return self

    def repeat_penalty(self, repeat_penalty: float) -> LLMSession:
        self._repeat_penalty = repeat_penalty
        return self

    def top_k(self, top_k: int) -> LLMSession:
        self._top_k = top_k
        return self

    def inference(
        self,
        messages: List[Message],
    ) -> Message:
        assertMessageList(messages)
        self._context += messages
        self._logs[self._current_context_id] += messages
        output = self._llm.inference(
            self._context,
            suffix=self._suffix,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
            logprobs=self._logprobs,
            echo=self._echo,
            stop=self._stop,
            frequency_penalty=self._frequency_penalty,
            presence_penalty=self._presence_penalty,
            repeat_penalty=self._repeat_penalty,
            top_k=self._top_k,
        )
        self._context.append(output)
        self._logs[self._current_context_id].append(output)
        return output

    def stream_inference(
        self,
        messages: List[Message],
    ) -> Iterator[Message]:
        assertMessageList(messages)
        self._context += messages
        self._logs[self._current_context_id] += messages
        complete_output: Message = {"role": "assistant", "content": ""}
        for output in self._llm.stream_inference(
            self._context,
            suffix=self._suffix,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
            logprobs=self._logprobs,
            echo=self._echo,
            stop=self._stop,
            frequency_penalty=self._frequency_penalty,
            presence_penalty=self._presence_penalty,
            repeat_penalty=self._repeat_penalty,
            top_k=self._top_k,
        ):
            if output["content"] is None:
                continue
            complete_output["content"] += output["content"]
            yield output
        self._context.append(complete_output)
        self._logs[self._current_context_id].append(complete_output)
