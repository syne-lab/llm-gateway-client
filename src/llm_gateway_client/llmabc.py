from __future__ import annotations
import json
import sys
import time
from typing import Dict, Iterator, List, Optional, Union
from llm_gateway_client.session import LLMSession
from llm_gateway_client.llms import LLMs, Message
import requests
from requests.adapters import HTTPAdapter, Retry

class BearerAuth(requests.auth.AuthBase):
    def __init__(self, token):
        self.token = token
    def __call__(self, r):
        r.headers["authorization"] = "Bearer " + self.token
        return r

class LLMAbstractBaseClass:
    """An LLM instance. Use create_session() to create a session for multiple times of inference."""

    def __init__(
        self,
        model: LLMs,
        endpoint: str = None,
        llm_gateway_token: str = None,
        tokens: Dict[str, str] = None
    ):
        if endpoint is None:
            raise ValueError("endpoint is required")
        self.tokens = tokens
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.auth = BearerAuth(llm_gateway_token) if llm_gateway_token else None
        self.session = requests.Session()
        retries = Retry(total=3,
                connect=3,
                backoff_factor=1,
                status_forcelist=[ 500, 502, 503, 504 ])
        self.session.mount(self.endpoint, HTTPAdapter(max_retries=retries))

    def inference(
        self,
        prompt: List[Message],
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
    ) -> Message:
        retry_time = 0
        while True:
            res = self.session.post(
                f"{self.endpoint}/inference",
                json={
                    "model": self.model.value,
                    "prompt": prompt,
                    "suffix": suffix,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "logprobs": logprobs,
                    "echo": echo,
                    "stop": stop,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                    "repeat_penalty": repeat_penalty,
                    "top_k": top_k,
                    "tokens": self.tokens,
                },
                auth=self.auth
            )
            if res.ok:
                return res.json()
            else:
                print(res.status_code, ':', res.text, file=sys.stderr)
                if res.status_code < 500:
                    break
            retry_time += 1
            if retry_time > 10:
                break
            time.sleep(3)

    def stream_inference(
        self,
        prompt: List[Message],
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
    ) -> Iterator[Message]:
        retry_time = 0
        while True:
            res = self.session.post(
                f"{self.endpoint}/stream_inference",
                stream=True,
                json={
                    "model": self.model.value,
                    "prompt": prompt,
                    "suffix": suffix,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "logprobs": logprobs,
                    "echo": echo,
                    "stop": stop,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                    "repeat_penalty": repeat_penalty,
                    "top_k": top_k,
                    "tokens": {
                        "OPENAI_API_KEY": self.openai_api_key,
                        "GOOGLE_API_KEY": self.google_api_key,
                        "ANTHROPIC_API_KEY": self.anthropic_api_key,
                        "HUGGING_FACE_HUB_TOKEN": self.hugging_face_hub_token,
                    },
                },
                auth=self.auth
            )
            if res.ok:
                for part in res.iter_lines():
                    yield json.loads(part)
                break
            else:
                print(res.status_code, ':', res.text, file=sys.stderr)
                if res.status_code < 500:
                    break
            retry_time += 1
            if retry_time > 10:
                break
            time.sleep(3)

    def create_session(self) -> LLMSession:
        """Create a session for multiple times of inference."""
        return LLMSession(self)
