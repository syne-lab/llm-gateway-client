from __future__ import annotations
from typing import Dict
from llm_gateway_client.llmabc import LLMAbstractBaseClass
from llm_gateway_client.llms import LLMs
import os


def token_from_env(env_var: str, tokens: Dict[str, str]):
    if tokens.get(env_var) is None:
        if os.getenv(env_var):
            tokens[env_var] = os.getenv(env_var)

def LLM(
    model: LLMs,
    endpoint: str = None,
    llm_gateway_token: str = None,
    tokens: Dict[str, str] = None,
) -> LLMAbstractBaseClass:
    """
    Create an LLM instance from the model name. Download and initialize the model if applicable.

    Argument:
        model: The model name.
        endpoint: The endpoint of the LLM server. Default to "http://172.17.0.1:31211".

    """
    if endpoint is None:
        endpoint = os.getenv("LLM_GATEWAY_ENDPOINT", "http://172.17.0.1:31211")
    if tokens is None:
        tokens = dict()
    for env_var in [
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "ANTHROPIC_API_KEY",
        "HUGGING_FACE_HUB_TOKEN",
    ]:
        token_from_env(env_var, tokens)
    if llm_gateway_token is None:
        llm_gateway_token = (
            os.getenv("LLM_GATEWAY_TOKEN") if os.getenv("LLM_GATEWAY_TOKEN") else None
        )
    return LLMAbstractBaseClass(
        model,
        endpoint=endpoint,
        llm_gateway_token=llm_gateway_token,
        tokens=tokens,
    )

