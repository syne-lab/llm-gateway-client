from __future__ import annotations
from enum import Enum
import json
import os
from typing import List, Literal, Optional, Set, TypedDict, Union
from typing_extensions import Annotated
from pydantic import BaseModel, Field


class LLMsQuery:
    def __init__(self, models: List[ModelInfo]):
        self._filtered_models = models

    def limit(self, limit: int) -> LLMsQuery:
        self._filtered_models = self._filtered_models[:limit]
        return self

    def exec(self) -> List[ModelInfo]:
        return [item for item in self._filtered_models]

    def where_engine_is(
        self, engine: Literal["llama.cpp", "transformers", "openai", "anthropic"]
    ) -> LLMsQuery:
        self._filtered_models = filter(
            lambda x: x.engine == engine, self._filtered_models
        )
        return self

    def where_quant_alternative_is(self, quant_alternative: bool) -> LLMsQuery:
        self._filtered_models = filter(
            lambda x: "quant_alternative" in x.model_fields, self._filtered_models
        )
        self._filtered_models = filter(
            lambda x: x.quant_alternative is not None
            if quant_alternative
            else x.quant_alternative is None,
            self._filtered_models,
        )
        return self

    def where_quantized_is(self, quantized: bool):
        self._filtered_models = filter(
            lambda x: "quantized" in x.model_fields, self._filtered_models
        )
        self._filtered_models = filter(
            lambda x: x.quantized == quantized, self._filtered_models
        )
        return self

    def where_name_is(self, name: str) -> LLMsQuery:
        self._filtered_models = filter(
            lambda x: x.name.value == name, self._filtered_models
        )
        return self

    def where_model_is(self, model: LLMs) -> LLMsQuery:
        self._filtered_models = filter(lambda x: x.name == model, self._filtered_models)
        return self

    def where_expert_in(self, expert: Set[Literal["code", "general"]]) -> LLMsQuery:
        self._filtered_models = filter(
            lambda x: x.expert in expert, self._filtered_models
        )
        return self

    def where_chat_is(self, chat: bool) -> LLMsQuery:
        self._filtered_models = filter(lambda x: x.chat == chat, self._filtered_models)
        return self

    def where_infill_is(self, infill: bool) -> LLMsQuery:
        self._filtered_models = filter(
            lambda x: x.infill == infill, self._filtered_models
        )
        return self

    def where_price_less_than(self, price: float) -> LLMsQuery:
        self._filtered_models = filter(
            lambda x: x.price <= price, self._filtered_models
        )
        return self

    def where_price_greater_than(self, price: float) -> LLMsQuery:
        self._filtered_models = filter(lambda x: x.price > price, self._filtered_models)
        return self

    def where_max_context_less_than(self, max_context: int) -> LLMsQuery:
        self._filtered_models = filter(
            lambda x: x.max_context <= max_context, self._filtered_models
        )
        return self

    def where_max_context_greater_than(self, max_context: int) -> LLMsQuery:
        self._filtered_models = filter(
            lambda x: x.max_context > max_context, self._filtered_models
        )
        return self

    def order_by_score(self, reverse: bool = False) -> LLMsQuery:
        self._filtered_models = sorted(
            self._filtered_models, key=lambda x: x.score, reverse=not reverse
        )
        return self



class LLMs(Enum):
    Llama_2_7b = "meta-llama/Llama-2-7b-hf"  # Need HF token
    Llama_2_13b = "meta-llama/Llama-2-13b-hf"  # Need HF token
    Llama_2_70b_Quant = "TheBloke/Llama-2-70B-GGUF"
    Llama_2_7b_Chat = "meta-llama/Llama-2-7b-chat-hf"  # Need HF token
    Llama_2_13b_Chat = "meta-llama/Llama-2-13b-chat-hf"  # Need HF token
    Llama_2_70b_Chat_Quant = "TheBloke/Llama-2-70B-Chat-GGUF"
    CodeLlama_7b = "codellama/CodeLlama-7b-hf"
    CodeLlama_13b = "codellama/CodeLlama-13b-hf"
    CodeLlama_34b = "codellama/CodeLlama-34b-hf"
    CodeLlama_34b_Quant = "TheBloke/CodeLlama-34B-GGUF"
    CodeLlama_7b_Instruct = "codellama/CodeLlama-7b-Instruct-hf"
    CodeLlama_13b_Instruct = "codellama/CodeLlama-13b-Instruct-hf"
    CodeLlama_34b_Instruct = "codellama/CodeLlama-34b-Instruct-hf"
    CodeLlama_34b_Instruct_Quant = "TheBloke/CodeLlama-34B-Instruct-GGUF"
    Llama_3_8b_Instruct = "meta-llama/Meta-Llama-3-8B-Instruct"  # Need HF token
    Llama_3_70b_Instruct_Quant = "QuantFactory/Meta-Llama-3-70B-Instruct-GGUF"
    StarCoder = "bigcode/starcoder"  # Need HF token
    Mistral_7b_0_1 = "TheBloke/Mistral-7B-v0.1-GGUF"
    Mistral_7b_0_1_Instruct = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
    Mixtral_8x7b_0_1 = "TheBloke/Mixtral-8x7B-v0.1-GGUF"
    Mixtral_8x7b_0_1_Instruct = "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF"
    Yi_6b = "01-ai/Yi-6B"
    Yi_34b = "01-ai/Yi-34B"
    Yi_6b_200k = "01-ai/Yi-6B-200K"
    Yi_34b_200k = "01-ai/Yi-34B-200K"
    Yi_6b_Chat = "01-ai/Yi-6B-Chat"
    Yi_34b_Chat = "01-ai/Yi-34B-Chat"
    Phi_3_Mini_128k_Instruct = "microsoft/Phi-3-mini-128k-instruct"
    CodeGen25_7b_Multi = "Salesforce/codegen25-7b-multi"
    GPT4_O = "gpt-4o"  # Need OpenAI token
    GPT4_O_2024_0513 = "gpt-4o-2024-05-13"  # Need OpenAI token
    GPT4_Turbo = "gpt-4-turbo"  # Need OpenAI token
    GPT4_Turbo_2024_0409 = "gpt-4-turbo-2024-04-09"  # Need OpenAI token
    GPT4_0125_Preview = "gpt-4-0125-preview" # Need OpenAI token
    GPT4_1106_Preview = "gpt-4-1106-preview" # Need OpenAI token
    GPT4_Turbo_Preview = "gpt-4-turbo-preview" # Need OpenAI token
    GPT4 = "gpt-4"  # Need OpenAI token
    GPT4_32k = "gpt-4-32k"  # Need OpenAI token
    GPT3_5_Turbo = "gpt-3.5-turbo"  # Need OpenAI token
    Claude_3_opus_20240229 = "claude-3-opus-20240229" # Need Anthropic token
    Claude_3_sonnet_20240229 = "claude-3-sonnet-20240229" # Need Anthropic token
    Claude_3_haiku_20240307 = "claude-3-haiku-20240307" # Need Anthropic token
    Claude_2_1 = "claude-2.1"  # Need Anthropic token
    Claude_2_0 = "claude-2.0"  # Need Anthropic token
    Claude_Instant_1_2 = "claude-instant-1.2"  # Need Anthropic token

    @classmethod
    def new_query(self) -> LLMsQuery:
        # https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard
        # The score contains many bias, just for reference only.
        basedir = os.path.dirname(__file__)
        with open(os.path.join(basedir, "llm_models.json"), "r") as f:
            self.all_models = json.load(f)

        def replace_llms(x):
            x["name"] = LLMs(x["name"])
            return x

        self.all_models: List[ModelInfo] = [
            i for i in map(replace_llms, self.all_models)
        ]
        for i in range(len(self.all_models)):
            model = self.all_models[i]
            if model["engine"] == "llama.cpp":
                self.all_models[i] = LlamaCppModelInfo(**model)
            elif model["engine"] == "transformers":
                self.all_models[i] = TransformersModelInfo(**model)
            elif model["engine"] == "openai":
                self.all_models[i] = OpenAIModelInfo(**model)
            elif model["engine"] == "anthropic":
                self.all_models[i] = AnthropicModelInfo(**model)
            else:
                raise ValueError(f"Unknown engine {model['engine']}")
        return LLMsQuery(self.all_models.copy())

    def getInfo(self) -> ModelInfo:
        query_result = LLMs.new_query().where_model_is(self).exec()
        assert len(query_result) == 1, "Model not found"
        return query_result[0]

class LlamaCppModelInfo(BaseModel):
    name: LLMs
    engine: Literal["llama.cpp"]
    template: str
    gguf_script: Literal["default", "hf"]
    revision: Optional[str]
    score: float
    expert: Literal["code", "general"]
    chat: bool
    infill: bool
    price: float
    max_context: int
    file_list: List[str]
    quantized: bool
    quant_alternative: Optional[str]


class TransformersModelInfo(BaseModel):
    name: LLMs
    engine: Literal["transformers"]
    template: str
    revision: Optional[str]
    score: float
    expert: Literal["code", "general"]
    chat: bool
    infill: bool
    price: float
    max_context: int
    file_list: List[str]
    quantized: bool
    quant_alternative: Optional[str]


class OpenAIModelInfo(BaseModel):
    name: LLMs
    engine: Literal["openai"]
    score: float
    expert: Literal["code", "general"]
    chat: bool
    infill: bool
    price: float
    max_context: int

class AnthropicModelInfo(BaseModel):
    name: LLMs
    engine: Literal["anthropic"]
    score: float
    expert: Literal["code", "general"]
    chat: bool
    infill: bool
    price: float
    max_context: int


ModelInfo = Annotated[
    Union[LlamaCppModelInfo, TransformersModelInfo, OpenAIModelInfo, AnthropicModelInfo],
    Field(..., discriminator="engine"),
]

class Message(TypedDict):
    role: Optional[Literal["system", "user", "assistant"]]
    content: str
