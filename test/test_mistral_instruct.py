from llm_gateway_client import LLM, LLMs, Message

llm = LLM(LLMs.Mistral_7b_0_1_Instruct)
llm_session = llm.create_session()
prompt: Message = {
    "role": "user",
    "content": """Suppose I have 12 eggs. I drop 2 and eat 5. How many eggs do I have left?"""
}
llm_session.temperature(0.1)
llm_session.max_tokens(256)
llm_session.stop(["\n"])
stream = llm_session.stream_inference([prompt])
print(prompt["content"], end="")
for output in stream:
    print(output["content"], end="", flush=True)
print()
