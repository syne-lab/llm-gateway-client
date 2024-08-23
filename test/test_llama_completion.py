from llm_gateway_client import LLM, LLMs

llm = LLM(LLMs.Llama_2_7b)
llm_session = llm.create_session()
prompt = {
    "role": None,
    "content": """def fib(n):\n"""
}
llm_session.temperature(1.1)
llm_session.max_tokens(256)
llm_session.stop("<EOT>")
stream = llm_session.stream_inference([prompt])
print(prompt["content"], end="")
for output in stream:
    print(output["content"], end="", flush=True)
print()
