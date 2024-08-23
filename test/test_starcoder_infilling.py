from llm_gateway_client import LLM, LLMs

llm = LLM(LLMs.StarCoder)
llm_session = llm.create_session()
prefix = """float Q_rsqrt( float number )
{
	long i;
	float x2, y;
	const float threehalfs = 1.5F;

	x2 = number * 0.5F;
	y  = number;
"""

suffix= """
	return y;
}
"""
llm_session.temperature(0.5)
llm_session.max_tokens(256)
stream = llm_session.stream_inference([
    {
        "role": None,
        "content": prefix + "<FILL>" + suffix
	}
])
print(prefix, end="")
for output in stream:
    print(output["content"], end="", flush=True)
print(suffix)
