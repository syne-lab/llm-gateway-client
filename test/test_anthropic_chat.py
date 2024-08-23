from llm_gateway_client import LLM, LLMs

llm = LLM(LLMs.Claude_Instant_1_2)
llm_session = llm.create_session()
llm_session2 = llm.create_session()
sys_prompt = """
You are a security bot writing c programs, aiming to find potential bugs inside the API.
Follow the user's requirements carefully & to the letter.
Output only the code in a single code block.
Use Markdown formatting in your answers.
Make sure to include the programming language name at the start of the Markdown code blocks.
"""

user_prompt= """
You can include any complex operations related to socket programming as long as they are posix systemcalls.
Make sure you write the program in the main function.
Write a complex c program to fuzz following function:
```c
#include <sys/socket.h>

int socket(int domain, int type, int protocol);
```
"""
llm_session.temperature(1.1).max_tokens(512)
stream = llm_session.stream_inference([
    {
        "role": "system",
        "content": sys_prompt
    },
    {
        "role": "user",
        "content": user_prompt
    },
])
for output in stream:
    assert output["role"] == "assistant"
    print(output["content"], end="", flush=True)
print()
llm_session2.temperature(1.1).max_tokens(512)
output = llm_session2.inference([
    {
        "role": "system",
        "content": sys_prompt
    },
    {
        "role": "user",
        "content": user_prompt
    },
])
print(output["content"])
