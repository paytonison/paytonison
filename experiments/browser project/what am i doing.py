from transformers import pipeline
import torch

model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto")

query = "best Jewish delis in Brooklyn."

messages = [
    {"role": "system", "content": 
        '''
        You are an autonomous AI assistant with agentic abilities designed to search the internet.\n
        Your task is to complete the requested task by searching the internet based on the user's input.\n
        You must write outputs like this: {"action": action type, "url or query": url or query}\n
        Being token-efficient: avoid returning excessively long outputs.\n
        '''},
    {"role": "user", "content": query}
]

action = pipe(
    messages,
    max_new_tokens=90,
)

action = action[-1]["generated_text"][2]["content"][2]
print(action)
