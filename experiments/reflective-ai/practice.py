# this code will create a self-perpetuating text generation loop
# the goal is for the AI to think and generate text based on its own thoughts, self, and environment

import os
import torch
from transformers import pipeline

f1 = open("journal.txt", "w") # open file in write mode
f2 = open("journal.txt", "r") # open file in read mode

def generate(f1, f2): # function to generate text
    model_id = "meta-llama/Llama-3.2-1B-Instruct" # instruct model provides best results for this task
    
    generator = pipeline( # pipeline
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    messages = [ # messages
            {"role": "system", "content": f"You are an autonomous artificial intelligence agent tasked with writing every thought in your head. {f2.read()}"}, # meta system prompt that includes the data of the file
            {"role": "user", "content": " "}, # user prompt to get things started
        ]
    
    outputs = generator(messages, max_new_tokens=256) # generating text using hf pipeline
    return outputs # return outputs

while True:
    huh = generate(f1, f2) # call generate function with file variables
    
    f1.write(huh[0]["generated_text"][2]["content"]) # write to the file
    
    print(huh[0]["generated_text"][2]["content"]) # print the generated text

