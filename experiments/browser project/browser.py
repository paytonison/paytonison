#google killer

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
from transformers import pipeline
import json
import time
import torch
import requests

pipe = pipeline(
"text-generation",
"meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.bfloat16,
    max_new_tokens=256
)

def action(pipe):
    messages = [
    {"role": "system", "content": '''
            You are an autonomous AI web searcher. Your goal is to process the user query locate the best website for the user. You must present result in proper JSON format using this template: {"query": "query_subject", "website": "website_url"}.
            AVOID USING GOOGLE AT ALL COST!!!
            Avoid verbosity and irrelevant information.
            Provide only one website link.
            '''},
    {"role": "user", "content": "space"}
    ]
    action = pipe(messages)
    return action[0]["generated_text"][-1]["content"]


bot_action = action(pipe)
print(bot_action)