#!/usr/bin/env python3
# socratic_duo_responses.py
#
# Requires:  pip install --upgrade openai python-dotenv
# Make sure your OPENAI_API_KEY is in the environment or a .env file.

import os, time
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI               # ← new import

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_NAME = "o3"                       # adjust if your deployment alias differs
TRANSCRIPT = "dialogue.txt"
TURNS      = 12                         # total exchanges (Z → A → Z …)

# -- persona instructions ----------------------------------------------------

ZARATHUSTRA_SYS = """You are Zarathustra. Engage in a Socratic dialogue with Asari."""

ASARI_SYS = """You are Asari. Engage in a Socratic dialogue with Zarathustra."""

# -- helper utilities --------------------------------------------------------

def stamp(text: str, who: str) -> None:
    """Append a timestamped utterance to the transcript file."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(TRANSCRIPT, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {who}: {text}\n\n")

def talk(history: list[dict], who: str, instructions: str) -> str:
    """Hit the Responses API and return the assistant’s line."""
    rsp = client.responses.create(
        model=MODEL_NAME,
        instructions=instructions,      # works like a system prompt
        input=history,                  # running conversation so far
        reasoning={"effort": "high"},
        stream=False,                   
        max_output_tokens=900,
    )
    text = rsp.output_text.strip()      # Responses API exposes .output_text  [oai_citation:0‡Learn R, Python & Data Science Online](https://www.datacamp.com/tutorial/openai-responses-api) [oai_citation:1‡Learn R, Python & Data Science Online](https://www.datacamp.com/tutorial/openai-responses-api)
    stamp(text, who)
    return text

# -- bootstrap both roles ----------------------------------------------------

z_history = []                          # conversation seen by Zarathustra
a_history = []                          # conversation seen by Asari

seed_q = "What is the root of human flourishing?"  # opening question

z_history.append({"role": "user", "content": seed_q})
z_reply = talk(z_history, "Zarathustra", ZARATHUSTRA_SYS)

a_history.append({"role": "user", "content": z_reply})

# -- main dialogue loop ------------------------------------------------------

for _ in range(TURNS - 1):
    # Asari responds
    a_reply = talk(a_history, "Asari", ASARI_SYS)
    z_history.append({"role": "user", "content": a_reply})

    # Zarathustra counters
    z_reply = talk(z_history, "Zarathustra", ZARATHUSTRA_SYS)
    a_history.append({"role": "user", "content": z_reply})

    time.sleep(1)                       # polite throttle

print(f"Dialogue complete ➜ {TRANSCRIPT}")
