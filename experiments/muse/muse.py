#!/usr/bin/env python3
"""
muse_chat.py – talk to your Muse fine-tune from the CLI
"""

import os, sys, json
from openai import OpenAI

# -------------------------------------------------------------------
# *** FILL THESE IN **************************************************
MODEL_ID       = "ft:gpt-4.1-nano-2025-04-14:the-singularity::BjD1PUXJ"   # model
PROMPT_ID      = "pmpt_6850b48705a48195b5fcb5de6f4df8e70558fee061453ddd"  # prompt ID
PROMPT_VERSION = "4"                                                      # prompt version
# -------------------------------------------------------------------

client = OpenAI()   # assumes OPENAI_API_KEY is set in the environment

conversation = [
    {
        "role": "user",
        "content": [{"type": "input_text", "text": "Hello. What is your name?"}]
    }
]


def call_muse(conv):
    """Send conversation, return assistant message dict."""
    resp = client.responses.create(
        model   = MODEL_ID,
        prompt  = {"id": PROMPT_ID, "version": PROMPT_VERSION},
        input   = conv,
        max_output_tokens = 2048,
        store   = True
    )
    # For the responses endpoint the assistant message is always the *last*
    # element of the returned `output` list.
    return resp.output[-1]


def main():
    print("Muse is ready – type messages, 'exit' to quit\n")
    try:
        while True:
            user = input("You  > ").strip()
            if user.lower() in {"exit", "quit", "q"}:
                break

            conversation.append({
                "role": "user",
                "content": [{"type": "input_text", "text": user}]
            })

            assistant_msg = call_muse(conversation)
            conversation.append(assistant_msg.model_dump(exclude_none=True))

            print("Muse >", assistant_msg.content[0].text, "\n")

    except (KeyboardInterrupt, EOFError):
        print("\n[Session ended]")

    # Optional local log
    # with open("muse_log.json", "w") as fp:
    #     json.dump(conversation, fp, indent=2)

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY first.")
    main()
