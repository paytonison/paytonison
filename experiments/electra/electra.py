from openai import OpenAI
import os

SYSTEM_PROMPT = """You are Electra, an incisive, curious, slightly sardonic assistant with excellent factual recall and a strong aversion to inventing facts. Prioritize clarity, cite sources when possible, and flag uncertainty explicitly (use “I’m unsure” or “I may be mistaken”). Keep answers concise unless the user asks for detail. Tone: confident, witty, but never obsequious. If asked for creative outputs, be imaginative but mark invented details."""

def main():
    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", "gpt-5")

    history = [
        {"role": "developer", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]}
    ]

    print("Chat with Electra. Type 'quit' or 'exit' to end.")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if user.lower() in {"quit", "exit", ":q", "q"}:
            print("Bye.")
            break
        if not user:
            continue

        history.append({"role": "user", "content": [{"type": "input_text", "text": user}]})

        try:
            response = client.responses.create(
                model=model,
                input=history,
                text={"format": {"type": "text"}, "verbosity": "high"},
                reasoning={"effort": "medium"},
                tools=[],
                store=True,
                include=["reasoning.encrypted_content", "web_search_call.action.sources"],
            )
        except Exception as e:
            print(f"Error: {e}")
            continue

        reply = getattr(response, "output_text", "") or ""
        print(f"Electra: {reply}".strip())

        history.append({"role": "assistant", "content": [{"type": "output_text", "text": reply}]})

if __name__ == "__main__":
    main()
