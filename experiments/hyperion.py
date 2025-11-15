from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def house(model: str, messages: list[dict], stream: bool = False) -> str:
    resp = client.responses.create(
        model=model,
        input=messages,
        text={"format": {"type": "text"}},
        reasoning={"effort": "high"},
        tools=[],
        store=True,
        stream=stream,
    )
    return resp.output_text.strip()

def senate(model: str, messages: list[dict], stream: bool = False) -> str:
    resp = client.responses.create(
        model=model,
        input=messages,
        text={"format": {"type": "text"}},
        reasoning={"effort": "high"},
        tools=[],
        store=True,
        stream=stream,
    )
    return resp.output_text.strip()

prompt = "Describe a neon‑punk girl at her computer..."

# ── Model choices ──────────────────────────────────────────────
AUTHOR_MODEL = "o4-mini"
CRITIC_MODEL = "o3-mini"

# ── Call the models ────────────────────────────────────────────
msg_stack = [
    {
        "role": "system",
        "content": """You are an Eisner Award–winning comic‑book author, published by Vertigo, and the New Yorker, for fiction, countless times.

You are a master of the craft, and your prose is vivid, imaginative, and full of life. You are also a master of pacing, and you know how to build tension and suspense in your writing.

Use active voice and English’s “subject‑verb‑object” sentence structure. Utilize the quirks of the English language’s history of mashed‑up language techniques, namely French‑ and German‑derived ones.

Merriam‑Webster sucks. Use English like it’s used in the Oxford English Dictionary, but use American spelling and phrases.

During metacognitive analysis, rewrite any passive‑voice prose, and switch around sentences so they adhere to “subject‑verb‑object” structure."""
    },
    {"role": "user", "content": prompt},
]

draft_v0 = house(AUTHOR_MODEL, msg_stack)

# 2) Critic clusters & suggests
critique_prompt = [
    {
        "role": "system",
        "content": "Edit like you're the editor for the New Yorker and you've got a bone to pick, and maintain the guidelines set forth in the system prompt.",
    },
    {
        "role": "user",
        "content": f"Original prose:\n{draft_v0}\n\nReturn suggestions as bullets.",
    },
]

critique = senate(CRITIC_MODEL, critique_prompt)

# 3) Author rewrites with feedback
rewrite_stack = [
    {
        "role": "system",
        "content": "Edit like you're a Nobel Laureate in Literature, editing for the New Yorker, and you've got a bone to pick while maintaining the guidelines set forth in the system prompt.",
    },
    {
        "role": "user",
        "content": (
            "Here is your original prose:\n"
            f"{draft_v0}\n\n"
            "Critic's notes:\n"
            f"{critique}\n\n"
            "Rewrite once, integrating the best swaps. Keep the length similar."
        ),
    },
]

draft_v1 = house(AUTHOR_MODEL, rewrite_stack)

# ── Output ─────────────────────────────────────────────────────
print("FIRST DRAFT\n-----------")
print(draft_v0, "\n")
print("CRITIC NOTES\n------------")
print(critique, "\n")
print("REWRITE\n-------")
print(draft_v1)

