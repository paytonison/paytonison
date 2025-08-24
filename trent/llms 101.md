
## **LLMs 101: A Practical Introduction**
---
> **Who this is for.** Developers who want a fast, working understanding of large language models and the knobs that matter in real apps.
---
### **At a glance**

```
Text prompt
   ↓  (tokenization)
Tokens → Embeddings → [Transformer layers × N] → Next‑token probabilities
   ↓                       ↓
Detokenization         Sampling (temperature/top_p) → Output text
```

- **LLMs** are neural networks (usually **transformers**) trained on lots of text to predict the next token.
- **Tokenization** splits text into subword units; **embeddings** map tokens to vectors; transformer layers build context‑aware representations.    
- Generation = repeated next‑token sampling until a stop condition (length, stop tokens) is met. 
---
### **Quick start: generate text**

**Python**
```
from openai import OpenAI

client = OpenAI()
resp = client.responses.create(
    model="gpt-4o",
    instructions="You are a concise technical explainer.",
    input="In one paragraph, explain what a token is in an LLM."
)
print(resp.output_text)
```
> The OpenAI Python SDK’s primary interface is the **Responses API**; response.output_text returns the aggregated text. Source: official SDK README. 

  **JavaScript / TypeScript**
```
import OpenAI from "openai";
const client = new OpenAI();

const resp = await client.responses.create({
  model: "gpt-4o",
  instructions: "You are a concise technical explainer.",
  input: "In one paragraph, explain what a token is in an LLM."
});
console.log(resp.output_text);
```
> The OpenAI Node SDK also centers on the **Responses API** with response.output_text. Source: official SDK README. 


> **Note.** Model names evolve; check the Models page for currently available options when you build. 
---
### **What can LLMs do?**

Despite the name, LLMs can be **multi‑modal** when models and inputs support it (text, code, sometimes images/audio). Core text tasks:
- **Generate**: draft, rewrite, continue, or brainstorm.    
- **Transform**: translate, rephrase, format, classify, extract.
- **Analyze**: summarize, compare, tag, or answer questions.
- **Tool use / agents**: call functions or APIs to act in a task loop. 

These patterns compose into search, assistants, form‑fillers, data extraction, and more.

---
### **How LLMs work (just enough to be dangerous)**

1. **Tokenization.** Input text → tokens (IDs). Whitespace and punctuation matter—“token‑budget math” is a real constraint.
2. **Embeddings.** Each token ID becomes a vector; positions are encoded so order matters.
3. **Transformer layers.** Self‑attention mixes information across positions so the representation at each step is **contextual** (richer than the raw embedding).
4. **Decoding.** The model outputs a probability distribution over the next token.
5. **Sampling.** You choose how “adventurous” generation is (see knobs below), append the token, and repeat until done. 
---
### **The knobs you’ll touch most**
- **Temperature**: 0.0–2.0. Lower → more deterministic/boring; higher → more diverse/creative.
- **Top‑p (nucleus)**: 0–1. Samples only from the most probable tokens whose cumulative probability ≤ _p_.
- **Max tokens**: hard limit on output length; protects latency and cost.
- **System/instructions**: steer behavior up front (role, constraints).
- **Stop sequences**: cleanly cut off output at known boundaries.
- **Streaming**: start receiving tokens as they are generated; improves perceived latency and UX.

> Practical defaults: temperature=0.2–0.7, top_p=1.0, set a **max output** that fits your UI, and **stream** for chat‑like UIs.
---
### **Effectiveness comes from context**
- **Context window.** Inputs + outputs share a finite token budget. Plan prompts and retrieval to fit.
- **Grounding with your data (RAG).** Retrieve relevant snippets and include them in the prompt to improve factuality.
- **Structured outputs.** Ask for JSON (and validate) when you need machine‑readable results.
- **Few‑shot examples.** Show the model how to respond (schema + 1–3 exemplars) to stabilize style and structure.
---
### **Limitations (design around these)**

- **Hallucinations.** LLMs will confidently generate plausible but false statements. Ground with citations/RAG; validate critical outputs.
    
- **Recency.** Models don’t inherently “know” the latest facts unless you retrieve or provide them.
    
- **Ambiguity sensitivity.** Vague prompts → vague answers; specify domain, audience, length, and format.
    
- **Determinism.** Even at temperature=0, generation can vary across runs/envs; don’t promise bit‑for‑bit reproducibility.
    
- **Cost and latency.** Longer prompts and bigger models are slower and more expensive; iterate toward the smallest model that meets quality.
    

---

### **Common gotchas**

- **Token mismatch with UI.** “Characters” ≠ “tokens.” Budget both input and output to avoid truncation.
    
- **Over‑engineering prompts.** Prefer simple, testable instructions; add examples sparingly.
    
- **Leaky formats.** If you need JSON, enforce it (schema, validators) and add a fallback repair step.
    
- **One prompt for all tasks.** Separate prompts per task/endpoint; design for testability and versioning.
    
- **Skipping evaluation.** Keep a tiny dataset of real tasks; score changes when you tweak prompts, models, or retrieval.
    

---

### **Minimal streaming example**

  

**Python**

```
from openai import OpenAI
client = OpenAI()

with client.responses.stream(
    model="gpt-4o",
    input="Stream a two-sentence explanation of context windows."
) as stream:
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="")
```

> Both Python and Node SDKs support streaming via the Responses API. See SDK streaming examples for details. 

---

### **Key terms (cheat sheet)**

- **Token**: the unit models read/write (≈ subword).
    
- **Embedding**: vector representation of a token or text span.
    
- **Context window**: max tokens the model can attend to at once (prompt + output).
    
- **Temperature / top‑p**: randomness controls during sampling.
    
- **System/instructions**: up‑front guidance that shapes responses.
    
- **RAG**: Retrieval‑Augmented Generation—retrieve relevant data and include it in the prompt.
    

---

### **Where to go next**

- Prompt patterns for structured outputs
    
- Retrieval‑augmented generation (RAG) basics
    
- Evaluating LLM quality (offline and online)
    
- Streaming UX patterns and backpressure handling
    
- Safety and policy‑aware prompting
    

---

## **Why this rewrite is more effective**

- **Actionable**: adds ready‑to‑run Python/JS snippets and a streaming pattern. 
    
- **Precise**: distinguishes raw embeddings from contextual representations; removes time‑sensitive model claims. 
    
- **Practical**: surfaces the knobs developers actually tune (temperature, top‑p, context, max tokens) and the trade‑offs (cost/latency/quality).
    
- **Honest**: names limitations (hallucinations, determinism) and mitigation strategies (RAG, validation).
    
- **Navigable**: includes a glossary and “next steps” to orient readers deeper into the Cookbook.
    

---

### **Notes on changes from your draft**

- Retitled and restructured for Cookbook conventions (overview → quick start → concepts → limits → glossary).
    
- Rewrote the embeddings paragraph to clarify that _relationships emerge in contextual representations_, not just raw embedding proximity.
    
- Removed version‑specific references (e.g., which GPT version powered early ChatGPT) to avoid dating the page.
    
- Added code and streaming examples based on the official SDK docs to ensure accuracy. 
    

  

If you want, I can also deliver this as a Markdown file with front‑matter and anchors formatted to your Cookbook’s sidebar patterns.