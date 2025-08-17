# **LLMs 101: What Even Is It?**
### Preface

A large language model (LLM) is an artificial neural network, a type of machine learning algorithm that mimics organic neural networks, trained on vast quantities of text-based data. The most widespread LLM-type artificial neural networks are known as Generative Pre-Trained Transformers (GPTs), which serve as a backbone for artificial intelligence. These gained international recognition when OpenAI released their ChatGPT chatbot that used GPT-3, thereby kicking off the AI Arms Race.

To understand LLMs, you need a little history first. In 2017, a group of Google researchers (aka, nerds) wrote the monumental paper _Attention Is All You Need_, which introduced the transformer architecture. Transformers take text, break it into smaller units called _tokens_, turn those into numerical values, run them through layers of matrix math and attention mechanisms, and finally convert the result back into human-readable text. Think of it like translating English to numbers, juggling those numbers around, and translating them back.

At the heart of it all is a mathematical process called **matrix multiplication**—a way of combining two grids of numbers to create a new one. In AI, these grids are called **weights**, and adjusting them during training is how models “learn.”

---
### What are they used for?

Despite the name “large language model,” LLMs aren’t limited to just language. They can take in many types of data (text, images, audio—depending on training) and spit out predictions in kind. The trick is always the same: encode the input into numbers the model understands, let it do the math, decode the output.

For language, LLMs shine. Some examples include:
- Text generation
- Text completion
- Text analysis
- Translation
- Semantic search
- Question answering
- Agentic tasks
- Countless others

We’ll stick with the first few here, since they’re the backbone of most Cookbook recipes.

---
### **Text generation**

So how does the magic actually happen? Under the hood, an LLM is a **prediction engine**. It’s trained on a massive corpus of text and learns statistical relationships between tokens. When you type a prompt, the model predicts the most likely next token, then the next, and so on until you stop it.

The lookup table that helps with this is called an **embedding matrix**. Instead of storing whole words, the model breaks text into _tokens_ (subword chunks like play, ##ing, or un). Each token is mapped to a vector in a giant space, where tokens with similar meanings or contexts are close together. For example:

- _king_ is to _queen_ as _emperor_ is to _empress_.
- These aren’t synonyms, but they share semantic and relational properties.
- The model can “see” these relationships numerically.

Using embeddings and attention, the model decides which tokens are most likely to follow. String enough predictions together and—presto—you’ve got a coherent sentence. Or at least, something that looks coherent.

---
### **So what?**

The key idea: LLMs don’t “understand” text like humans do. They crunch probabilities to predict the next token. That prediction power is exactly what you’ll tap into in the Cookbook: generating text, completing thoughts, analyzing documents, and more.

---
