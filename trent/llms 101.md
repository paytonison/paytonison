# LLMs 101: What Even Is It?

### Preface
A large language model (LLM) is an artificial neural network, a type of machine learning algorithm that mimics organic neural networks, trained on vast quantities of text-based data. The most widespread LLM-type artificial neural networks are known as Generative Pre-Trained Transformers (GPTs), which serve as a backbone for artificial intelligence. These gained international recognition when OpenAI released their ChatGPT chatbot that used GPT-3, thereby kicking off the AI Arms Race.

To understand LLMs, you have to know a little history before you begin. In 2017, a group of Google researchers (aka, nerds) wrote the monumental paper, "Attention Is All You Need," which outlined a type of neural network called a "transformer." The transformer takes data, in this case text, converts the text into numerical values, passes the values into a mass of matrices that undergo matrix multiplication, pop out a final numerical result, which is then converted to human-readable text. It's like translating English to machine code and back again.

The mechanic that underlies the heart of all transformers is a mathematical process called "matrix multiplication." This linear algebraic method takes two blocks of numbers, separated into columns and rows, and then specially multiplies them, creating a new matrix. In artificial intelligence circles, these are referred to as "weights."

### What are they used for?
While called "large language models," LLMs can take various types of inputs and generate outputs. All you need to do is take a piece of data, translate it into the numerical values the neural network can understand, have it do its math, pop out the final result, and decode it. Will any data work? Only if it's trained explicitly for a particular type of data in context, but we'll cover that later.

For language-based tasks, LLMs excel. Some types of language-based use cases include:
* Text generation
* Text completion
* Text analysis
* Translation
* Semantic analysis
* Question answering
* Agentic tasks
* Countless others

We'll stick with the first three for now.

### Text generation
All of the math is well and good, but how *exactly* does it work? In short, it's a prediction algorithm. The LLM is trained on a massive corpus of pre-existing data that is individually labeled, taught to understand this and the relationships between other words, and then predict what words come next following a user's input, be it a query or prompt to generate creative text.

The table is called an "embedding matrix." Traditionally, the embedding matrix assigns numerical values to words and estimates their relationship to each other. For example, the king is to the emperor as the queen is to the empress. While these aren't synonyms, they do have similar connotations. All four words involve royalty, and all four words involve gender. However, the words "emperor" and "empress" are weighted more highly than "king" and "queen" because an emperor (or empress) rules over an empire—a vast domain of political subdivisions, rather than a self-contained polity.

Then, these words and values are split up based on bit values that approximate word groups found in words called phonemes. Using the embedding matrix, the model finds the appropriate "phoneme" and strings together which ones are more likely to come next.

Do this enough times, and presto, you have a string of words that may or may not be coherent.

