# Personal Engineering Repository

This repository is a record of my AI-assisted software development workflow: not "vibe coding," not passive prompting, and not outsourcing understanding to a model. It is a practical archive of how I design, direct, test, refine, and evaluate software projects using frontier language models as collaborators and implementation agents.

The central idea is simple: AI can write code, but it cannot own a project. Project ownership still requires taste, judgment, specification, testing, debugging, architectural understanding, and the ability to decide whether the output actually satisfies the intended behavior. That is the role I occupy in this workflow.

## What This Repository Represents

This repo documents a development process built around human-directed AI-assisted software development.

My process usually works like this:

1. I identify the desired behavior, failure mode, or product goal.
2. I discuss the problem with an LLM collaborator to clarify the design, constraints, and possible implementation strategies.
3. I turn that reasoning into a concrete prompt, specification, or task brief for Codex.
4. Codex modifies or generates the code.
5. I test the result in the actual target environment.
6. I inspect the behavior, identify regressions or mismatches, and decide what should happen next.
7. I repeat the loop until the implementation matches the project vision.

That loop is the important part. The code is not treated as magic output. It is reviewed, tested, revised, and judged against a concrete design goal.

## AI-Assisted Software Development

The best description for this workflow is **AI-assisted software development**.

That means I use AI systems as collaborators and implementation accelerators while retaining responsibility for the project itself. I define the desired behavior, identify the constraints, decide what counts as correct, test the result in the target environment, and accept or reject changes based on whether they satisfy the project requirements.

This is different from treating AI output as finished software. The model can propose or implement a change, but the change still has to survive human review, live testing, and project-level judgment. In this workflow, AI is part of the development toolchain; it is not a replacement for ownership.

## Why This Is Not Vibe Coding

"Vibe coding" implies a lack of structure: asking an AI for code, accepting whatever it produces, and moving forward without understanding the system or its failure modes.

That is not what is happening here.

The workflow represented in this repo is closer to an AI-assisted software development loop:

- I define the product behavior.
- I reason through implementation strategy with an AI collaborator.
- I convert intent into actionable engineering prompts.
- Codex acts as an implementation agent.
- I test the result in real conditions.
- I decide whether the result is correct.
- I guide the next iteration.

This means I am not simply asking a model to "make something." I am directing the system, evaluating the output, preserving the project vision, and learning the structure of the code as the project evolves.

The important distinction is ownership. I do not need to have hand-written every line for the work to be mine. The relevant question is whether I understand what the system is supposed to do, how the major parts fit together, where the fragile areas are, and how to diagnose problems when they appear.

That is the standard I use.

## AI as an Implementation Accelerator

The purpose of using AI here is not to skip understanding. It is to accelerate execution.

A good AI-assisted software development workflow lets me operate at a higher level of abstraction while still remaining responsible for the final system. Instead of spending all of my energy on syntax, boilerplate, and repetitive implementation work, I can focus more attention on:

- product behavior
- architecture
- edge cases
- testing strategy
- debugging loops
- prompt quality
- failure analysis
- design coherence
- user experience
- model capability evaluation

The AI is not the author in the human sense. It is a toolchain component: part code generator, part refactoring assistant, part implementation agent, part rubber duck, and part stress test for whether my own specification is clear enough to execute.

## The Three-Layer Workflow

My workflow usually has three active layers.

### 1. Human Direction

I define the project goal, decide what matters, and judge whether a solution is acceptable. This includes deciding what is worth building, what counts as a bug, what behavior should be preserved, and what tradeoffs are acceptable.

This layer is not optional. Without it, an AI coding agent can easily produce code that looks plausible while violating the actual intent of the project.

### 2. Design and Specification

Before Codex touches the code, I often use an LLM collaborator to discuss the problem. This helps turn a vague issue into an actionable task.

That discussion can include:

- identifying likely causes of a bug
- choosing between competing implementation approaches
- describing the expected behavior precisely
- anticipating browser or environment-specific edge cases
- deciding what should be tested after the change
- writing a clean prompt for Codex

This step matters because better specifications produce better code. The quality of the agent's output is often downstream of the quality of the task framing.

### 3. Codex Execution

Codex is used as an implementation agent. It can inspect code, make changes, refactor, and apply fixes faster than I could manually type everything myself.

But Codex is not treated as infallible. Its output still has to survive review, live testing, and project-level judgment.

In practice, Codex is strongest when it is given a clear target, bounded scope, and enough surrounding context to preserve the existing architecture.

## Model Progression: GPT-5.3, GPT-5.4, and GPT-5.5

Some of the projects in this repository were developed across multiple model generations. That matters because the workflow itself is partly a model-evaluation process.

Earlier models were useful, but they required heavier supervision. They could implement pieces of a solution, but they were more likely to lose track of the larger project shape, introduce regressions, or make changes that solved one problem while damaging another.

GPT-5.5 is the first model generation I have felt broadly comfortable using to execute my vision on these projects. The difference is not merely that it can write code. Earlier models could also write code. The difference is that GPT-5.5 is better at preserving the architecture, respecting the project constraints, and making changes that remain aligned with the intended behavior.

That shift matters.

The bottleneck in AI-assisted coding is often not raw code generation. The bottleneck is whether the model can modify a living project without quietly setting it on fire. GPT-5.5 is the first version in this sequence that feels reliable enough to act as a serious implementation partner rather than a clever but unstable assistant.

## Example: The Userscript Project

The userscript is one of the clearest examples of this workflow.

At a surface level, it is a browser userscript. But the real project is more complicated than that: it involves live testing, site-specific behavior, media detection, browser quirks, hover states, caching, DOM observation, and the need to preserve a fast, lightweight user experience.

The project required attention to details such as:

- detecting images and videos in different DOM structures
- resolving higher-quality media URLs when possible
- handling sites with unusual containers, previews, or blob-based media
- preserving Safari-native behavior and performance
- keeping hover interactions responsive rather than bloated
- avoiding regressions when adding support for new sites
- testing against real pages instead of relying only on theoretical correctness

This is the kind of project where AI assistance is extremely useful but also dangerous if unsupervised. A model can easily make a plausible change that works on one site while breaking another. That is why the human testing and review loop matters.

The userscript went through multiple model generations. GPT-5.3 and GPT-5.4 helped move the project forward, but GPT-5.5 is the first model that felt stable enough to execute the broader vision with confidence. It could better preserve the project shape while making changes, which is the key capability needed for this kind of iterative browser tool.

A concrete anti-vibe-coding example: during one refactor, Codex produced changes that looked structurally useful, but testing did not give me enough confidence to accept the refactor as-is. I later realized that one of the relevant site paths had not actually been enabled during the test, which meant the result could not honestly be described as a confirmed regression. That distinction matters. Rather than laundering uncertainty into a stronger claim, the right engineering move is to treat the test as inconclusive, discard or defer the change, and preserve the known-good implementation until the behavior can be verified properly.

This matters because AI-assisted coding becomes dangerous when uncertainty gets papered over. A model can make a change that improves organization, style, or generality while quietly damaging an edge case that the project specifically needs to support. Human project ownership means recognizing that "cleaner" is not the same as "correct," and that the final authority is observed behavior under valid test conditions.

## Example: Tater Tot

Tater Tot is my tiny language-model project. It exists partly as a learning artifact, partly as a portfolio artifact, and partly as a way to understand language-model internals from the implementation side.

The important standard for this project is not memorizing every line of code. The standard is source-code fluency.

With the code available, I should be able to trace what the system is doing. For a tiny language model or transformer-style project, that means being able to point to the relevant sections and explain things like:

- how text becomes tokens
- how tokens become embeddings
- how positional information is represented
- how attention mixes information across a context window
- how queries, keys, and values are used
- how logits are produced
- how loss is computed
- how gradients and optimization update parameters
- how sampling turns model outputs back into text
- how the implementation differs from a production-scale frontier model

That is the right bar for this kind of project. Not wizard-like memorization. Not pretending I manually invented every concept from scratch. The goal is to understand the pipeline well enough to explain it while navigating the source.

The purpose of Tater Tot is to move language-model internals out of abstraction and into concrete code. It gives me something I can inspect, modify, test, and reason about directly.

## What I Mean by Project Ownership

Project ownership means I can answer the important questions:

- What is this project trying to do?
- Why does this feature exist?
- What are the major components?
- How does data move through the system?
- What parts are fragile?
- What assumptions does the implementation make?
- What did the AI produce, and what did I direct?
- What did I test?
- What broke during development?
- What changed across iterations?
- What would I inspect first if the project failed?

That is the difference between using AI as a crutch and using AI as an accelerator.

I do not claim that every line originated from my hands. I claim responsibility for the direction, testing, judgment, and integration of the work.

## The Skill Being Demonstrated

The skill demonstrated here is not simply "coding with AI." It is AI-assisted software development: a workflow where the human remains responsible for direction, specification, review, testing, and final judgment.

That includes:

- decomposing problems into executable tasks
- writing prompts that preserve intent
- recognizing when a model has misunderstood the assignment
- evaluating code changes against real behavior
- detecting regressions
- maintaining project coherence across iterations
- deciding when to accept, reject, or revise an implementation
- comparing model capabilities across generations
- turning experiments into reusable artifacts

As AI coding agents become more capable, these skills become more important rather than less. The person directing the system still needs to know what should be built, how to verify it, and when the output is wrong.

## Documentation Philosophy

This repository should make the process legible.

The goal is not only to store final code. It is also to preserve the reasoning trail: why decisions were made, how bugs were discovered, what tradeoffs mattered, and how AI assistance was used responsibly.

Good documentation for this repo should include:

- README explanations of project intent
- architecture notes for complex systems
- issue writeups for bugs and regressions
- pull request descriptions explaining what changed and why
- postmortems for difficult fixes
- notes on model behavior when relevant
- examples of prompts or specifications that led to successful changes

The purpose is to show that the work was directed, not merely generated.

## How to Read This Repo

When reviewing this repository, do not read it as a claim that I manually typed every line without assistance. That is not the point.

Read it as evidence of a modern AI-assisted software development process where the human role is architectural, editorial, evaluative, and strategic.

The projects here are artifacts of an iterative loop between human intent and AI-assisted execution. The value is in the loop: the specification, the judgment, the testing, the corrections, and the ability to explain the finished system.

## Working Definition

My preferred framing is this:

> I practice AI-assisted software development: I use AI as an implementation accelerator while retaining project ownership. I define the goals, shape the architecture, specify the changes, test the behavior, evaluate the output, and learn the system deeply enough to explain and modify it. The AI helps execute the work, but the project direction and final judgment are mine.

That is the principle behind this repository.

## Current Focus

The current focus is to keep turning experiments into artifacts.

That means taking projects that began as live AI-assisted development loops and packaging them into forms that other people can understand:

- cleaner READMEs
- clearer architecture notes
- better issue histories
- stronger project summaries
- explicit explanations of the AI-assisted software development workflow
- evidence that I understand the systems I am building

The final goal is not just to have code that works. The goal is to have code, documentation, and development history that demonstrate technical judgment.

## Closing Note

AI changes how software can be built, but it does not eliminate the need for understanding. If anything, it raises the premium on people who can direct powerful systems without losing the plot.

This repository is my record of learning to do that: using frontier models to move faster while still preserving ownership, comprehension, and responsibility for the final result.
