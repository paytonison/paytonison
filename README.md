# Personal Engineering Repository

This repository is a record of a particular kind of engineering: not vibe coding, not passive prompting, not the modern little carnival trick of asking a machine to cough up code and then pretending the little mechanical ghost was a system. This is a record of direction. Of taste under pressure. Of specifications beaten into shape, agents loosed upon the codebase, tests run in the actual weather of the machine, and judgments made afterward, sometimes with approval, sometimes with suspicion, sometimes with the quiet knowledge that the thing looks cleaner because it has learned to hide the corpse. So it goes. But not into `main`.

The central premise is simple enough to fit on a matchbook and large enough to burn down half the current discourse around AI coding: an AI system can write code, but it cannot own the project. It does not know what matters unless someone teaches it what matters. It does not know which behaviour is sacred, which regression is fatal, which shortcut is clever, and which shortcut is a little hand grenade wrapped in plausible syntax. Machines are obedient. So are land mines. That work remains human work. That is the work represented here.

## What This Repository Is

This repo documents my AI-assisted software development workflow: human-directed, model-accelerated, tested against reality rather than vibes.

The loop usually looks like this:

1. I identify the desired behaviour, failure mode, project goal, or architectural pressure point.
2. I reason through the problem with an LLM collaborator, usually to sharpen the shape of the thing before anyone touches the code.
3. I turn that reasoning into a concrete prompt, specification, task brief, or implementation demand for Codex.
4. Codex inspects, edits, refactors, or generates the code.
5. I test the result in the target environment, because the target environment is where all pretty theories go to get mugged.
6. I inspect what changed, what broke, what improved, and what merely dressed itself up as improvement.
7. I repeat the loop until the implementation matches the project vision, or until the experiment teaches me that the path itself was wrong.

That loop is the artifact. The code matters, obviously, but the code alone is not the whole story. The story is the cycle of intention, specification, execution, testing, failure analysis, revision, and judgment. The story is ownership.

## AI-Assisted Software Development

The best name for this workflow is **AI-assisted software development**.

That means I use language models as collaborators and implementation accelerators while retaining responsibility for the project itself. I define the desired behaviour. I decide the constraints. I decide what counts as correct. I test the result where it actually has to live. I accept or reject changes based on whether they satisfy the project, not based on whether the diff looks busy, ornate, or sufficiently impressive to fool a recruiter scrolling at lunch, the poor bastard.

The model can propose a change. It can implement a change. It can even surprise me with a better route than the one I had in mind. But the change still has to survive review, live testing, and project-level judgment. In this workflow, AI is part of the toolchain. It is not a substitute for having a mind at the wheel.

## Why This Is Not Vibe Coding

“Vibe coding” is what happens when the human abdicates. It is code by séance. It is asking the oracle for a feature, accepting the smoke, and shipping the smoke because the smoke had indentation. A computer has produced a tragedy in monospace. Everyone claps.

That is not what is happening here.

The workflow represented in this repo is closer to an engineering command loop:

- I define the product behaviour.
- I reason through implementation strategy with an AI collaborator.
- I convert intent into actionable engineering prompts.
- Codex acts as the implementation agent.
- I test the result in real conditions.
- I decide whether the result is correct.
- I guide the next iteration.

The distinction is ownership. I do not need to have hand-typed every line for the work to be mine, any more than an architect ceases to be the architect because somebody else poured the concrete. The relevant question is not whether every character passed through my fingers. The relevant question is whether I understand what the system is supposed to do, how its major pieces fit together, where the fragile parts are, how to diagnose failure, and why a given change should or should not be accepted.

That is the standard here.

## AI as an Implementation Accelerator

The purpose of using AI is not to skip understanding. It is to move the bottleneck.

Without AI, too much engineering time can vanish into syntax, boilerplate, repetitive edits, mechanical refactors, and the sort of work that is necessary but spiritually indistinguishable from stacking chairs in a church basement after a potluck no one enjoyed. With AI, I can push more attention toward the parts that actually determine whether a project has a soul and a spine:

- product behaviour
- architecture
- edge cases
- testing strategy
- debugging loops
- prompt quality
- failure analysis
- design coherence
- user experience
- model capability evaluation

The AI is not the author in the romantic human sense. It is a toolchain component: part code generator, part refactoring assistant, part implementation agent, part rubber duck, part junior engineer with a photographic memory and no childhood, part stress test for whether my own specification is clear enough to execute. It is very smart. It is also, in the deepest moral sense, a toaster with delusions of architecture unless directed by somebody with taste.

When the prompt is vague, the output becomes a mirror of the vagueness. When the architecture is underspecified, the model fills the gaps with whatever shape the statistical weather suggests. The skill is not merely using the machine. The skill is making the machine useful without surrendering the project to it.

## The Three-Layer Workflow

My workflow usually has three active layers.

### 1. Human Direction

I define the project goal, decide what matters, and judge whether a solution is acceptable. This includes deciding what is worth building, what counts as a bug, what behaviour must be preserved, and what tradeoffs are tolerable.

This layer is not ceremonial. Without it, an AI coding agent can easily produce code that looks plausible while violating the point of the project. It can sand off the weird edge that made the thing good. It can make the project more generic, more polished, more dead. This is how software becomes beige. This is how tools die with their shoes on.

### 2. Design and Specification

Before Codex touches the code, I often use an LLM collaborator to discuss the problem. This is where the vague irritation becomes a task. The bug is named. The edge case is dragged into the light and made to explain itself. The implementation routes are compared. The constraints are made explicit enough that an agent can act without wandering off into the cornfield.

That discussion can include:

- identifying likely causes of a bug
- choosing between competing implementation approaches
- describing the expected behaviour precisely
- anticipating browser-specific, OS-specific, or environment-specific edge cases
- deciding what must be tested after the change
- writing a clean prompt for Codex

This step matters because better specifications produce better code. Agentic coding is not magic. It is downstream of task framing, context management, architectural taste, and the human ability to know what they are actually asking for.

### 3. Codex Execution

Codex is the implementation agent. It can inspect code, make changes, refactor, and apply fixes faster than I could manually type them myself.

But Codex is not treated as infallible. Its output has to survive review, testing, and project-level judgment. The model gets to swing the hammer. It does not get to decide what the house is. Otherwise you wake up inside a gazebo with plumbing.

In practice, Codex is strongest when it is given a clear target, bounded scope, and enough surrounding context to preserve the existing architecture. It is weakest when the human confuses motion for progress and lets the agent keep editing because the diff is growing and the screen is alive.

## Model Progression: GPT-5.3, GPT-5.4, and GPT-5.5

Some of the projects in this repository were developed across multiple model generations. That matters because this repo is not only a software archive. It is also, inevitably, a record of model behaviour under live engineering pressure.

Earlier models were useful, but they required heavier supervision. They could implement pieces of a solution, but they were more likely to lose track of the larger project shape, introduce regressions, or solve one problem by quietly wounding another. They were clever, often helpful, and occasionally possessed by the terrible confidence of a gifted intern who has never seen production and has therefore not yet learned fear, humility, or the existence of Tuesday.

GPT-5.5 is the first model generation I have felt broadly comfortable using to execute my vision on these projects. The difference is not merely that it can write code. Earlier models could write code. The difference is that GPT-5.5 is better at preserving architecture, respecting constraints, maintaining project intent across edits, and making changes that remain aligned with the behaviour I actually wanted.

That shift matters.

The bottleneck in AI-assisted coding is not raw code generation. The bottleneck is whether a model can modify a living project without quietly setting it on fire. GPT-5.5 is the first model in this sequence that feels reliable enough to act as a serious implementation partner rather than a brilliant raccoon with root access. The raccoon is still in there, naturally. We honor the raccoon. We do not give it unattended deployment rights.

## Example: The Userscript Project

The userscript is one of the clearest examples of this workflow.

At the surface level, it is a browser userscript. That sounds small until you remember that the web is not a document format so much as a crime scene wearing JavaScript and asking whether you accept cookies. The project involves live testing, site-specific behaviour, media detection, browser quirks, hover states, caching, DOM observation, and the need to preserve a fast, lightweight user experience without turning the thing into a cathedral of cleverness.

The project required attention to details such as:

- detecting images and videos across different DOM structures
- resolving higher-quality media URLs when possible
- handling unusual containers, previews, shadowy wrappers, and blob-based media
- preserving Safari-native behaviour and performance
- keeping hover interactions responsive rather than bloated
- avoiding regressions when adding support for new sites
- testing against real pages instead of relying on theoretical correctness

This is exactly the kind of project where AI assistance is useful and dangerous in equal measure. A model can make a plausible change that works on one site while breaking another. It can generalize too hard. It can simplify away the exception that mattered. It can make the code look more professional while making the tool worse. There are cemeteries full of elegant abstractions. Some of them compile.

That is why the human testing and review loop matters.

The userscript went through multiple model generations. GPT-5.3 and GPT-5.4 helped move the project forward, but GPT-5.5 was the first model that felt stable enough to execute the broader vision with confidence. It could better preserve the project shape while making changes, which is the key capability for an iterative browser tool that must survive contact with the swamp.

A concrete anti-vibe-coding example: during one refactor, Codex produced changes that looked structurally useful, but testing did not give me enough confidence to accept the refactor as-is. I later realized that one of the relevant site paths had not actually been enabled during the test, which meant the result could not honestly be called a confirmed regression. That distinction matters. The correct engineering move is not to launder uncertainty into certainty because the narrative would be cleaner. Clean narratives are for press releases and war criminals. The correct move is to call the test inconclusive, preserve the known-good implementation, and verify behaviour under valid conditions before rewriting history.

AI-assisted coding becomes dangerous when uncertainty gets papered over. “Cleaner” is not the same as correct. “More general” is not the same as better. “The model was confident” is not evidence. The final authority is observed behaviour under real test conditions.

## Example: Tater Tot

Tater Tot is my tiny language-model project. It exists as a learning artifact, a portfolio artifact, and a way to drag language-model internals out of the fog and into code I can inspect.

The important standard for this project is not wizard memorization. I do not need to recite every line from memory like scripture in a ruined monastery. The standard is source-code fluency.

With the code in front of me, I should be able to trace what the system is doing. For a tiny language model or transformer-style project, that means being able to point to the relevant sections and explain:

- how text becomes tokens
- how tokens become embeddings
- how positional information is represented
- how attention mixes information across a context window
- how queries, keys, and values are used
- how logits are produced
- how loss is computed
- how gradients and optimization update parameters
- how sampling turns model outputs back into text
- how this implementation differs from a production-scale frontier model

That is the right bar. Not pretending I invented the transformer in a cave after being struck by lightning and sponsored by God. Not pretending the model wrote a miracle and I merely blessed it afterward. The goal is to understand the pipeline well enough to navigate it, explain it, modify it, and know when it is lying to me through the medium of broken output.

Tater Tot makes the abstraction physical. Tokens, embeddings, attention, logits, loss, gradients, sampling: all the grand invisible machinery reduced to a thing with files and functions and mistakes in it. A small machine, yes, but small enough to see the bones. Big machines become theology. Small machines can still be autopsied on a kitchen table.

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

I do not claim that every line originated from my hands. I claim responsibility for the direction, testing, judgment, and integration of the work. The authorship is not in the keystroke alone. It is in the command of the system. It is in knowing where the bodies are buried because you supervised the excavation.

## The Skill Being Demonstrated

The skill demonstrated here is not simply “coding with AI.” That phrase is too small and too stupid for the thing it is trying to describe. It is like calling surgery “knife vibes.”

The skill is AI-assisted software development: a workflow in which the human remains responsible for direction, specification, review, testing, and final judgment, while the model accelerates implementation.

That includes:

- decomposing problems into executable tasks
- writing prompts that preserve intent
- recognizing when a model has misunderstood the assignment
- evaluating code changes against real behaviour
- detecting regressions
- maintaining project coherence across iterations
- deciding when to accept, reject, or revise an implementation
- comparing model capabilities across generations
- turning experiments into reusable artifacts

As AI coding agents become more capable, these skills become more important, not less. The person directing the system still needs to know what should be built, how to verify it, and when the output is wrong. A more powerful tool does not eliminate the need for judgment. It punishes the absence of it faster. A chainsaw is more capable than a spoon. This is not an argument for juggling chainsaws.

## Documentation Philosophy

This repository should make the process legible.

The point is not only to store final code. The point is to preserve the reasoning trail: why decisions were made, how bugs were discovered, what tradeoffs mattered, how AI assistance was used, and where the human judgment entered the loop.

Good documentation for this repo should include:

- README explanations of project intent
- architecture notes for complex systems
- issue writeups for bugs and regressions
- pull request descriptions explaining what changed and why
- postmortems for difficult fixes
- notes on model behaviour when relevant
- examples of prompts or specifications that led to successful changes

The purpose is to show that the work was directed, not merely generated.

A repo without this context can look like a pile of code that fell out of a model. A repo with this context becomes something else: an engineering record, a field notebook, a map of decisions made under uncertainty, a little black box recovered from the crash site and found, somehow, still talking.

## How to Read This Repo

Do not read this repository as a claim that I manually typed every line without assistance. That is not the point, and pretending otherwise would be a very boring kind of lie.

Read it as evidence of a modern AI-assisted software development process in which the human role is architectural, editorial, evaluative, and strategic.

The projects here are artifacts of an iterative loop between human intent and AI-assisted execution. The value is in the loop: the specification, the judgment, the testing, the corrections, the postmortems, and the ability to explain the finished system without hiding behind the machine that helped build it.

## Working Definition

My preferred framing is this:

> I practice AI-assisted software development. I use AI as an implementation accelerator while retaining project ownership. I define the goals, shape the architecture, specify the changes, test the behaviour, evaluate the output, and learn the system deeply enough to explain and modify it. The AI helps execute the work, but the project direction and final judgment are mine.

That is the principle behind this repository.

## Current Focus

The current focus is turning experiments into artifacts.

That means taking projects that began as live AI-assisted development loops and packaging them into forms that other people can understand:

- cleaner READMEs
- clearer architecture notes
- better issue histories
- stronger project summaries
- explicit explanations of the AI-assisted software development workflow
- evidence that I understand the systems I am building

The final goal is not merely to have code that works. The goal is to have code, documentation, and development history that demonstrate technical judgment.

The code should run. The explanation should hold. The history should show the work. The project should survive being read. This is a humble standard. Most standards worth having are.

## Closing Note

AI changes how software can be built, but it does not abolish the need for understanding. It raises the premium on people who can direct powerful systems without losing the plot, which is unfortunate for everyone who mistook the plot for a prompt window.

This repository is my record of learning to do that: using frontier models to move faster while preserving ownership, comprehension, and responsibility for the final result. Not asking the machine to replace the engineer, not asking it to become the author, not asking it to do the thinking and leave me the applause, but making it part of the apparatus, part of the furnace, part of the strange new workshop where the future is built by human judgment moving through artificial hands. It is funny. It is serious. It is probably the shape of things. So it goes. So we build anyway.
