# Payton Ison

I work on language models at [The Singularity AI](https://thesingularityai.org/). I like being able to follow a byte through a Transformer, see where a browser got the wrong image, or point to the exact frame a model was looking at when it answered. I use Codex to build and test things with me. My part is deciding what we're trying to find out and what would convince me we got it right.

## Zoetrope

What happens when you ask a visual assistant about a light and the light changes while it's answering? I wrote the Zoetrope proposal to get at that question. Codex built the prototype and runs the experiments. [Zoetrope on GitHub](https://github.com/paytonison/zoetrope)

In our first two replays, the model saw the same green frame and answered “green” both times. In one recording the light stayed green. In the other it turned red after 267 milliseconds. Zoetrope kept watching, so it could tell us which answer still described the light. [First replay results](https://thesingularityai.org/research/zoetrope/phase-one/)

On November 18, 2026, we tried it in a live macOS window. The local tracker called out two green pulses while each was still visible. The second pulse came and went while the model was still answering about the first. Its answer was right about the frame it saw. By the time it arrived, Zoetrope marked it as history. [Live window trace](https://github.com/paytonison/zoetrope/blob/main/experiments/2026-11-18-two-pulses/window-openai.jsonl)

## Models and papers

- **Tater Tot** is my private byte-level language-model lab. I can train a small C11 Transformer from scratch, watch the loss in a native Mac app, and load a checkpoint to see what it learned. The app runs the same C++ tools I use from the terminal. My early design notes for a low-level Transformer framework are public in [tot](https://github.com/paytonison/tot).
- **Ziggy** is where I'm working through backpropagation by hand in C. The network is small enough that I can pick one weight and follow its contribution to the error.
- **Wayfinder** is an early experiment in letting a local language model pick a useful place to go on the web. [Wayfinder on GitHub](https://github.com/paytonison/wayfinder)
- Our research library has 14 papers. **Ouroboros** proposes turning a conversation into a harder test that a human can score. **Magi** asks when a question is worth sending to another specialist and how to settle it when the answers disagree. [Ouroboros](https://thesingularityai.org/ouroboros/) · [Magi](https://thesingularityai.org/research/magi-gpt-6/) · [Research library](https://thesingularityai.org/research/)
- In **Causal Capacity Quantum Gravity**, I ask whether spacetime geometry can emerge from causal relations and finite information capacity. I lay out what follows from the model and what I still have to prove. [Read the PDF](causal_capacity_quantum_gravity.pdf) · [LaTeX source](causal_capacity_quantum_gravity.tex)

## Other things I've built

- **The Macintosh** recreates the black-and-white Mac desktop in Electron, down to the outline that grows out of a folder when you double-click it and shrinks back when you close it. It has a persistent Finder, a virtual disk, and a working word processor. [The Macintosh](https://github.com/paytonison/macintosh)
- **Dungeoning a Dragon** is an offline D&D character builder in C++ and Qt. It handles the original 1981 Basic/Expert rules and has experimental support for the 2014 and 2024 editions. [Dungeoning a Dragon](https://github.com/paytonison/dnd)
- I build **Safari userscripts** for things I want to do without leaving the page. Hover Zoom lets me inspect and pin an image under the cursor. The downloaders save photos and videos from the post I'm looking at. [hover-zoom](https://github.com/paytonison/hover-zoom) · [insta-loader](https://github.com/paytonison/insta-loader) · [x-loader](https://github.com/paytonison/x-loader) · [threads-loader](https://github.com/paytonison/threads-loader)
- **EMBER** is my playable Doom-style engine in C and SDL2. It draws every pixel in software at 320 × 200. Let go of the movement keys and you coast. Turn mid-stride and your momentum still carries you in the old direction.

## Get in touch

You can reach me at [payton@thesingularityai.org](mailto:payton@thesingularityai.org). Follow The Singularity AI on [Instagram](https://www.instagram.com/thesingularity_ai/) and [Facebook](https://www.facebook.com/profile.php?id=61594975332377).
