# Mario Clone (Pygame + Responses API)

This repository contains a compact Mario-style platformer built with Python and Pygame. The game exposes a JSON state snapshot each tick and can accept simple action commands from an agent, including an optional OpenAI Responses API controller.

## Features

- Goomba enemies that can be stomped or avoided
- Coins that increase score; every 10 coins grants an extra life
- Goalpost to complete the level
- Pitfall gaps that cost a life if you fall
- Simple graphics and synthesized sound effects

## Requirements

- Python 3.10+
- Pygame (`pip install -r requirements.txt`)

## Run

```bash
python src/main.py
```

### Controls

- Arrow keys or WASD to move
- Space to jump
- R to restart after win or game over
- Esc to quit

## Agent Mode

Use a built-in heuristic agent:

```bash
python src/main.py --agent heuristic
```

Use the Responses API (requires `OPENAI_API_KEY`):

```bash
OPENAI_API_KEY=... python src/main.py --agent responses --model gpt-5.2
```

Write the latest state JSON to a file for external clients:

```bash
python src/main.py --state-file state.json
```

Launch the browser client (serves a local dashboard for actions and state):

```bash
python src/main.py --agent browser
```

Then open `http://127.0.0.1:8765` in your browser.
