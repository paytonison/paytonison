# Minimal Mario Clone with Async LLM Controller

A tiny Pygame side-scroller with a non-blocking, latest-only agent controller. The main loop stays at 60 FPS while model inference runs on a background thread.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Keyboard (default):
```bash
python src/main.py --agent keyboard
```

Random agent (async):
```bash
python src/main.py --agent random
```

Model agent (async):
```bash
python src/main.py --agent model --model gpt2
```

Useful flags:
- `--decision-hz 20` throttles inference rate (0 = no throttle)
- `--fps 60` game loop target FPS
- `--width 800 --height 480` window size

## Async agent contract (short)
- The game loop never calls model tokenization or generation.
- Inference runs in a background thread and never blocks rendering/physics.
- State submission is latest-only (no backlog).
- The model returns a single word: `left`, `right`, or `jump`.
- Invalid output falls back safely (last action or `right`).
- Jump is edge-triggered (fires once on transition to `jump`).

## Sanity check

Run a slow fake agent to prove the loop still hits 60 FPS:
```bash
python scripts/slow_agent_sanity.py
```

## Tests

```bash
python -m unittest
```
