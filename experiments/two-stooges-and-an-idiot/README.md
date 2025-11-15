# Two Stooges and an Idiot

Welcome to the dumbest smart experiment in the repo.

- **The Two Stooges**: Two Codex instances, running side by side, each tasked with a different slice of the problem.  
- **The Idiot**: The user (that’s me). I’m the controller with the bonk stick, the one who points them at the files, merges their JSON notes, applies their diffs, and inevitably trips over my own shoelaces in the process.

## What this is

A proof-of-concept for parallel Codex workflows:
- Each agent works from the same **scratchpad.json** (append-only, structured).
- One scouts the backend/tests, the other scouts CLI/DX/perf.
- They swap notes, diffs, and requests through JSON instead of giant prompts.
- I, the Idiot, reconcile their output using `duo.sh` and `duo-apply`.

## Why bother?

Because running one Codex across a hundred million files (yes, literally) is like asking Moe to solve calculus. Splitting the work makes it:
- Faster (bounded scope per agent).
- Cheaper (less context blow-up).
- Funnier (watching two LLMs argue in JSON is peak slapstick).

## Files

- `scratchpad.json` — shared brain between the stooges.  
- `duo.sh` — orchestration script, sets up candidates and scratchpad.  
- `duo-apply` — applies diffs in order, with 3-way merge fallback.  
- `prompts.md` — the scripts for Larry and Curly to play their parts.  

## How to run

```zsh
# bootstrap scratchpad and candidate list
./duo.sh

# open two Chat tabs
#  - run prompts.md: Scout-A
#  - run prompts.md: Scout-B
# save their JSON outputs to /tmp/A1.json and /tmp/B1.json

# merge into scratchpad.json
jq -s '.[0] * {notes:((.[0].notes//[])+(.[1].notes//[])),artifacts:((.[0].artifacts//[])+(.[1].artifacts//[])),requests:((.[0].requests//[])+(.[1].requests//[]))}' scratchpad.json /tmp/A1.json /tmp/B1.json > scratchpad.json.new
mv scratchpad.json.new scratchpad.json

# apply their diffs
./duo-apply
