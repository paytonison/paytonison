#!/bin/zsh
set -euo pipefail

# --- config ---
OBJ="${OBJ:-Demo: harden PIL imports in image_utils.py}"
K="${K:-40}"                 # max files Scout-B opens
BR="${BR:-feat/duo-scouts}" # working branch

# --- helpers ---
exists() { command -v "$1" >/dev/null 2>&1; }
err() { print -ru2 -- "error: $*"; exit 1; }

exists rg || err "ripgrep (rg) required"
exists jq || err "jq required"

# --- manifest + candidates (fast lexical sieve) ---
if [[ ! -f .manifest ]]; then
  rg --files -uu --hidden -g '!.git' > .manifest
fi
rg -n --color=never -e 'PIL|ImageOps|Resampling|image_utils' -f .manifest > .candidates.rg || true
cut -d: -f1 .candidates.rg | sort -u > .candidates.files || true

# --- scratchpad bootstrap ---
[[ -f scratchpad.json ]] || cat > scratchpad.json <<'JSON'
{
  "schema": 1,
  "objective": "TBD",
  "constraints": ["zsh shell", "POSIX-first", "no secrets", "diff-first"],
  "notes": [],
  "artifacts": [],
  "requests": []
}
JSON

# --- git prep ---
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || err "run inside a git repo"
git switch -c "$BR" 2>/dev/null || git switch "$BR"

# --- controller notes (optional) ---
jq --arg o "$OBJ" '.objective=$o' scratchpad.json > scratchpad.json.new && mv scratchpad.json.new scratchpad.json

print -- "SHELL=$SHELL"
print -- "Objective: $OBJ"
print -- "Candidates: $(wc -l < .candidates.files) files"

print -- "\n== ROUND 1 =="
print -- "Open two Chat tabs as Scout-A and Scout-B with the prompts below."
print -- "\nAfter each returns JSON, save to /tmp/A1.json and /tmp/B1.json, then run:"
print -- "  jq -s '.[0] * {notes:((.[0].notes//[])+(.[1].notes//[])),artifacts:((.[0].artifacts//[])+(.[1].artifacts//[])),requests:((.[0].requests//[])+(.[1].requests//[]))}' scratchpad.json /tmp/A1.json > scratchpad.json.new && mv scratchpad.json.new scratchpad.json"
print -- "  jq -s '.[0] * {notes:((.[0].notes//[])+(.[1].notes//[])),artifacts:((.[0].artifacts//[])+(.[1].artifacts//[])),requests:((.[0].requests//[])+(.[1].requests//[]))}' scratchpad.json /tmp/B1.json > scratchpad.json.new && mv scratchpad.json.new scratchpad.json"
print -- "\nThen apply diffs with:  duo-apply\n"
