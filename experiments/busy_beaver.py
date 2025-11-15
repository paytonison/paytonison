"""
Enumerate 2-state, 2-symbol Turing machines to recover the busy beaver champion.

Brute-forces all 20,736 machines, simulates each with a step limit, and reports
the halting machine that leaves the most `1` symbols on an empty tape. This
recovers the known optimum (4 ones in 6 steps) and serves as a quick smoke test.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, Iterator, Tuple

SYMBOLS = (0, 1)
STATES = ("A", "B")
HALT = "HALT"

TRANSITION_KEYS: Tuple[Tuple[str, int], ...] = (("A", 0), ("A", 1), ("B", 0), ("B", 1))


@dataclass(frozen=True)
class Action:
    write: int
    move: str  # "L" or "R"
    next_state: str

    def step(self) -> int:
        return 1 if self.move == "R" else -1


Machine = Dict[Tuple[str, int], Action]


def enumerate_actions() -> Tuple[Action, ...]:
    moves = ("L", "R")
    next_states: Tuple[str, ...] = (*STATES, HALT)
    return tuple(Action(write, move, state) for write, move, state in product(SYMBOLS, moves, next_states))


def generate_machines() -> Iterator[Machine]:
    actions = enumerate_actions()
    for combo in product(actions, repeat=len(TRANSITION_KEYS)):
        yield {key: action for key, action in zip(TRANSITION_KEYS, combo)}


def run_machine(machine: Machine, max_steps: int = 64) -> Tuple[bool, int, int]:
    tape: Dict[int, int] = {}
    state = "A"
    head = 0
    steps = 0
    while steps < max_steps and state != HALT:
        symbol = tape.get(head, 0)
        action = machine[(state, symbol)]
        tape[head] = action.write
        head += action.step()
        state = action.next_state
        steps += 1
    halted = state == HALT
    ones = sum(1 for value in tape.values() if value == 1)
    return halted, steps, ones


def search_busy_beaver(max_steps: int = 64) -> Tuple[int, Machine, int]:
    total = 0
    halting = 0
    best_ones = -1
    best_machine = None
    best_steps = 0
    for machine in generate_machines():
        total += 1
        halted, steps, ones = run_machine(machine, max_steps=max_steps)
        if not halted:
            continue
        halting += 1
        if ones > best_ones or (ones == best_ones and steps > best_steps):
            best_ones = ones
            best_machine = machine
            best_steps = steps
    if best_machine is None:
        raise RuntimeError("No halting machines found.")
    return halting, best_machine, best_steps


def format_machine(machine: Machine) -> Iterable[str]:
    for key in TRANSITION_KEYS:
        action = machine[key]
        state, symbol = key
        yield f"delta({state},{symbol}) -> write {action.write}, move {action.move}, next {action.next_state}"


if __name__ == "__main__":
    halting, machine, steps = search_busy_beaver()
    ones = run_machine(machine)[2]
    print("Enumerated 20,736 two-state machines.")
    print(f"Halting machines: {halting}")
    print(f"Busy beaver candidate leaves {ones} ones after {steps} steps.")
    print("Transitions:")
    for line in format_machine(machine):
        print(f"  {line}")
