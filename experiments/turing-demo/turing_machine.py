from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional


Symbol = str
State = str
Move = str  # "L", "R", or "N"


@dataclass(frozen=True)
class Transition:
    next_state: State
    write_symbol: Symbol
    move: Move


class Tape:
    """A sparse, unbounded tape with a default blank symbol.

    The tape grows left/right as needed. Only non-blank cells are stored.
    """

    def __init__(self, input_string: str = "", blank: Symbol = "_") -> None:
        self.blank = blank
        self._cells: Dict[int, Symbol] = {
            i: ch for i, ch in enumerate(input_string) if ch != blank
        }

    def read(self, position: int) -> Symbol:
        return self._cells.get(position, self.blank)

    def write(self, position: int, symbol: Symbol) -> None:
        if symbol == self.blank:
            self._cells.pop(position, None)
        else:
            self._cells[position] = symbol

    def contents(self) -> str:
        if not self._cells:
            return ""
        min_i = min(self._cells)
        max_i = max(self._cells)
        return "".join(self._cells.get(i, self.blank) for i in range(min_i, max_i + 1)).strip(self.blank)

    def snapshot(self, head_position: int) -> str:
        if not self._cells:
            return f"[{self.blank}]"
        indices = list(self._cells.keys()) + [head_position]
        start = min(indices)
        end = max(indices)
        parts: List[str] = []
        for i in range(start, end + 1):
            s = self._cells.get(i, self.blank)
            parts.append(f"[{s}]" if i == head_position else f" {s} ")
        return "".join(parts).strip()


class TuringMachine:
    """A single-tape deterministic Turing machine.

    Transitions are provided as a mapping from (state, symbol) -> Transition.
    The machine halts when it enters an accept state.
    """

    def __init__(
        self,
        transitions: Dict[Tuple[State, Symbol], Transition],
        start_state: State,
        accept_states: Iterable[State],
        blank_symbol: Symbol = "_",
    ) -> None:
        self.transitions = transitions
        self.state = start_state
        self.accept_states = set(accept_states)
        self.blank = blank_symbol

    def step(self, tape: Tape, head: int) -> int:
        read_symbol = tape.read(head)
        key = (self.state, read_symbol)
        if key not in self.transitions:
            raise RuntimeError(f"No transition for state={self.state!r}, symbol={read_symbol!r}")

        t = self.transitions[key]
        tape.write(head, t.write_symbol)
        if t.move == "L":
            head -= 1
        elif t.move == "R":
            head += 1
        elif t.move != "N":
            raise ValueError(f"Invalid move {t.move!r}; expected 'L', 'R', or 'N'")
        self.state = t.next_state
        return head

    def run(
        self,
        tape: Tape,
        head_position: int = 0,
        max_steps: int = 10_000,
        trace: bool = False,
    ) -> List[dict]:
        history: List[dict] = []
        for step in range(max_steps):
            record = {
                "step": step,
                "state": self.state,
                "head": head_position,
                "tape": tape.snapshot(head_position),
            }
            history.append(record)
            if trace:
                print(
                    f"Step {step:>5} | state={record['state']:<10} | head={record['head']:>5} | {record['tape']}"
                )

            if self.state in self.accept_states:
                return history

            head_position = self.step(tape, head_position)
        raise RuntimeError("Exceeded maximum number of steps without halting")


def parse_transitions(spec: Dict[str, List[str]]) -> Dict[Tuple[State, Symbol], Transition]:
    """Parse a transitions mapping from a JSON-friendly form.

    Expected format:
      {
        "q0,0": ["q0", "0", "R"],
        "q0,1": ["q0", "1", "R"],
        ...
      }
    Whitespace around the comma is ignored.
    """
    transitions: Dict[Tuple[State, Symbol], Transition] = {}
    for key, triple in spec.items():
        if not isinstance(triple, list) or len(triple) != 3:
            raise ValueError(f"Invalid transition for {key!r}: expected [next, write, move]")
        parts = [p.strip() for p in key.split(",", 1)]
        if len(parts) != 2:
            raise ValueError(f"Invalid transition key {key!r}; expected 'state,symbol'")
        state, symbol = parts
        next_state, write_symbol, move = triple
        transitions[(state, symbol)] = Transition(next_state, write_symbol, move)
    return transitions


def builtin_binary_incrementer() -> TuringMachine:
    transitions = parse_transitions(
        {
            "seek_end,0": ["seek_end", "0", "R"],
            "seek_end,1": ["seek_end", "1", "R"],
            "seek_end,_": ["add", "_", "L"],
            "add,0": ["halt", "1", "N"],
            "add,1": ["add", "0", "L"],
            "add,_": ["halt", "1", "N"],
        }
    )
    return TuringMachine(transitions, start_state="seek_end", accept_states={"halt"})


def builtin_unary_incrementer() -> TuringMachine:
    """Unary increment: Given n 1s, produce n+1 1s.

    Encoding: input is a string of '1's; blank is '_'.
    Algorithm: Move right to blank, write '1', halt.
    """
    transitions = parse_transitions(
        {
            "seek_end,1": ["seek_end", "1", "R"],
            "seek_end,_": ["halt", "1", "N"],
        }
    )
    return TuringMachine(transitions, start_state="seek_end", accept_states={"halt"})


BUILTINS = {
    "binary_incrementer": builtin_binary_incrementer,
    "unary_incrementer": builtin_unary_incrementer,
}


def load_machine(path: str) -> TuringMachine:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    blank = cfg.get("blank", "_")
    start = cfg["start"]
    accept = cfg["accept"]
    transitions = parse_transitions(cfg["transitions"])
    return TuringMachine(transitions, start_state=start, accept_states=set(accept), blank_symbol=blank)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Single-tape Turing machine runner")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--machine", choices=sorted(BUILTINS.keys()), help="Run a built-in machine")
    g.add_argument("--load", metavar="PATH", help="Load machine from JSON file")
    p.add_argument("--input", default="", help="Initial tape contents (string of symbols)")
    p.add_argument("--head", type=int, default=0, help="Initial head position (default: 0)")
    p.add_argument("--blank", default="_", help="Blank symbol (default: '_')")
    p.add_argument("--max-steps", type=int, default=10_000, help="Max execution steps")
    p.add_argument("--trace", action="store_true", help="Print step-by-step trace")
    p.add_argument("--list", action="store_true", help="List built-in machines and exit")

    args = p.parse_args(argv)

    if args.list:
        print("Built-in machines:")
        for name in sorted(BUILTINS.keys()):
            print(f"- {name}")
        return

    if args.machine:
        tm = BUILTINS[args.machine]()
    else:
        tm = load_machine(args.load)

    tape = Tape(args.input, blank=args.blank)
    print("Turing machine demo")
    print(f"Blank: {tm.blank!r} | Start: {tm.state!r} | Accept: {sorted(tm.accept_states)!r}")
    print(f"Input: {args.input!r}\n")

    history = tm.run(tape, head_position=args.head, max_steps=args.max_steps, trace=args.trace)
    last = history[-1]
    print("\nHalted.")
    print(f"Steps: {last['step']} | Final state: {last['state']}")
    print(f"Final tape: {tape.contents()!r}")


if __name__ == "__main__":
    main()

