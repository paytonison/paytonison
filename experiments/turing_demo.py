from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


Symbol = str
State = str
Move = str  # "L", "R", or "N"


@dataclass(frozen=True)
class Transition:
    next_state: State
    write_symbol: Symbol
    move: Move


class Tape:
    """Sparse tape representation with a default blank symbol."""

    def __init__(self, input_string: str, blank: Symbol = "_") -> None:
        self.blank = blank
        self._cells: Dict[int, Symbol] = {
            index: ch for index, ch in enumerate(input_string)
        }

    def read(self, position: int) -> Symbol:
        return self._cells.get(position, self.blank)

    def write(self, position: int, symbol: Symbol) -> None:
        if symbol == self.blank:
            self._cells.pop(position, None)
        else:
            self._cells[position] = symbol

    def snapshot(self, head_position: int) -> str:
        if not self._cells:
            return f"[{self.blank}]"

        indices: Iterable[int] = list(self._cells.keys()) + [head_position]
        start = min(indices)
        end = max(indices)

        cells: List[str] = []
        for i in range(start, end + 1):
            symbol = self._cells.get(i, self.blank)
            cells.append(f"[{symbol}]" if i == head_position else f" {symbol} ")
        return "".join(cells).strip()

    def contents(self) -> str:
        if not self._cells:
            return ""

        min_index = min(self._cells)
        max_index = max(self._cells)
        symbols = [
            self._cells.get(i, self.blank) for i in range(min_index, max_index + 1)
        ]
        return "".join(symbols).strip(self.blank)


class TuringMachine:
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

    def run(self, tape: Tape, head_position: int = 0, max_steps: int = 100) -> List[dict]:
        history: List[dict] = []
        for step in range(max_steps):
            history.append(
                {
                    "step": step,
                    "state": self.state,
                    "head": head_position,
                    "tape": tape.snapshot(head_position),
                }
            )

            if self.state in self.accept_states:
                return history

            read_symbol = tape.read(head_position)
            key = (self.state, read_symbol)
            if key not in self.transitions:
                raise RuntimeError(f"No transition for state={self.state}, symbol={read_symbol}")

            transition = self.transitions[key]
            tape.write(head_position, transition.write_symbol)

            if transition.move == "L":
                head_position -= 1
            elif transition.move == "R":
                head_position += 1
            elif transition.move != "N":
                raise ValueError(f"Unsupported move: {transition.move}")

            self.state = transition.next_state

        raise RuntimeError("Exceeded maximum number of steps without halting.")


def build_binary_incrementer() -> TuringMachine:
    transitions: Dict[Tuple[State, Symbol], Transition] = {
        ("seek_end", "0"): Transition("seek_end", "0", "R"),
        ("seek_end", "1"): Transition("seek_end", "1", "R"),
        ("seek_end", "_"): Transition("add", "_", "L"),
        ("add", "0"): Transition("halt", "1", "N"),
        ("add", "1"): Transition("add", "0", "L"),
        ("add", "_"): Transition("halt", "1", "N"),
    }
    return TuringMachine(transitions, start_state="seek_end", accept_states={"halt"})


def main() -> None:
    machine = build_binary_incrementer()
    tape = Tape("1011")  # Binary 11

    print("Binary increment Turing machine demo")
    print("Input : 1011 (decimal 11)")
    print("Goal  : Increment to 1100 (decimal 12)\n")

    history = machine.run(tape)
    for record in history:
        print(
            f"Step {record['step']:>2} | state={record['state']:<8} "
            f"| head={record['head']:>2} | tape={record['tape']}"
        )

    print("\nHalted.")
    print(f"Final tape: {tape.contents()} (decimal {int(tape.contents() or '0', 2)})")


if __name__ == "__main__":
    main()
