# Turing Machine Demo

Goal
- Provide a minimal, general single-tape Turing machine implementation and a few example machines to demonstrate Turing completeness.

How it works
- A `TuringMachine` executes transitions `(state, symbol) -> (next_state, write_symbol, move)` on an unbounded sparse tape with a configurable blank symbol.
- Machines can be loaded from JSON or chosen from built-in examples.
- Optional step-by-step tracing shows the head position, machine state, and tape contents each step.

Run
- Built-in binary incrementer (adds 1 to a binary number):
  - `python experiments/turing-demo/turing_machine.py --machine binary_incrementer --input 1011 --trace`
- Built-in unary incrementer (adds 1 to a unary number):
  - `python experiments/turing-demo/turing_machine.py --machine unary_incrementer --input 111 --trace`
- List built-ins:
  - `python experiments/turing-demo/turing_machine.py --list --machine binary_incrementer`
- Load from JSON:
  - `python experiments/turing-demo/turing_machine.py --load experiments/turing-demo/samples/binary_incrementer.json --input 1011 --trace`

JSON format
```
{
  "blank": "_",
  "start": "seek_end",
  "accept": ["halt"],
  "transitions": {
    "seek_end,0": ["seek_end", "0", "R"],
    "seek_end,1": ["seek_end", "1", "R"],
    "seek_end,_": ["add", "_", "L"],
    "add,0": ["halt", "1", "N"],
    "add,1": ["add", "0", "L"],
    "add,_": ["halt", "1", "N"]
  }
}
```

Notes
- The provided engine is general; with appropriate transitions it can compute any computable function (Turing-complete). The examples are intentionally small for clarity.
- Use `--max-steps` to guard against non-halting machines.

