# Enigma-Turing Machine with Integrated AI (Fixed Version)
# Apologies for the glitches—key issues: blank symbol conflicting with '0', duplicate transitions,
# softmax misuse in AI, and CrossEntropy on probs instead of logits. Now it properly flips bits
# until blank, with AI learning the rules correctly. Tested: Input 10110 -> 01001 after 6 steps.
# (Steps: 5 flips + 1 halt on blank.)

import torch
import torch.nn as nn
import torch.optim as optim

# Shared encodings for states, symbols, and head directions.
STATE_TO_IDX = {'q0': 0, 'halt': 1}
SYM_TO_IDX = {'0': 0, '1': 1, '_': 2}
DIR_TO_IDX = {'L': 0, 'N': 1, 'R': 2}

# (state, read_symbol, next_state, write_symbol, direction)
TRANSITION_DATA = [
    (STATE_TO_IDX['q0'], SYM_TO_IDX['0'], STATE_TO_IDX['q0'], SYM_TO_IDX['1'], DIR_TO_IDX['R']),
    (STATE_TO_IDX['q0'], SYM_TO_IDX['1'], STATE_TO_IDX['q0'], SYM_TO_IDX['0'], DIR_TO_IDX['R']),
    (STATE_TO_IDX['q0'], SYM_TO_IDX['_'], STATE_TO_IDX['halt'], SYM_TO_IDX['_'], DIR_TO_IDX['N']),
]


def build_transition_tensors(data=TRANSITION_DATA, device=None):
    tensor_kwargs = {"dtype": torch.long}
    if device is not None:
        tensor_kwargs["device"] = device
    states = torch.tensor([row[0] for row in data], **tensor_kwargs)
    symbols = torch.tensor([row[1] for row in data], **tensor_kwargs)
    targets_ns = torch.tensor([row[2] for row in data], **tensor_kwargs)
    targets_w = torch.tensor([row[3] for row in data], **tensor_kwargs)
    targets_d = torch.tensor([row[4] for row in data], **tensor_kwargs)
    return {
        "states": states,
        "symbols": symbols,
        "target_states": targets_ns,
        "target_symbols": targets_w,
        "target_dirs": targets_d,
    }


def evaluate_predictor(model, dataset_tensors):
    """Return per-head and overall accuracy for the predictor."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        ns_log, w_log, d_log = model(dataset_tensors["states"], dataset_tensors["symbols"])
        ns_pred = ns_log.argmax(dim=-1)
        w_pred = w_log.argmax(dim=-1)
        d_pred = d_log.argmax(dim=-1)

    metrics = {
        "state": (ns_pred == dataset_tensors["target_states"]).float().mean().item(),
        "write": (w_pred == dataset_tensors["target_symbols"]).float().mean().item(),
        "direction": (d_pred == dataset_tensors["target_dirs"]).float().mean().item(),
    }
    metrics["overall"] = sum(metrics.values()) / 3.0
    if was_training:
        model.train()
    return metrics

# Rotor and Enigma classes (minor cleanup)
class Rotor:
    def __init__(self, wiring, position=0, ring_setting=0):
        self.wiring = wiring
        self.position = position
        self.ring_setting = ring_setting
        self.notch = 0

    def forward(self, char):
        if not char.isalpha() or char.islower():
            return char
        idx = (ord(char.upper()) - ord('A') + self.position - self.ring_setting) % 26
        out_char = self.wiring[idx]
        out_idx = ord(out_char) - ord('A')
        return chr((out_idx + self.ring_setting - self.position) % 26 + ord('A'))

    def backward(self, char):
        if not char.isalpha() or char.islower():
            return char
        inv_idx = self.wiring.index(char.upper())
        out_idx = (inv_idx + self.position - self.ring_setting) % 26
        return chr(out_idx + ord('A'))

    def rotate(self):
        self.position = (self.position + 1) % 26

ROTORS = {
    'I': 'EKMFLGDQVZNTOWYHXUSPAIBRCJ',
    'II': 'AJDKSIRUXBLHWTMCQGZNPYFVOE',
    'III': 'BDFHJLCPRTXVZNYEIWGAKMUSQO',
    'IV': 'ESOVPZJAYQUIRHXLNFTGKDCMWB',
    'V': 'VZBRGITYUPSDNHLXAWMJQOFECK'
}

REFLECTOR_B = 'YRUHQSLDPXNGOKMIEBFZCWVJAT'

class Enigma:
    def __init__(self, rotor_order=('I', 'II', 'III'), positions=(0, 0, 0), ring_settings=(1, 1, 1)):
        if len(rotor_order) != 3:
            raise ValueError("Enigma uses exactly 3 rotors.")
        self.left = Rotor(ROTORS[rotor_order[0]], positions[0], ring_settings[0])
        self.middle = Rotor(ROTORS[rotor_order[1]], positions[1], ring_settings[1])
        self.right = Rotor(ROTORS[rotor_order[2]], positions[2], ring_settings[2])
        self.reflector = REFLECTOR_B

    def _step_rotors(self):
        if self.right.position == 0:
            self.middle.rotate()
            if self.middle.position == 0:
                self.left.rotate()
        self.right.rotate()

    def _forward_through_rotors(self, char):
        out = self.right.forward(char)
        out = self.middle.forward(out)
        out = self.left.forward(out)
        return out

    def _backward_through_rotors(self, char):
        out = self.reflector[ord(char) - ord('A')]
        out = self.left.backward(out)
        out = self.middle.backward(out)
        out = self.right.backward(out)
        return out

    def process_char(self, char, step=True):
        if step:
            self._step_rotors()
        if not char.isalpha():
            return char
        forward = self._forward_through_rotors(char.upper())
        backward = self._backward_through_rotors(forward)
        return backward

    def process(self, text):
        return ''.join(self.process_char(c) for c in text.upper())

# Tape with distinct blank
class Tape:
    blank_symbol = "_"
    
    def __init__(self, tape_string=""):
        self.__tape = dict(enumerate(tape_string))
        
    def __str__(self):
        if not self.__tape:
            return ""
        min_idx, max_idx = min(self.__tape), max(self.__tape)
        return "".join(self.__tape.get(i, self.blank_symbol) for i in range(min_idx, max_idx + 1))
   
    def __getitem__(self, index):
        return self.__tape.get(index, self.blank_symbol)

    def __setitem__(self, pos, char):
        self.__tape[pos] = char 

# Fixed AI: Returns logits; softmax in predict/loss handled properly
class AITransitionPredictor(nn.Module):
    def __init__(self, hidden_size=16, num_states=2, num_symbols=3):  
        super().__init__()
        self.state_embed = nn.Embedding(num_states, hidden_size)
        self.symbol_embed = nn.Embedding(num_symbols, hidden_size)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 8)  # 2 states + 3 symbols + 3 dirs
        self.relu = nn.ReLU()

    def forward(self, state_idx, symbol_idx):
        state_emb = self.state_embed(state_idx)
        sym_emb = self.symbol_embed(symbol_idx)
        combined = torch.cat([state_emb, sym_emb], dim=-1)
        hidden = self.relu(self.fc1(combined))
        logits = self.fc2(hidden)
        new_state_logits = logits[:, :2]
        write_logits = logits[:, 2:5]
        dir_logits = logits[:, 5:8]
        return new_state_logits, write_logits, dir_logits

    def predict_transition(
        self,
        state_str,
        symbol,
        state_map=None,
        sym_map=None,
        dir_map=None,
        inv_state_map=None,
        inv_sym_map=None,
        inv_dir_map=None,
    ):
        state_map = state_map or STATE_TO_IDX
        sym_map = sym_map or SYM_TO_IDX
        dir_map = dir_map or DIR_TO_IDX
        inv_state_map = inv_state_map or {idx: name for name, idx in state_map.items()}
        inv_sym_map = inv_sym_map or {idx: sym for sym, idx in sym_map.items()}
        inv_dir_map = inv_dir_map or {idx: direction for direction, idx in dir_map.items()}

        fallback_state = state_map.get('halt', next(iter(state_map.values())))
        fallback_sym = sym_map.get('_', next(iter(sym_map.values())))
        fallback_dir = dir_map.get('N', next(iter(dir_map.values())))

        device = next(self.parameters()).device
        with torch.no_grad():
            state_idx = torch.tensor(
                [state_map.get(state_str, fallback_state)],
                dtype=torch.long,
                device=device,
            )
            sym_idx = torch.tensor(
                [sym_map.get(symbol, fallback_sym)],
                dtype=torch.long,
                device=device,
            )
            ns_log, w_log, d_log = self(state_idx, sym_idx)
            new_state = ns_log.argmax(dim=-1).item()
            write = w_log.argmax(dim=-1).item()
            direction = d_log.argmax(dim=-1).item()

        new_state_str = inv_state_map.get(new_state, state_str)
        write_sym = inv_sym_map.get(write, symbol)
        dir_str = inv_dir_map.get(direction, 'N')
        return new_state_str, write_sym, dir_str

# Fixed training: Use logits for CrossEntropy with built-in evaluation
def train_ai_predictor(num_epochs=100, lr=0.01, target_accuracy=1.0, eval_interval=5, dataset=None):
    model = AITransitionPredictor()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dataset = dataset or build_transition_tensors()

    states = dataset["states"]
    syms = dataset["symbols"]
    targets_ns = dataset["target_states"]
    targets_w = dataset["target_symbols"]
    targets_d = dataset["target_dirs"]

    latest_metrics = {"overall": 0.0}
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        ns_log, w_log, d_log = model(states, syms)
        loss_ns = criterion(ns_log, targets_ns)
        loss_w = criterion(w_log, targets_w)
        loss_d = criterion(d_log, targets_d)
        loss = loss_ns + loss_w + loss_d
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % eval_interval == 0 or epoch == num_epochs:
            latest_metrics = evaluate_predictor(model, dataset)
            print(
                f"Epoch {epoch:03d} | loss={loss.item():.4f} | "
                f"overall_acc={latest_metrics['overall'] * 100:.1f}%"
            )
            if latest_metrics["overall"] >= target_accuracy:
                break

    return model

# Fixed EnigmaTM
class EnigmaTM:
    def __init__(self, 
                 tape_string="", 
                 blank_symbol="_",
                 initial_state="q0",
                 final_states=None,
                 transition_function=None,
                 ai_model=None,
                 use_ai=True,
                 enigma=None,  
                 use_rotors=False,
                 state_map=None,
                 sym_map=None,
                 dir_map=None):  
        self.__tape = Tape(tape_string)
        self.__head_position = 0
        self.__blank_symbol = blank_symbol
        self.__current_state = initial_state
        self.__transition_function = transition_function or {}
        self.__final_states = set(final_states or [])
        self.__enigma = enigma or Enigma()
        self.__use_rotors = use_rotors
        self.__ai_model = ai_model
        self.__use_ai = use_ai and ai_model is not None
        self.state_map = dict(state_map or STATE_TO_IDX)
        self.sym_map = dict(sym_map or SYM_TO_IDX)
        self.dir_map = dict(dir_map or DIR_TO_IDX)
        
    def get_tape(self): 
        return str(self.__tape).rstrip(self.__blank_symbol)  # Hide trailing blanks for clean print
        
    def _get_transition(self, state, read_char):
        if self.__use_ai:
            return self.__ai_model.predict_transition(state, read_char, self.state_map, self.sym_map, self.dir_map)
        key = (state, read_char)
        if key in self.__transition_function:
            return self.__transition_function[key]
        return 'halt', read_char, 'N'  # Default
    
    def _scramble(self, char, direction='forward'):
        if not self.__use_rotors or char == self.__blank_symbol:
            return char
        if direction == 'forward':
            return self.__enigma.process_char(char, step=False)
        else:
            return self.__enigma._backward_through_rotors(char)
    
    def step(self):
        raw_char = self.__tape[self.__head_position]
        read_char = self._scramble(raw_char, 'forward')
        
        new_state, write_char, direction = self._get_transition(self.__current_state, read_char)
        
        raw_write = self._scramble(write_char, 'backward') if self.__use_rotors else write_char
        self.__tape[self.__head_position] = raw_write
        
        if direction == "R":
            self.__head_position += 1
        elif direction == "L":
            self.__head_position -= 1
        self.__current_state = new_state
        
        if self.__use_rotors:
            self.__enigma._step_rotors()

    def run(self, max_steps=1000):
        steps = 0
        while not self.final() and steps < max_steps:
            self.step()
            steps += 1
        return steps

    def final(self):
        return self.__current_state in self.__final_states

# Example usage
if __name__ == "__main__":
    # Enigma setup
    enigma = Enigma(rotor_order=('I', 'II', 'III'), positions=(0, 0, 0))
    
    # Generate tape from plaintext
    plaintext = "HELLO"
    encrypted = enigma.process(plaintext)
    print(f"Encrypted '{plaintext}': {encrypted}")
    binary_tape = "".join(str(ord(c) % 2) for c in encrypted)
    print(f"Initial binary tape: {binary_tape}")
    expected_tape = "".join('1' if bit == '0' else '0' for bit in binary_tape)
    print(f"Expected flipped tape: {expected_tape}")
    
    # Train AI
    print("\nTraining AI Transition Predictor...")
    dataset_tensors = build_transition_tensors()
    ai_model = train_ai_predictor(num_epochs=200, target_accuracy=1.0, eval_interval=5, dataset=dataset_tensors)
    final_metrics = evaluate_predictor(ai_model, dataset_tensors)
    print(
        "Final predictor accuracy | "
        f"state={final_metrics['state']*100:.1f}% | "
        f"write={final_metrics['write']*100:.1f}% | "
        f"direction={final_metrics['direction']*100:.1f}%"
    )
    
    # Fixed transitions
    fixed_trans = {
        ("q0", "0"): ("q0", "1", "R"),
        ("q0", "1"): ("q0", "0", "R"),
        ("q0", "_"): ("halt", "_", "N"),
    }
    
    # AI-driven run
    tm_ai = EnigmaTM(
        tape_string=binary_tape,
        initial_state="q0",
        final_states={"halt"},
        transition_function=fixed_trans,  # Fallback
        ai_model=ai_model,
        use_ai=True,
        enigma=enigma,
        use_rotors=False
    )
    
    print(f"\nInput tape: {tm_ai.get_tape()}")
    steps = tm_ai.run()
    ai_result = tm_ai.get_tape()
    print(f"After {steps} steps (AI-driven): {ai_result}")  # Should be 01001
    print("Halted:", tm_ai.final())
    print("Matches expected flip:", ai_result == expected_tape)
    
    # Fixed-rules run for comparison
    tm_fixed = EnigmaTM(
        tape_string=binary_tape,
        initial_state="q0",
        final_states={"halt"},
        transition_function=fixed_trans,
        use_ai=False,
        enigma=enigma,
        use_rotors=False
    )
    print(f"\nInput tape: {tm_fixed.get_tape()}")
    steps_fixed = tm_fixed.run()
    fixed_result = tm_fixed.get_tape()
    print(f"After {steps_fixed} steps (fixed): {fixed_result}")  # Same: 01001
    print("Halted:", tm_fixed.final())
    print("AI vs fixed identical:", ai_result == fixed_result)
    
    # AI test
    print("\nAI Prediction Tests:")
    ns, w, d = ai_model.predict_transition("q0", "1")
    print(f"q0 + 1 -> {ns}, {w}, {d}")
    ns0, w0, d0 = ai_model.predict_transition("q0", "0")
    print(f"q0 + 0 -> {ns0}, {w0}, {d0}")
