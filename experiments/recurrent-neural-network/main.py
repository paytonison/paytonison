import random
import torch
import torch.nn as nn
from collections import deque
import torch.nn.functional as F
import torch.optim as optim

# Define the TextCompletionEnv class
class TextCompletionEnv:
    def __init__(self, vocab_file_path):
        # Load vocabulary from the provided file
        self.vocab = self.load_vocab(vocab_file_path)
        
        # Define some example phrases that the agent will interact with
        self.phrases = self.load_vocab(vocab_file_path)

    def load_vocab(self, vocab_file_path):
        # Read the vocabulary file and return a list of words
        with open(vocab_file_path, 'r') as f:
            vocab = [line.strip() for line in f.readlines()]
        
        # Optionally, add a special "<UNK>" token to handle out-of-vocabulary words
        if "<UNK>" not in vocab:
            vocab.insert(0, "<UNK>")
        
        return vocab

    def reset(self):
        # Start with a random phrase
        self.current_phrase = random.choice(self.phrases)
        return self.current_phrase

    def step(self, action):
        chosen_word = self.vocab[action]

        if chosen_word == "<EOS>":
           reward = 1 == done
        # Update state (append the chosen word)
        self.current_phrase += " " + chosen_word
        
        return self.current_phrase, reward, done

# Define the TextQLearningRNN class
class TextQLearningRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(TextQLearningRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x):
        x = self.embedding(x)
        if len(x.shape) == 2:  # Unbatched input (e.g., [seq_length, hidden_size])
            batch_size = 1  # Treat it as a single example batch
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            out, _ = self.rnn(x.unsqueeze(0), h0)  # Add batch dimension to input
            out = out.squeeze(0)  # Remove batch dimension after RNN processing
        else:  # Batched input (e.g., [batch_size, seq_length, hidden_size])
            batch_size = x.size(0)
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            out, _ = self.rnn(x, h0)
        
        # Apply the fully connected layer
        q_values = self.fc(out[-1, :]) if len(out.shape) == 2 else self.fc(out[:, -1, :])
        return q_values

# Define the TextDQLAgent class
class TextDQLAgent:
    def __init__(self, model, env, replay_buffer_size=1000, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.model = model
        self.env = env
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def choose_action(self, state):
        # Get last word from the state and convert to token index
        last_word = state.split()[-1]
        token_index = self.env.vocab.index(last_word) if last_word in self.env.vocab else self.env.vocab.index("<UNK>")
        if random.random() < self.epsilon:
            return random.randint(0, len(self.env.vocab) - 1)  # Explore
        else:
            with torch.no_grad():
                state_tensor = torch.tensor([token_index], dtype=torch.long).unsqueeze(0)
                q_values = self.model(state_tensor)
                return torch.argmax(q_values).item()  # Exploit

    def train(self, num_episodes=1000, batch_size=32):
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0  # Track the total reward for each episode

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)

                # Store in replay buffer
                self.replay_buffer.append((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward

                # Sample random batch from replay buffer and update model
                if len(self.replay_buffer) >= batch_size:
                    self.update_model(batch_size)

            # Decrease epsilon (more exploitation over time)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            print(f"Total Reward for Episode {episode + 1}: {total_reward}")

    def update_model(self, batch_size):
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert states and next_states to tensors
        states = torch.tensor([
            self.env.vocab.index(s.split()[-1]) if s.split()[-1] in self.env.vocab else self.env.vocab.index("<UNK>")
            for s in states
        ], dtype=torch.long).unsqueeze(1)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor([
            self.env.vocab.index(ns.split()[-1]) if ns.split()[-1] in self.env.vocab else self.env.vocab.index("<UNK>")
            for ns in next_states
        ], dtype=torch.long).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.bool)

        # Calculate Q-values and targets
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (~dones)

        # Update model
        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Print loss information
        print(f"Loss: {loss.item()}")

# Set up and run the training
vocab_file_path = '/Users/paytonison/personal git/rnn/output_vocab.txt'  # Make sure this is the correct path
env = TextCompletionEnv(vocab_file_path)
vocab_size = len(env.vocab)
hidden_size = 128
num_layers = 2
model = TextQLearningRNN(vocab_size, hidden_size, num_layers)
agent = TextDQLAgent(model, env)

# Train the model
agent.train(num_episodes=500)

