import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer, util

# Load models
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Set device (use MPS for Apple Silicon, or switch to CUDA/CPU as needed)
device = torch.device("mps")
model.to(device)

# Define Q-Network
class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Prepare input
def prepare_input(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    return inputs.input_ids.to(device), inputs.attention_mask.to(device)

# Reward calculation with normalization
def calculate_reward(generated_text, reference_texts):
    semantic_coherence_reward = 0.0
    stylistic_matching_reward = 0.0
    syntactic_variety_reward = 0.0
    philosophical_tone_reward = 0.0
    diversity_penalty = 0.0

    # Semantic Coherence
    generated_embedding = semantic_model.encode(generated_text, convert_to_tensor=True)
    reference_embedding = semantic_model.encode(reference_texts, convert_to_tensor=True)
    semantic_similarity = util.pytorch_cos_sim(generated_embedding, reference_embedding).item()

    if semantic_similarity > 0.89:
        semantic_coherence_reward = 1.0

    # Stylistic Matching
    if any(phrase in generated_text.lower() for phrase in [
        "will to power", "eternal return", "overman", "beyond good and evil", "god is dead", "ubermensch", "herd instinct", "slave morality", "master morality", "eternal will", "eternal recurrence"]):
        stylistic_matching_reward = 0.5

    # Syntactic Variety
    if len(generated_text.split()) > 15 and (';' in generated_text or '-—' in generated_text):
        syntactic_variety_reward = 1.0

    # Philosophical Tone
    philosophical_keywords = ["truth", "power", "moral", "chaos", "being", "essence", "doubt", "freedom",
                              "apollonian", "dionysian", "eternity", "nihilism", "eternal", "will", "abyss", "void", "existence", "consciousness"]
    for word in philosophical_keywords:
        if word in generated_text.lower():
            philosophical_tone_reward += 0.2

    # Diversity Penalty
    token_list = generated_text.split()
    unique_token_ratio = len(set(token_list)) / len(token_list)
    if unique_token_ratio < 0.4:
        diversity_penalty = -0.2 * (0.8 - unique_token_ratio)
    diversity_penalty = max(diversity_penalty, -0.5)

    # Total Reward
    total_reward = (semantic_coherence_reward +
                    stylistic_matching_reward +
                    syntactic_variety_reward +
                    philosophical_tone_reward +
                    diversity_penalty)

    print(f"Reward Breakdown:")
    print(f"  Semantic Coherence: {semantic_coherence_reward}")
    print(f"  Stylistic Matching: {stylistic_matching_reward}")
    print(f"  Syntactic Variety: {syntactic_variety_reward}")
    print(f"  Philosophical Tone: {philosophical_tone_reward}")
    print(f"  Diversity Penalty: {diversity_penalty}")
    print(f"  Total Reward: {total_reward}")

    return total_reward

# Normalize rewards for better alignment
def normalize_reward(reward, min_value=0, max_value=5):
    return (reward - min_value) / (max_value - min_value)

# Initialize Q-Network and optimizer
input_dim = 768
hidden_dim = 512
output_dim = tokenizer.vocab_size
q_network = QNetwork(input_dim, hidden_dim, output_dim).to(device)
optimizer = optim.SGD(q_network.parameters(), lr=0.01)
loss_fn = nn.HuberLoss(delta=1.0)

# Training Loop
epochs = 33
for epoch in range(epochs):
    model.train()
    reference_text = (
        'Supposing that Truth is a woman—what then? Is there not ground for suspecting that all philosophers...'
    )

    # Generate initial text
    input_ids, attention_mask = prepare_input(
        "In the vast silence of the universe, where light and shadow dance eternally, there lies a truth..."
    )
    # Generate initial text from the transformer model with repetition penalty
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=200,        # Allow for longer generation
        num_return_sequences=1,    # Generate one output at a time
        pad_token_id=model.config.pad_token_id,
        do_sample=True,            # Enable sampling for variety
        top_k=66,                  # Explore top-40 tokens
        top_p=0.75,                # Use nucleus sampling for diversity
        temperature=1.2,           # Slightly increase temperature for stylistic variation
        no_repeat_ngram_size=3,    # Prevents repeating phrases of 3 or more tokens
        repetition_penalty=1.2)    # Apply repetition penalty to avoid loops

    # Decode and calculate reward
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated Text: {generated_text}")
    reward = calculate_reward(generated_text, reference_text)
    normalized_reward = normalize_reward(reward)

    # Pass embeddings through Q-Network
    inputs_embeds = model.transformer.wte(input_ids)
    q_values = q_network(inputs_embeds.view(-1, input_dim))

    # Compute loss
    target = torch.tensor([normalized_reward]).to(device)
    loss = loss_fn(q_values, target)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Reward: {reward}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
