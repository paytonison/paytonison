import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from torch.optim import AdamW
from torchtext.datasets import AG_NEWS
import torch.nn as nn

# Custom Transformer Encoder Layer
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.rnn = nn.LSTM(d_model, d_model, batch_first=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, src):
        # Self-attention layer
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout(src2)
        src = self.norm1(src)

        # LSTM Layer (replacing Feed-Forward network)
        rnn_output, _ = self.rnn(src)
        src = src + self.dropout(rnn_output)
        src = self.norm2(src)

        return src

# Custom Hybrid Model with Embedding Layer
class CustomHybridModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_heads, num_layers, num_classes):
        super(CustomHybridModel, self).__init__()
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Input Linear Layer to transform to hidden_dim
        self.input_linear = nn.Linear(embedding_dim, hidden_dim)

        # Custom Transformer Encoder
        self.custom_encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
            for _ in range(num_layers)
        ])

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output Linear Layer
        self.output_linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, src, tgt):
        # Embedding Layer
        src = self.embedding(src)  # (batch_size, seq_len, embedding_dim)
        tgt = self.embedding(tgt)  # (batch_size, seq_len, embedding_dim)

        # Input Linear Layer
        src = self.input_linear(src)  # (batch_size, seq_len, hidden_dim)
        tgt = self.input_linear(tgt)  # (batch_size, seq_len, hidden_dim)

        # Reshape for multi-head attention (seq_len, batch_size, hidden_dim)
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        # Custom Transformer Encoder
        for layer in self.custom_encoder_layers:
            src = layer(src)

        # Transformer Decoder
        decoded_output = self.transformer_decoder(tgt, src)

        # Reshape back to (batch_size, seq_len, hidden_dim)
        decoded_output = decoded_output.permute(1, 0, 2)

        # Output Linear Layer
        output = self.output_linear(decoded_output)

        # For classification, we can take the output corresponding to the first token
        output = output[:, 0, :]  # (batch_size, num_classes)

        return output

# Define collate_fn to pad sequences and prepare labels
def collate_fn(batch):
    articles, labels = zip(*batch)
    articles = pad_sequence(articles, batch_first=True, padding_value=0).long()  # Convert to long tensor
    labels = torch.tensor(labels).long()  # Convert labels to tensor
    return articles, labels

# Load AG News Dataset
train_iter = AG_NEWS(split='train')

# Extract articles and labels
articles_train = []
labels_train = []

for label, line in train_iter:
    articles_train.append(line)
    labels_train.append(label - 1)  # AG News labels are 1-based, converting to 0-based

# Use a similar process for a validation set
test_iter = AG_NEWS(split='test')
articles_test = []
labels_test = []

for label, line in test_iter:
    articles_test.append(line)
    labels_test.append(label - 1)

class AGNewsDataset(Dataset):
    def __init__(self, articles, labels, tokenizer, max_length):
        self.articles = articles
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        article = self.articles[idx]
        label = self.labels[idx]
        tokens = self.tokenizer.encode(article, max_length=self.max_length, truncation=True, padding='max_length')
        tokens = torch.tensor(tokens)
        return tokens, label

# Prepare Data
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size  # Get vocabulary size from tokenizer
embedding_dim = 128  # You can adjust this

train_dataset = AGNewsDataset(articles_train, labels_train, tokenizer, max_length=128)
train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn, shuffle=True)

# Device Setup
device = torch.device("mps")

# Model Initialization
model = CustomHybridModel(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=256, num_heads=4, num_layers=2, num_classes=4)
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-4)

# Training Loop
for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in train_loader:
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward Pass
        outputs = model(inputs, inputs)  # Using inputs as both src and tgt for simplicity

        # Compute Loss and Backpropagate
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

# Sample Evaluation
model.eval()
with torch.no_grad():
    sample_text = "This movie was great!"
    sample_input = tokenizer.encode(sample_text, max_length=128, truncation=True, padding='max_length')
    sample_input = torch.tensor(sample_input).unsqueeze(0).to(device)  # Shape: (1, seq_len)

    # Forward Pass
    output = model(sample_input, sample_input)
    prediction = torch.argmax(output, dim=-1).item()
    print(f"Predicted Sentiment: {'Positive' if prediction == 1 else 'Negative'}")

