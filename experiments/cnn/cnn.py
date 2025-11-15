import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# Dummy dataset (to simulate a text dataset)
class TextDataset(Dataset):
    def __init__(self, data, targets, vocab_size, seq_length):
        self.data = data
        self.targets = targets
        self.vocab_size = vocab_size
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return a tensor of word indices and corresponding target
        return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(self.targets[idx], dtype=torch.long)

# Example text data (sequences of word indices) and targets
vocab_size = 5000  # Size of the vocabulary
seq_length = 100   # Length of each text sequence
num_classes = 2    # Binary classification (e.g., positive/negative sentiment)

# Simulated data (replace with actual text dataset)
train_data = [[1, 2, 3, 4, 5] * 20 for _ in range(1000)]  # Dummy sequences of word indices
train_targets = [1 if i % 2 == 0 else 0 for i in range(1000)]  # Binary labels

train_dataset = TextDataset(train_data, train_targets, vocab_size, seq_length)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

test_data = [[1, 2, 3, 4, 5] * 20 for _ in range(1000)]  # Dummy sequences of word indices
test_targets = [1 if i % 2 == 0 else 0 for i in range(1000)]  # Binary labels

test_dataset = TextDataset(train_data, train_targets, vocab_size, seq_length)
test_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)


# Define the CNN architecture for text classification
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes):
        super(TextCNN, self).__init__()
        # Embedding layer to convert word indices into dense vectors
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(3, embed_size), stride=1, padding=(1, 0))  # 3-gram filters
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(4, embed_size), stride=1, padding=(2, 0))  # 4-gram filters
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(5, embed_size), stride=1, padding=(2, 0))  # 5-gram filters
        
        # Pooling layers (to downsample the convolutions)
        self.pool = nn.MaxPool1d(kernel_size=seq_length - 3 + 1)  # Pooling to reduce dimensionality
        
        # Fully connected layers
        self.fc1 = nn.Linear(300, 128)  # 100 filters from 3 conv layers, flattened
        self.fc2 = nn.Linear(128, num_classes)  # Output layer
    
    def forward(self, x):
        # Convert input word indices to embeddings
        x = self.embedding(x)  # Shape: (batch_size, seq_length, embed_size)
        x = x.unsqueeze(1)  # Add channel dimension (batch_size, 1, seq_length, embed_size)
        
        # Apply convolutions + ReLU
        conv1 = F.relu(self.conv1(x)).squeeze(3)  # Remove last dimension after convolution
        conv2 = F.relu(self.conv2(x)).squeeze(3)
        conv3 = F.relu(self.conv3(x)).squeeze(3)
        
        # Apply pooling
        pooled1 = F.max_pool1d(conv1, conv1.size(2)).squeeze(2)
        pooled2 = F.max_pool1d(conv2, conv2.size(2)).squeeze(2)
        pooled3 = F.max_pool1d(conv3, conv3.size(2)).squeeze(2)
        
        # Concatenate pooled features from all convolutional layers
        out = torch.cat((pooled1, pooled2, pooled3), dim=1)  # Shape: (batch_size, 300)
        
        # Pass through fully connected layers
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)  # No activation (CrossEntropyLoss will handle it)
        
        return out

# Hyperparameters
embed_size = 128  # Size of word embeddings
num_classes = 2   # Binary classification (positive/negative sentiment)

# Instantiate the model
model = TextCNN(vocab_size, embed_size, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0459)

num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Processing batch {batch_idx+1}/{len(train_loader)}")
                
        # Forward pass
        output = model(data)

        loss = criterion(output, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss for every 100 batches
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Step {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')


model.eval()

# Initialize variables to track performance
correct = 0
total = 0

# Disable gradient calculation for evaluation
with torch.no_grad():
    for data, target in test_loader:
        # Forward pass: Get predictions
        outputs = model(data)
        
        # Get the predicted class (the one with the highest score)
        _, predicted = torch.max(outputs.data, 1)
        
        # Update total number of samples and correctly predicted samples
        total += target.size(0)
        correct += (predicted == target).sum().item()

# Calculate accuracy
accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')
