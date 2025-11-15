import json
from collections import Counter
import re

def clean_text(text):
    # Basic text cleaning: lowercasing, removing punctuation, etc.
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    return text

def create_vocabulary_from_json(json_path, vocab_output_path):
    all_text = []
    with open(json_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line.strip())  # Load each line as a separate JSON object
                if 'text' in item:  # Adjust this based on your JSON structure
                    cleaned = clean_text(item['text'])
                    all_text.extend(cleaned.split())
            except json.JSONDecodeError:
                print(f"Skipping line due to JSONDecodeError: {line.strip()}")
    
    # Count word frequencies and create vocabulary
    word_freq = Counter(all_text)
    vocabulary = [word for word, freq in word_freq.items() if freq > 2]  # Include words appearing more than twice
    
    # Save vocabulary to a text file
    with open(vocab_output_path, 'w') as f:
        for word in vocabulary:
            f.write(word + "\n")

# Example usage
json_path = '/Users/paytonison/personal git/rnn/c4-train.00000-of-01024.json'
vocab_output_path = '/Users/paytonison/personal git/rnn/output_vocab.txt'
create_vocabulary_from_json(json_path, vocab_output_path)

