#Transformer 3.0 - George W - 29,NOV,2024
import numpy as np
import pickle
import re
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Constants
KB_MEMORY_UNCOMPRESSED = 4000
n = 4  # Use quadgrams for training
num_epochs = 20
generate_length = 140
temperature = 0.7

# Preprocessing and Vocabulary
def preprocess_text(text):
    """Clean and tokenize text."""
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.lower().split()[:KB_MEMORY_UNCOMPRESSED]
    return [word for word in tokens if len(word) > 1 or word in {"i", "a"}]

def build_vocabulary(text_data):
    """Build vocabulary with word frequencies."""
    tokens = preprocess_text(text_data)
    word_counts = {word: tokens.count(word) for word in set(tokens)}
    vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    word_to_index = {word: i for i, word in enumerate(vocab)}
    return word_to_index, len(vocab)

def create_sequences(word_to_index, text, sequence_length):
    """Convert text into sequences."""
    encoded = [word_to_index[word] for word in text if word in word_to_index]
    return [(encoded[i-sequence_length:i], encoded[i]) for i in range(sequence_length, len(encoded))]

# Dataset Class
class TextDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, target = self.sequences[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

# Knowledge-Augmented LSTM Model
class KANEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, knowledge_dim):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.knowledge_embedding = nn.Embedding(vocab_size, knowledge_dim)

    def forward(self, x):
        return torch.cat((self.word_embedding(x), self.knowledge_embedding(x)), dim=-1)

class KnowledgeAugmentedLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=150, knowledge_dim=100, rnn_units=386, dropout_rate=0.3):
        super().__init__()
        self.embedding = KANEmbedding(vocab_size, embedding_dim, knowledge_dim)
        self.lstm = nn.LSTM(embedding_dim + knowledge_dim, rnn_units, batch_first=True)
        self.fc = nn.Linear(rnn_units, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        return self.fc(self.dropout(lstm_out[:, -1, :]))

# Training Function
def train_model(model, data_loader, num_epochs, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        for inputs, targets in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

# Save and Load Functions
def save_model_and_vocab(model, word_to_index):
    torch.save(model.state_dict(), 'knowledge_augmented_lstm.mdl')
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(word_to_index, f)
    print("Model and vocabulary saved.")

def load_model_and_vocab(vocab_path='vocab.pkl', model_path='knowledge_augmented_lstm.mdl'):
    with open(vocab_path, 'rb') as f:
        word_to_index = pickle.load(f)
    vocab_size = len(word_to_index)
    model = KnowledgeAugmentedLSTM(vocab_size)
    model.load_state_dict(torch.load(model_path, weights_only= True))
    model.eval()
    print("Model and vocabulary loaded.")
    return model, word_to_index

# Text Generation
def generate_text(model, word_to_index, input_text, sequence_length, generate_length, temperature):
    input_sequence = preprocess_text(input_text)
    indices = [word_to_index.get(word, -1) for word in input_sequence if word in word_to_index]

    if not indices:
        return "Input text contains no recognizable words."

    generated_text = []
    input_tensor = torch.tensor(indices[-sequence_length:], dtype=torch.long).unsqueeze(0)

    for _ in range(generate_length):
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output / temperature, dim=-1).squeeze()
            next_word_idx = torch.multinomial(probabilities, 1).item()
            generated_text.append(next_word_idx)

            input_tensor = torch.cat((input_tensor[:, 1:], torch.tensor([[next_word_idx]])), dim=1)

    reverse_vocab = {i: word for word, i in word_to_index.items()}
    return ' '.join([reverse_vocab.get(idx, "<UNK>") for idx in generated_text])

# Main Function
def main():
    choice = input("Do you want to (1) train or (2) load a model: ")

    if choice == '1':
        with open("test.txt", encoding="utf-8") as f:
            text = f.read()

        word_to_index, vocab_size = build_vocabulary(text)
        sequences = create_sequences(word_to_index, preprocess_text(text), sequence_length=4)
        dataset = TextDataset(sequences)
        data_loader = DataLoader(dataset, batch_size=256, shuffle=True)

        model = KnowledgeAugmentedLSTM(vocab_size)
        train_model(model, data_loader, num_epochs=num_epochs)
        save_model_and_vocab(model, word_to_index)
    elif choice == '2':
        model, word_to_index = load_model_and_vocab()
    else:
        print("Invalid option.")
        return

    while True:
        user_input = input("User: ")
        print("AI:", generate_text(model, word_to_index, user_input, sequence_length=4, generate_length=generate_length, temperature=temperature))

if __name__ == "__main__":
    main()
