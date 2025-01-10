import torch
import torch.nn as nn
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader
import os

KB_MEMORY_UNCOMPRESSED = 10000
file_path = "model.pt"
class TextPreprocessor:
    def __init__(self):
        self.word_to_index = None
        self.index_to_word = None
        self.vocab_size = 0
        self.unknown_token = '<UNK>'
        self.pad_token = '<PAD>'

    def preprocess_text(self, text):
        """Clean and tokenize text."""
        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        tokens = cleaned_text.split()[:KB_MEMORY_UNCOMPRESSED]
        return [word for word in tokens if len(word) > 1 or word in {"i", "a"}]

    def build_vocabulary(self, text_data):
        """Build vocabulary with word frequencies."""
        tokens = self.preprocess_text(text_data)
        word_counts = {word: tokens.count(word) for word in set(tokens)}
        vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        
        # Add special tokens
        special_tokens = [self.unknown_token, self.pad_token]
        vocab = special_tokens + vocab
        
        self.word_to_index = {word: i for i, word in enumerate(vocab)}
        self.index_to_word = {i: word for word, i in self.word_to_index.items()}
        self.vocab_size = len(vocab)
        return self.word_to_index, self.vocab_size

    def get_word_index(self, word):
        """Get index for a word, return unknown token index if word not in vocabulary."""
        return self.word_to_index.get(word, self.word_to_index[self.unknown_token])

    def create_sequences(self, text, sequence_length=3):
        """Convert text into sequences for training."""
        tokens = self.preprocess_text(text)
        encoded = [self.get_word_index(word) for word in tokens]
        
        X, y = [], []
        for i in range(len(encoded) - sequence_length):
            X.append(encoded[i:i + sequence_length])
            y.append(encoded[i + sequence_length])
            
        return torch.LongTensor(X), torch.LongTensor(y)

class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512, hidden_size=1024, num_layers=2):
        super(TextGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out[:, -1, :])
        return output

class TextGeneratorHandler:
    def __init__(self):
        self.model = None
        self.preprocessor = TextPreprocessor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def train(self, text, sequence_length=3, batch_size=32, epochs=50, learning_rate=0.001):
        # Build vocabulary
        self.preprocessor.build_vocabulary(text)
        
        # Prepare sequences
        X, y = self.preprocessor.create_sequences(text, sequence_length)
        
        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(X, y)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        self.model = TextGenerator(self.preprocessor.vocab_size).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(data_loader):.4f}')
                
    def generate_text(self, seed_text, num_words=50, temperature=0.7):
        self.model.eval()
        words = self.preprocessor.preprocess_text(seed_text)
        
        generated_words = words  # Use last 3 words as context
        
        with torch.no_grad():
            for _ in range(num_words):
                # Convert context to tensor, handling unknown words
                sequence = [self.preprocessor.get_word_index(w) for w in generated_words]
                sequence = torch.LongTensor([sequence]).to(self.device)
                
                # Get predictions
                output = self.model(sequence)
                
                # Apply temperature sampling
                output = output.div(temperature)
                probabilities = torch.nn.functional.softmax(output, dim=-1)
                next_word_idx = torch.multinomial(probabilities[0], 1).item()
                
                # Add the generated word
                next_word = self.preprocessor.index_to_word[next_word_idx]
                if next_word not in {self.preprocessor.unknown_token, self.preprocessor.pad_token}:
                    generated_words.append(next_word)
        
        return ' '.join(generated_words)
    
    def save_model(self, file_path):
        """Save the model to a file."""
        torch.save(self.model.state_dict(), file_path)
        print(f'Model saved to {file_path}')
    
    def load_model(self, file_path):
        """Load the model from a file."""
        self.model = TextGenerator(self.preprocessor.vocab_size).to(self.device)
        self.model.load_state_dict(torch.load(file_path))
        self.model.eval()
        print(f'Model loaded from {file_path}')

def main():
    handler = TextGeneratorHandler()
    while True:
        print("\n1. Train Model")
        print("2. Generate Text")
        print("3. Save Model")
        print("4. Load Model")
        print("5. Exit")
        choice = input("Enter choice: ")
        
        if choice == '1':
            text_file = input("Enter KB filename for training: ")
            with open(text_file, "r", encoding="utf-8") as f:
                text = ' '.join(f.read().split())[:KB_MEMORY_UNCOMPRESSED]
            epochs = 50
            handler.train(text, epochs=epochs)
        
        elif choice == '2':
            num_words = 120
            while True:
                seed_text = input("Enter seed text for generating text: ")
                generated_text = handler.generate_text(seed_text, num_words=num_words)
                print(f"\nSeed text: {seed_text}")
                print(f"Generated text: {generated_text}")
        
        elif choice == '3':
            handler.save_model(file_path)
        
        elif choice == '4':
            handler.load_model(file_path)
        
        elif choice == '5':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
