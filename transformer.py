import torch
import torch.nn as nn
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn.functional as F
EPOCHS = 50
KB_MEMORY_UNCOMPRESSED = 3000
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
        #cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        tokens = text.split()[:KB_MEMORY_UNCOMPRESSED]
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
        
        # Handle unknown tokens
        indices = [self.get_word_index(word) for word in tokens]
        
        X = []
        y = []
        
        num_sequences = len(indices) - sequence_length
        if num_sequences <= 0:
            raise ValueError("Text is too short for the given sequence length")
        
        for i in range(num_sequences):
            sequence = indices[i:i + sequence_length]
            X.append(sequence)
            y.append(indices[i + sequence_length])
            
        return torch.LongTensor(X), torch.LongTensor(y)

class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, word_to_index, num_layers=1):
        super(TextGenerator, self).__init__()
        self.word_to_index = word_to_index
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.intersection_projection = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def compute_intersection(self, x, embedded):
        batch_size = x.size(0)
        stop_words = [
            'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't",
            'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't",
            'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during',
            'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he',
            "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's",
            'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's",
            'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or',
            'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll",
            "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them',
            'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this',
            'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're",
            "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who',
            "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're",
            "you've", 'your', 'yours', 'yourself', 'yourselves'
        ]

        stop_word_indices = torch.tensor([self.word_to_index.get(word, -1) for word in stop_words 
                                        if word in self.word_to_index], device=x.device)
        
        masks = torch.ones((batch_size, self.vocab_size), device=x.device)
        for i, sequence in enumerate(x):
            unique_tokens = torch.unique(sequence)
            masks[i].index_fill_(0, unique_tokens, 1)
            masks[i].index_fill_(0, stop_word_indices, 1)
        
        return -1 / np.exp(masks)  # Invert the mask so 1s indicate tokens to keep

    def forward(self, x):
        embedded = self.embedding(x)
        intersection_mask = self.compute_intersection(x, embedded)
        
        lstm_out, _ = self.lstm(embedded)
        hidden = lstm_out[:, -1, :]
        
        intersection_hidden = self.intersection_projection(hidden)
        output = self.fc(intersection_hidden)
        
        # Apply mask and get probability distribution
        masked_output = output * intersection_mask
        logits = F.log_softmax(masked_output, dim=-1)
        
        return logits

class TextGeneratorHandler:
    def __init__(self):
        self.model = None
        self.preprocessor = TextPreprocessor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def train(self, text, sequence_length=3, batch_size=32, epochs=EPOCHS, learning_rate=120):
        # Build vocabulary
        self.preprocessor.build_vocabulary(text)
        
        # Prepare sequences
        X, y = self.preprocessor.create_sequences(text, sequence_length)
        
        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(X, y)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        self.model = TextGenerator(
            self.preprocessor.vocab_size,
            embedding_dim=128,
            hidden_dim=128,
            word_to_index=self.preprocessor.word_to_index
        ).to(self.device)
        
        criterion = nn.NLLLoss()  # Use NLLLoss since we're using log_softmax in forward
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
        words = self.preprocessor.preprocess_text(seed_text)  # Keep last 3 words as context
        sequence = [self.preprocessor.get_word_index(w) for w in words]
        sequence = torch.LongTensor([sequence]).to(self.device)
        with torch.no_grad():
            for _ in range(num_words):

                
                output = self.model(sequence)
                output = output.div(temperature)
                probabilities = torch.exp(output)  # Convert log probabilities back to probabilities
                next_word_idx = torch.multinomial(probabilities[0], 1).item()
                
                next_word = self.preprocessor.index_to_word[next_word_idx]
                if next_word not in {self.preprocessor.unknown_token, self.preprocessor.pad_token}:
                    words.append(next_word)
                sequence = [self.preprocessor.get_word_index(w) for w in words[-3:]]
                sequence = torch.LongTensor([sequence]).to(self.device)
        return ' '.join(words)
    
    def save_model(self, file_path):
        """Save the model and preprocessor to a file."""
        if self.model is None:
            raise ValueError("No model to save")
        
        state = {
            'model_state': self.model.state_dict(),
            'preprocessor': self.preprocessor
        }
        torch.save(state, file_path)
        print(f'Model saved to {file_path}')
    
    def load_model(self, file_path):
        """Load the model and preprocessor from a file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No model file found at {file_path}")
        
        state = torch.load(file_path)
        self.preprocessor = state['preprocessor']
        
        self.model = TextGenerator(
            self.preprocessor.vocab_size,
            embedding_dim=128,
            hidden_dim=128,
            word_to_index=self.preprocessor.word_to_index
        ).to(self.device)
        
        self.model.load_state_dict(state['model_state'])
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
            try:
                with open(text_file, "r", encoding="utf-8") as f:
                    text = ' '.join(f.read().split())[:KB_MEMORY_UNCOMPRESSED]
                epochs = EPOCHS
                handler.train(text, epochs=epochs)
            except FileNotFoundError:
                print(f"Error: File '{text_file}' not found.")
            except Exception as e:
                print(f"Error during training: {str(e)}")
        
        elif choice == '2':
            if handler.model is None:
                print("Please train or load a model first.")
                continue
                
            num_words = 120
            try:
                while True:
                    seed_text = input("Enter seed text for generating text: ")

                    generated_text = handler.generate_text(seed_text, num_words=num_words)
                    print(f"\nSeed text: {seed_text}")
                    print(f"Generated text: {generated_text}")
            except Exception as e:
                print(f"Error generating text: {str(e)}")
        
        elif choice == '3':
            try:
                handler.save_model(file_path)
            except Exception as e:
                print(f"Error saving model: {str(e)}")
        
        elif choice == '4':
            try:
                handler.load_model(file_path)
            except Exception as e:
                print(f"Error loading model: {str(e)}")
        
        elif choice == '5':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()