import torch
import torch.nn as nn
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn.functional as F

EPOCHS = 50
KB_MEMORY_UNCOMPRESSED = 1000
setting = 512
file_path = "model.pt"

class TextPreprocessor:
    # [Previous TextPreprocessor implementation remains the same]
    def __init__(self):
        self.word_to_index = None
        self.index_to_word = None
        self.vocab_size = 0
        self.unknown_token = '<UNK>'
        self.pad_token = '<PAD>'

    def preprocess_text(self, text):
        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        tokens = cleaned_text.split()[:KB_MEMORY_UNCOMPRESSED]
        return [word for word in tokens if len(word) > 1 or word in {"i", "a"}]

    def build_vocabulary(self, text_data):
        tokens = self.preprocess_text(text_data)
        word_counts = {word: tokens.count(word) for word in set(tokens)}
        vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        special_tokens = [self.unknown_token, self.pad_token]
        vocab = special_tokens + vocab
        self.word_to_index = {word: i for i, word in enumerate(vocab)}
        self.index_to_word = {i: word for word, i in self.word_to_index.items()}
        self.vocab_size = len(vocab)
        return self.word_to_index, self.vocab_size

    def get_word_index(self, word):
        return self.word_to_index.get(word, self.word_to_index[self.unknown_token])

    def create_sequences(self, text, sequence_length=3):
        tokens = self.preprocess_text(text)
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

class SequentialWeightLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight_update_idx = 0
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x):
        # Update weights sequentially during training
        if self.training:
            with torch.no_grad():
                # Update a subset of weights each forward pass
                update_size = max(1, self.weight.size(1) // 10)  # Update 10% of weights
                start_idx = self.weight_update_idx
                end_idx = min(start_idx + update_size, self.weight.size(1))
                
                # Apply small random perturbation to selected weights
                self.weight[:, start_idx:end_idx] += torch.randn_like(
                    self.weight[:, start_idx:end_idx]) * 0.01
                
                # Update index for next forward pass
                self.weight_update_idx = (end_idx if end_idx < self.weight.size(1) else 0)
        
        return F.linear(x, self.weight, self.bias)

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
        
        # Replace standard linear layers with sequential weight layers
        self.intersection_projection = SequentialWeightLayer(hidden_dim, hidden_dim)
        self.hardsigmoid = nn.Hardsigmoid()
        self.fc = SequentialWeightLayer(hidden_dim, vocab_size)
        
        # Keep track of update cycle
        self.update_cycle = 0
        self.update_frequency = 100  # Update weights every 100 forward passes

    def forward(self, x):
        embedded = self.embedding(x)
        
        lstm_out, _ = self.lstm(embedded)
        hidden = lstm_out[:, -1, :]
        
        # Increment update cycle
        if self.training:
            self.update_cycle = (self.update_cycle + 1) % self.update_frequency
        
        intersection_hidden = self.intersection_projection(hidden)
        activated_hidden = self.hardsigmoid(intersection_hidden)
        output = self.fc(activated_hidden)
        
        masked_output = output
        logits = F.log_softmax(masked_output, dim=-1)
        
        return logits

class TextGeneratorHandler:
    def __init__(self):
        self.model = None
        self.preprocessor = TextPreprocessor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def train(self, text, sequence_length=3, batch_size=32, epochs=EPOCHS, learning_rate=0.001):
        self.preprocessor.build_vocabulary(text)
        
        self.model = TextGenerator(
            self.preprocessor.vocab_size,
            embedding_dim=setting,
            hidden_dim=setting,
            word_to_index=self.preprocessor.word_to_index
        ).to(self.device)
        
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            X, y = self.preprocessor.create_sequences(text, sequence_length)
            
            dataset = torch.utils.data.TensorDataset(X, y)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                batch_X = torch.tensor(np.roll(batch_X, shift=batch_X[-1]))
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
        device = self.device

        words = self.preprocessor.preprocess_text(seed_text)
        while len(words) < 3:
            words.append(self.preprocessor.pad_token)

        sequence = [self.preprocessor.get_word_index(w) for w in words[-3:]]
        sequence = torch.tensor([sequence], dtype=torch.long).to(device)

        generated_words = words[:]
        logits = self.model(sequence)
        logits = logits / temperature
        probabilities = F.softmax(logits, dim=-1)
        
        with torch.no_grad():
            for _ in range(num_words):
                next_word_idx = torch.multinomial(probabilities, 1).item()
                probabilities = torch.tensor(np.roll(probabilities.cpu().numpy(), shift=sequence[-1]), dtype=torch.float32)
                next_word = self.preprocessor.index_to_word.get(next_word_idx, self.preprocessor.unknown_token)

                if next_word not in {self.preprocessor.unknown_token, self.preprocessor.pad_token}:
                    generated_words.append(next_word)

                sequence = torch.tensor([[
                    self.preprocessor.get_word_index(w) 
                    for w in generated_words[-3:]
                ]], dtype=torch.long).to(device)

        return ' '.join(generated_words)

    def save_model(self, file_path):
        if self.model is None:
            raise ValueError("No model to save")
        
        state = {
            'model_state': self.model.state_dict(),
            'preprocessor': self.preprocessor
        }
        torch.save(state, file_path)
        print(f'Model saved to {file_path}')
    
    def load_model(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No model file found at {file_path}")
        
        state = torch.load(file_path)
        self.preprocessor = state['preprocessor']
        
        self.model = TextGenerator(
            self.preprocessor.vocab_size,
            embedding_dim=setting,
            hidden_dim=setting,
            word_to_index=self.preprocessor.word_to_index
        ).to(self.device)
        
        self.model.load_state_dict(state['model_state'])
        self.model.eval()
        print(f'Model loaded from {file_path}')

# Main function remains the same
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
