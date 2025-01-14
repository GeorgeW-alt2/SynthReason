import torch
import torch.nn as nn
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn.functional as F

EPOCHS = 50
KB_MEMORY_UNCOMPRESSED = 30000
setting = 1280
file_path = "model.pt"

# Concept Locator Class
class ConceptLocator:
    def __init__(self, latent_dim=128):
        # Latent space to represent abstract concepts
        self.latent_space =   torch.nn.Linear(latent_dim, latent_dim) # Randomly initialized
        self.mapping_function = torch.randn(latent_dim, latent_dim) # Mapping for inference

    def encode_concept(self, input_data):
        """
        Encode input into latent space.
        """
        if isinstance(input_data, (list, np.ndarray, torch.Tensor)):
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
        else:
            raise ValueError("Unsupported input type for encoding.")

        # Normalize and map input to latent space
        normalized_input = F.normalize(input_tensor, dim=0)
        latent_representation = self.mapping_function(normalized_input)
        return latent_representation

    def find_related_concepts(self, query_vector, threshold=0.2):
        """
        Search for related concepts in the latent space.
        """
        query_vector = F.normalize(query_vector, dim=-1)
        similarity_scores = torch.mm(self.latent_space.t()[-1],query_vector.unsqueeze(0) )
        
        # Identify indices of concepts above the threshold
        related_indices = torch.where(query_vector > F.normalize(similarity_scores, dim=-1))[0]
        return related_indices
        
# Text Preprocessor Class
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

# Text Generator Class
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
        self.hardsigmoid = nn.Hardsigmoid()  # Add Hardshrink activation function
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        
        lstm_out, _ = self.lstm(embedded)
        hidden = lstm_out[:, -1, :]
        
        intersection_hidden = self.intersection_projection(hidden)
        activated_hidden = self.hardsigmoid(intersection_hidden)  # Apply Hardshrink activation
        output = self.fc(activated_hidden)
        
        # Apply mask and get probability distribution
        masked_output = output
        logits = F.log_softmax(masked_output, dim=-1)
        
        return logits

# Text Generator Handler Class
class TextGeneratorHandler:
    def __init__(self):
        self.model = None
        self.preprocessor = TextPreprocessor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.concept_locator = ConceptLocator()  # Initialize ConceptLocator
    
    def train(self, text, sequence_length=3, batch_size=32, epochs=EPOCHS, learning_rate=0.001):
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
            embedding_dim=setting,
            hidden_dim=setting,
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
                output = output.div(temperature)  # Apply temperature for randomness
                probabilities = torch.exp(output)  # Convert log probabilities back to probabilities
                
                # Sample two words and choose one from them
                next_word_idx = torch.multinomial(probabilities[0], 2)  # Sample 2 words
                
                # Extract the indices of the two sampled words
                idx_1 = next_word_idx[0].item()
                idx_2 = next_word_idx[1].item()
                
                # Get the probabilities for these two words
                prob_word_1 = probabilities[0, idx_1].item()
                prob_word_2 = probabilities[0, idx_2].item()
                
                # Use np.max to get the index of the word with the highest probability
                max_prob_idx = np.argmax([prob_word_1, prob_word_2])
                
                # Choose the index of the word with the highest probability
                next_word_idx = next_word_idx[max_prob_idx].item()

                # Convert index to word using the preprocessor
                next_word = self.preprocessor.index_to_word[next_word_idx]
                
                # Avoid adding unknown or padding tokens
                if next_word not in {self.preprocessor.unknown_token, self.preprocessor.pad_token}:
                    words.append(next_word)
                
                # Update the sequence with the latest word
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
            embedding_dim=setting,
            hidden_dim=setting,
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
                print(f"Error during text generation: {str(e)}")
        
        elif choice == '3':
            save_file = input("Enter filename to save the model: ")
            handler.save_model(save_file)
        
        elif choice == '4':
            load_file = input("Enter filename to load the model: ")
            handler.load_model(load_file)
        
        elif choice == '5':
            break

if __name__ == "__main__":
    main()
