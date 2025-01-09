import numpy as np
import pickle
import re
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict

# Constants
KB_MEMORY_UNCOMPRESSED = 1000  # Increased memory limit
SEQUENCE_LENGTH = 3
NUM_EPOCHS = 5
GENERATE_LENGTH = 140
TEMPERATURE = 0.7
EMBEDDING_DIM = 256
KNOWLEDGE_DIM = 256
HIDDEN_DIM = 512
BATCH_SIZE = 256
LEARNING_RATE = 0.001

class TextPreprocessor:
    def __init__(self):
        self.word_to_index = None
        self.vocab_size = 0

    def preprocess_text(self, text):
        """Clean and tokenize text."""
        tokens = text.lower().split()[:KB_MEMORY_UNCOMPRESSED]
        return tokens

    def build_vocabulary(self, text_data):
        """Build vocabulary with word frequencies."""
        tokens = self.preprocess_text(text_data)
        word_counts = {word: tokens.count(word) for word in set(tokens)}
        self.word_to_index = {word: i for i, word in enumerate(word_counts)}
        self.vocab_size = len(word_counts)
        return self.word_to_index, self.vocab_size

    def create_sequences(self, text):
        """Convert text into sequences and include inverse sequences."""
        tokens = self.preprocess_text(text)
        encoded = [self.word_to_index[word] for word in tokens if word in self.word_to_index]
        
        sequences = []
        for i in range(SEQUENCE_LENGTH, len(encoded)):
            seq = encoded[i-SEQUENCE_LENGTH:i]
            target = encoded[i]
            if len(seq) == SEQUENCE_LENGTH:
                # Add original sequence
                sequences.append((seq, target))
                # Add inverse sequence
                inverse_seq = seq[i::-1]
                sequences.append((inverse_seq, target))
        return sequences

class TextDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, target = self.sequences[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

class KnowledgeAugmentedTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM, knowledge_dim=KNOWLEDGE_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.kane_embedding = KANEmbedding(vocab_size, embedding_dim, knowledge_dim)
        total_dim = embedding_dim + knowledge_dim
        
        # Transformer encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=total_dim,
            nhead=8,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.fc = nn.Linear(total_dim, vocab_size)

    def forward(self, x):
        # Create attention mask for padding tokens
        mask = None  # Implement masking if needed
        
        # Get embeddings
        embeddings = self.kane_embedding(x)
        
        # Pass through transformer
        transformer_out = self.transformer_encoder(embeddings, mask)
        
        # Take the last sequence element for prediction
        output = self.fc(transformer_out[:, -1, :])
        return output

class KANEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, knowledge_dim):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.knowledge_embedding = nn.Embedding(vocab_size, knowledge_dim)
        
        # Initialize rotation matrix
        rotation_matrix = torch.empty(embedding_dim, embedding_dim)
        nn.init.orthogonal_(rotation_matrix)
        self.rotation_matrix = nn.Parameter(rotation_matrix)
        
        # Position encoding
        self.pos_encoding = self.create_positional_encoding(SEQUENCE_LENGTH, embedding_dim)

    def create_positional_encoding(self, max_seq_len, d_model):
        pos_enc = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pos_enc, requires_grad=False)

    def rotate(self, embedding):
        return torch.matmul(embedding, self.rotation_matrix)

    def forward(self, x):
        batch_size = x.size(0)
        word_embeddings = self.word_embedding(x)
        knowledge_embeddings = self.knowledge_embedding(x)
        
        # Add positional encoding
        word_embeddings = word_embeddings + self.pos_encoding.unsqueeze(0)
        
        # Apply rotation
        rotated_word_embeddings = self.rotate(word_embeddings)
        rotated_knowledge_embeddings = self.rotate(knowledge_embeddings)
        
        return torch.cat((rotated_word_embeddings, rotated_knowledge_embeddings), dim=-1)

class DisjointSet:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])  # Path compression
        return self.parent[item]

    def union(self, set1, set2):
        root1 = self.find(set1)
        root2 = self.find(set2)
        if root1 != root2:
            # Union by rank
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1

    def add(self, item):
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

class ModelHandler:
    def __init__(self):
        self.model = None
        self.preprocessor = TextPreprocessor()

    def train_model(self, data_loader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        optimizer = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(NUM_EPOCHS):
            self.model.train()
            epoch_loss = 0
            progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
            
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            print(f"Epoch {epoch+1}, Average Loss: {epoch_loss/len(data_loader):.4f}")

    def extract_correlations(self, text):
        """Extract common sequences or correlations from the uncompressed text data using a disjoint-set."""
        tokens = self.preprocessor.preprocess_text(text)
        n = len(tokens)
        disjoint_set = DisjointSet()
        
        for i in range(n - SEQUENCE_LENGTH):
            seq = tuple(tokens[i:i + SEQUENCE_LENGTH])
            next_word = tokens[i + SEQUENCE_LENGTH]
            disjoint_set.add(seq)
            disjoint_set.add(next_word)
            disjoint_set.union(seq, next_word)
        
        correlation_groups = defaultdict(list)
        for seq in disjoint_set.parent:
            root = disjoint_set.find(seq)
            correlation_groups[root].append(seq)
        
        return correlation_groups

    def generate_text(self, input_text, max_length=GENERATE_LENGTH):
        """Generate text using the trained model."""
        self.model.eval()

        # Preprocess input text
        input_sequence = self.preprocessor.preprocess_text(input_text)
        indices = [self.preprocessor.word_to_index.get(word, 0) 
                  for word in input_sequence]

        if not indices:
            return "Input text contains no recognizable words."

        # Initialize sequence
        current_sequence = indices[-SEQUENCE_LENGTH:]
        if len(current_sequence) < SEQUENCE_LENGTH:
            padding = [0] * (SEQUENCE_LENGTH - len(current_sequence))
            current_sequence = padding + current_sequence

        generated_indices = []
        input_tensor = torch.tensor([current_sequence], dtype=torch.long)

        # Extract correlations from the input text
        correlations = self.extract_correlations(input_text)
        
        # Generate text
        with torch.no_grad():
            for _ in range(max_length):
                output = self.model(input_tensor)
                probabilities = F.softmax(output / TEMPERATURE, dim=-1).squeeze()

                # Adjust probabilities based on correlations
                current_seq = tuple(current_sequence[-SEQUENCE_LENGTH:])
                for root, sequences in correlations.items():
                    if current_seq in sequences:
                        for seq in sequences:
                            if seq != current_seq:
                                next_word = seq[-1]
                                probabilities[next_word] *= 1.5  # Increase probability for correlated words

                # Normalize the probabilities back to sum to 1
                probabilities /= probabilities.sum()

                # Sample the next word based on adjusted probabilities
                next_word_idx = torch.multinomial(probabilities, 1).item()
                generated_indices.append(next_word_idx)

                # Update input sequence (shift and append the generated word)
                input_tensor = torch.cat((input_tensor[:, 1:], torch.tensor([[next_word_idx]])), dim=1)
                current_sequence.append(next_word_idx)
                current_sequence = current_sequence[-SEQUENCE_LENGTH:]

        # Convert indices back to words
        reverse_vocab = {i: word for word, i in self.preprocessor.word_to_index.items()}
        return ' '.join([reverse_vocab.get(idx, "<UNK>") for idx in generated_indices])

    def save_model(self, model_path='transformer_model.pt', vocab_path='vocab.pkl'):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab_size': self.preprocessor.vocab_size,
        }, model_path)
        
        with open(vocab_path, 'wb') as f:
            pickle.dump(self.preprocessor.word_to_index, f)
        print("Model and vocabulary saved successfully.")

    def load_model(self, model_path='transformer_model.pt', vocab_path='vocab.pkl'):
        checkpoint = torch.load(model_path)
        with open(vocab_path, 'rb') as f:
            self.preprocessor.word_to_index = pickle.load(f)
        self.preprocessor.vocab_size = len(self.preprocessor.word_to_index)

        # Initialize model and load parameters
        self.model = KnowledgeAugmentedTransformer(self.preprocessor.vocab_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("Model and vocabulary loaded successfully.")

def main():
    handler = ModelHandler()
    
    while True:
        print("\nOptions:")
        print("1. Train new model")
        print("2. Load existing model")
        print("3. Generate text")
        print("4. Exit")
        
        choice = input("Select an option (1-4): ")
        
        if choice == '1':
            try:
                with open("test.txt", "r", encoding="utf-8") as f:
                    text = f.read()
                
                # Prepare data
                handler.preprocessor.build_vocabulary(text)
                sequences = handler.preprocessor.create_sequences(text)
                dataset = TextDataset(sequences)
                data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
                
                # Initialize and train model
                handler.model = KnowledgeAugmentedTransformer(handler.preprocessor.vocab_size)
                handler.train_model(data_loader)
                
                # Save model
                handler.save_model()
                
            except Exception as e:
                print(f"Error during training: {str(e)}")
                
        elif choice == '2':
            try:
                handler.load_model()
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                
        elif choice == '3':
            if handler.model is None:
                print("Please train or load a model first.")
                continue
                
            while True:
                user_input = input("Enter text prompt: ")
                generated_text = handler.generate_text(user_input)
                print("\nGenerated text:")
                print(generated_text)
            
        elif choice == '4':
            print("Exiting program.")
            break
            
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()
