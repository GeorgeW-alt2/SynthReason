import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, Counter

class GPUTextGenerator:
    def __init__(self, context_window=3, traceback_depth=5, batch_size=1000, use_gpu=True):
        """
        Text generator with optional GPU acceleration.
        
        Args:
            context_window (int): Number of previous words to consider
            traceback_depth (int): Depth of pattern tracking
            batch_size (int): Batch size for processing
            use_gpu (bool): Whether to use GPU if available
        """
        self.context_window = context_window
        self.traceback_depth = traceback_depth
        self.batch_size = batch_size
        
        # GPU setup
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Vocabulary management
        self.word_to_idx = {}
        self.idx_to_word = []
        
        # Transition matrix will be a PyTorch tensor
        self.transition_matrix = None
        self.pattern_store = defaultdict(float)
    
    def _get_word_index(self, word, train=False):
        """Get or create word index."""
        if word not in self.word_to_idx:
            if train:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word.append(word)
                return idx
            return None
        return self.word_to_idx[word]

    def train(self, text):
        """Train model with GPU-accelerated approach."""
        words = text.split()
        print(f"Vocabulary building...")
        
        # Build vocabulary more efficiently
        word_counts = Counter(words)
        # Only keep words that appear more than once
        for word, count in word_counts.items():
            if count > 1:
                self._get_word_index(word, train=True)
        
        vocab_size = len(self.word_to_idx)
        print(f"Vocabulary size: {vocab_size}")
        
        # Convert words to indices, skipping unknown words
        word_indices = []
        for word in words:
            idx = self.word_to_idx.get(word)
            if idx is not None:
                word_indices.append(idx)
        
        print("Building transition matrix...")
        # Create a PyTorch tensor for transition matrix
        self.transition_matrix = torch.zeros((vocab_size, vocab_size), 
                                             dtype=torch.float32, 
                                             device=self.device)
        
        # Count transitions
        for i in range(1, len(word_indices)):
            prev_idx = word_indices[i-1]
            curr_idx = word_indices[i]
            self.transition_matrix[prev_idx, curr_idx] += 1
        
        # Normalize rows (add small epsilon to avoid division by zero)
        row_sums = self.transition_matrix.sum(dim=-1, keepdim=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self.transition_matrix = self.transition_matrix / row_sums
        
        # Move to device
        self.transition_matrix = self.transition_matrix.to(self.device)
        
        print("Processing patterns...")
        # Store only frequent patterns
        pattern_counts = Counter()
        for i in range(2, len(word_indices)):
            pattern = tuple(word_indices[i-1:i+1])
            pattern_counts[pattern] += 1
        
        # Keep only patterns that appear multiple times
        for pattern, count in pattern_counts.items():
            if count > 1:
                self.pattern_store[pattern] = count
        
        print("Training complete!")

    def _get_next_word_probabilities(self, sequence):
        """Calculate probabilities for next word using GPU."""
        if not sequence:
            # Uniform distribution
            return torch.ones(len(self.idx_to_word), device=self.device) / len(self.idx_to_word)
        
        # Get transition probabilities from last word
        last_idx = sequence[-1]
        probs = self.transition_matrix[last_idx]
        
        # Add pattern influence
        if len(sequence) >= 2:
            pattern_prefix = tuple(sequence[-2:])
            for word_idx in range(len(self.idx_to_word)):
                pattern = pattern_prefix + (word_idx,)
                if pattern in self.pattern_store:
                    probs[word_idx] += 0.1 * self.pattern_store[pattern]
        
        # Normalize and ensure no zero probabilities
        probs = probs + 1e-10
        return probs / probs.sum()

    def generate_text(self, seed, length=100):
        """Generate text using GPU-accelerated processing."""
        sequence = []
        
        # Convert seed to indices
        for word in seed.split():
            idx = self._get_word_index(word)
            if idx is not None:
                sequence.append(idx)
        
        if not sequence:
            return "Error: Seed words not found in training data"
        
        while len(sequence) < length:
            # Use GPU for probability calculation
            with torch.no_grad():
                probs = self._get_next_word_probabilities(sequence)
                # Move probabilities to CPU for sampling
                probs_cpu = probs.cpu().numpy()
                
                # Sample next word
                next_idx = torch.tensor(
                    random.choices(
                        range(len(self.idx_to_word)), 
                        weights=probs_cpu
                    )[0], 
                    device=self.device
                )
                sequence.append(next_idx.item())
        
        return " ".join(self.idx_to_word[idx] for idx in sequence)

def main():
    # Create generator with GPU support
    generator = GPUTextGenerator(context_window=5, 
                                 traceback_depth=15, 
                                 batch_size=100, 
                                 use_gpu=True)
    
    filename = input("Enter training file path: ")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            text = ' '.join(f.read().split()[:-1])
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    print("\nTraining model...")
    generator.train(text)
    
    while True:
        try:
            seed = input("\nEnter seed text (or 'quit' to exit): ")
            if seed.lower() == 'quit':
                break
                
            result = generator.generate_text(seed, 250)
            print("\nGenerated text:")
            print(result)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    main()