from collections import defaultdict
from typing import Optional, List, Dict, Tuple
import numpy as np
import random

class TrigramPredictor:
    def __init__(self):
        # Store trigram frequencies
        self.trigram_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Store word frequencies for random sampling
        self.word_frequencies: Dict[str, int] = defaultdict(int)
        
    def _get_trigrams(self, words: List[str]) -> List[Tuple[str, str]]:
        """
        Generate trigrams from a list of words.
        
        Args:
            words: List of words
            
        Returns:
            List of (context, target) tuples
        """
        trigrams = []
        for i in range(len(words) - 2):
            context = f"{words[i]} {words[i+1]}"
            target = words[i+2]
            trigrams.append((context, target))
        return trigrams
    
    def learn(self, text: str) -> None:
        """
        Learn trigrams from input text.
        
        Args:
            text: Input text to learn from
        """
        # Tokenize and normalize text
        words = text.lower().split()
        
        # Count word frequencies
        for word in words:
            self.word_frequencies[word] += 1
        
        # Learn trigrams
        for context, target in self._get_trigrams(words):
            self.trigram_counts[context][target] += 1
    
    def predict(self, sequence: str) -> List[Tuple[str, float]]:
        """
        Predict next word given a sequence using trigram probabilities.
        
        Args:
            sequence: Input sequence of words
            
        Returns:
            List of (word, probability) tuples
        """
        words = sequence.lower().split()[-3:]  # Get last three words
        
        if len(words) < 2:
            return []
        
        context = f"{words[-2]} {words[-1]}"
        
        if context not in self.trigram_counts:
            return []
        
        # Calculate probabilities
        counts = self.trigram_counts[context]
        total = sum(counts.values())
        probs = [(word, count/total) for word, count in counts.items()]
        
        return sorted(probs, key=lambda x: x[1], reverse=True)

    def _sample_next_word(self, predictions: List[Tuple[str, float]], temperature: float = 1.0) -> str:
        """
        Sample next word from predictions using temperature.
        
        Args:
            predictions: List of (word, probability) tuples
            temperature: Controls randomness (higher = more random)
        
        Returns:
            Sampled word
        """
        if not predictions:
            # If no predictions, sample from word frequencies
            words = list(self.word_frequencies.keys())
            frequencies = list(self.word_frequencies.values())
            total = sum(frequencies)
            probs = [f/total for f in frequencies]
            return np.random.choice(words, p=probs)
            
        words, probs = zip(*predictions)
        probs = np.array(probs)
        
        # Apply temperature scaling
        probs = np.power(probs, 1/temperature)
        probs = probs / np.sum(probs)
        
        return np.random.choice(words, p=probs)

    def generate_text(self, seed: str, length: int = 50, temperature: float = 1.0) -> str:
        """
        Generate text starting from a seed sequence.
        
        Args:
            seed: Initial sequence of words
            length: Number of words to generate
            temperature: Controls randomness (higher = more random)
        
        Returns:
            Generated text
        """
        if not seed:
            # If no seed, randomly select two words based on frequency
            words = list(self.word_frequencies.keys())
            frequencies = list(self.word_frequencies.values())
            total = sum(frequencies)
            probs = [f/total for f in frequencies]
            current_sequence = list(np.random.choice(words, size=5, p=probs))
        else:
            current_sequence = seed.lower().split()
            # Pad or trim to get exactly two words
            if len(current_sequence) < 2:
                words = list(self.word_frequencies.keys())
                frequencies = list(self.word_frequencies.values())
                total = sum(frequencies)
                probs = [f/total for f in frequencies]
                padding = list(np.random.choice(words, size=4-len(current_sequence), p=probs))
                current_sequence = padding + current_sequence
            current_sequence = current_sequence[-2:]

        generated_text = current_sequence.copy()
        
        for _ in range(length):
            # Get predictions for current sequence
            predictions = self.predict(" ".join(current_sequence))
            
            # Sample next word
            next_word = self._sample_next_word(predictions, temperature)
            
            # Add to generated text and update current sequence
            generated_text.append(next_word)
            current_sequence = current_sequence[1:] + [next_word]
        
        return " ".join(generated_text)

# Example usage
if __name__ == "__main__":
    # Create predictor
    predictor = TrigramPredictor()
    
    # Training text
    with open("test.txt", 'r', encoding='utf-8') as f:
        training_text = ' '.join(f.read().strip().split()[:-1])
    
    # Train the model
    predictor.learn(training_text)
    
    while True:
        print("AI: ", predictor.generate_text(input("USER: "), length=250, temperature=0.7))
