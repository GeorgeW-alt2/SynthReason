from collections import defaultdict
from typing import Optional, List, Dict, Tuple, NamedTuple
import numpy as np
import random
from dataclasses import dataclass

class TrigramPredictor:
    def __init__(self, frequency_weight: float = 0.7):
        self.trigram_counts = defaultdict(lambda: defaultdict(int))
        self.word_frequencies = defaultdict(int)
        self.frequency_weight = frequency_weight
       
        
    def _get_trigrams(self, words: List[str]) -> List[Tuple[str, str]]:
        """Generate trigrams from a list of words."""
        trigrams = []
        for i in range(len(words) - 2):
            context = f"{words[i]} {words[i+1]}"
            target = words[i+2]
            trigrams.append((context, target))
        return trigrams

        
    def predict(self, sequence: str) -> List[Tuple[str, float]]:
        """Predict next word with probability hole analysis."""
        words = sequence.lower().split()
        
        if len(words) < 2:
            return []
        
        context = f"{words[-2]} {words[-1]}"
        
        if context not in self.trigram_counts:
            
            return []
        
        # Calculate base probabilities
        counts = self.trigram_counts[context]
        total_count = sum(counts.values())
        total_freq = sum(self.word_frequencies.values())
        
        # Combined predictions with frequency weighting
        predictions = []
        for word, count in counts.items():
            trigram_prob = count / total_count
            freq_prob = self.word_frequencies[word] / total_freq
            combined_prob = ((1 - self.frequency_weight) * trigram_prob + 
                           self.frequency_weight * freq_prob)
            predictions.append((word, combined_prob))
        
        # Normalize probabilities
        total_prob = sum(p[1] for p in predictions)
        predictions = [(w, p/total_prob) for w, p in predictions]
        
        return sorted(predictions, key=lambda x: x[1], reverse=True)

    def generate_text(self, seed: str, length: int = 50, temperature: float = 1.0) -> Tuple[str, Dict]:
        """Generate text and return hole statistics."""
        current_sequence = seed.lower().split()
        generated_text = current_sequence.copy()
        
        self._pending_words = []
        
        if len(current_sequence) < 2:
            words = list(self.word_frequencies.keys())
            frequencies = list(self.word_frequencies.values())
            total = sum(frequencies)
            probs = [f/total for f in frequencies]
            padding = list(np.random.choice(words, size=max(0, 2-len(current_sequence)), p=probs))
            current_sequence = padding + current_sequence
        
        for _ in range(length):
            predictions = self.predict(" ".join(current_sequence[-2:]))
            next_word = self._sample_next_word(predictions, temperature)
            generated_text.append(next_word)
            current_sequence = current_sequence[-2:] + [next_word]
        
        return " ".join(generated_text)
    def learn(self, text: str) -> None:
        """Learn trigrams from input text."""
        # Tokenize and normalize text
        words = text.lower().split()
        
        # Count word frequencies
        for word in words:
            self.word_frequencies[word] += 1
        
        # Learn trigrams
        for context, target in self._get_trigrams(words):
            self.trigram_counts[context][target] += 1
            
    def _sample_next_word(self, predictions: List[Tuple[str, float]], temperature: float = 1.0) -> str:
        """Sample next word from predictions using temperature."""
            
        # If we have pending words from a phrase, return the next one
        if hasattr(self, '_pending_words') and self._pending_words:
            next_word = self._pending_words.pop(0)
            return next_word
            
        if not predictions:
            # If no predictions, sample from word frequencies
            words = list(self.word_frequencies.keys())
            frequencies = list(self.word_frequencies.values())
            total = sum(frequencies)
            probs = [f/total for f in frequencies]
            return np.random.choice(words, p=probs)
            
        words, probs = zip(*predictions)
        probs = np.array(probs)
        
        # Fix temperature handling - ensure it's positive
        temperature = max(0.1, abs(temperature))
        
        # Apply temperature scaling
        probs = np.power(probs, 1/temperature)
        probs = probs / np.sum(probs)
        
        return np.random.choice(words, p=probs)
# Example usage
if __name__ == "__main__":
    predictor = TrigramPredictor()
    
    # Example training text
    with open(input("Enter filename: "), 'r', encoding='utf-8') as f:
        training_text = f.read().strip()
    
    predictor.learn(training_text)
    while True:
        generated = predictor.generate_text(input("USER: "), length=250)
        print("\nGenerated text:", generated)
