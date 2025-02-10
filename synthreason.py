import random
import math
from collections import defaultdict, Counter

class SemanticGenerator:
    def __init__(self):
        self.words = defaultdict(lambda: defaultdict(Counter))
        self.context_window = 2  # Defines how many words influence the next choice
    
    def train(self, text):
        """Trains the model with given text."""
        tokens = text.split()
        for i in range(len(tokens) - 1):
            word, next_word = tokens[i], tokens[i + 1]
            category = "global"
            self.words[word][category][next_word] += 1
    
    def _normalize_probabilities(self, category_counts):
        """Convert raw frequency counts into probabilities."""
        total_count = sum(category_counts.values())
        return {word: count / total_count for word, count in category_counts.items()}
    
    def _cross_entropy_loss(self, true_dist, predicted_dist):
        """Compute cross-entropy loss between two probability distributions."""
        epsilon = 1e-10  # Avoid log(0)
        return -sum(true_dist[word] * math.log(predicted_dist.get(word, epsilon)) for word in true_dist)
    
    def generate_text(self, seed, length=10):
        """Generates text starting with a seed word."""
        if seed not in self.words:
            return "[ERROR] Seed word not found in training data."
        
        generated_words = [seed]
        for _ in range(length - 1):
            current_word = generated_words[-1]
            category = "global"
            
            if current_word not in self.words:
                break
            
            true_dist = self._normalize_probabilities(self.words[current_word][category])
            predicted_dist = self._normalize_probabilities(Counter(generated_words))
            
            loss = self._cross_entropy_loss(true_dist, predicted_dist)
            adjusted_weights = {word: count / (loss + 1e-5) for word, count in true_dist.items()}
            
            next_word = random.choices(list(adjusted_weights.keys()), weights=adjusted_weights.values())[-1]
            generated_words.append(next_word)
        
        return " ".join(generated_words)

# Example usage:
generator = SemanticGenerator()
KB_limit = -1
with open(input("Enter filename: "), 'r', encoding='utf-8') as f:
    text = ' '.join(f.read().split()[:KB_limit])
generator.train(text)
while True:
    print("AI:", generator.generate_text(input("Enter prompt: "), 250))
