import random
import math
from collections import defaultdict, Counter

class CharSemanticGenerator:
    def __init__(self, context_window=3):
        self.chars = defaultdict(lambda: defaultdict(Counter))
        self.context_window = context_window  # Defines how many characters influence the next choice
    
    def _get_context(self, text, i):
        """Get the context as a tuple of previous characters up to `context_window` size."""
        return tuple(text[max(0, i - self.context_window):i])
    
    def train(self, text):
        """Trains the model with given text, using n-gram character context."""
        for i in range(1, len(text)):
            context = self._get_context(text, i)  # Get previous `context_window` characters
            next_char = text[i]
            category = "global"
            self.chars[context][category][next_char] += 1  # Store transition probabilities
    
    def _normalize_probabilities(self, category_counts):
        """Convert raw frequency counts into probabilities."""
        total_count = sum(category_counts.values())
        return {char: count / total_count for char, count in category_counts.items()}
    
    def _cross_entropy_loss(self, true_dist, predicted_dist):
        """Compute cross-entropy loss between two probability distributions."""
        epsilon = 1e-10  # Avoid log(0)
        return -sum(true_dist[char] * math.log(predicted_dist.get(char, epsilon)) for char in true_dist)
    
    def generate_text(self, seed, length=100):
        """Generates text starting with a seed phrase using context-based character selection."""
        if len(seed) < self.context_window:
            return "[ERROR] Seed must have at least {} characters.".format(self.context_window)
        
        generated_chars = list(seed)
        
        for _ in range(length - len(seed)):
            current_context = tuple(generated_chars[-self.context_window:])  # Get last `context_window` characters
            category = "global"
            
            if current_context not in self.chars or not self.chars[current_context][category]:
                break  # Stop if context is unknown
            
            true_dist = self._normalize_probabilities(self.chars[current_context][category])
            predicted_dist = self._normalize_probabilities(Counter(generated_chars))
            
            # Compute cross-entropy loss
            loss = self._cross_entropy_loss(true_dist, predicted_dist)
            
            # Adjust the weights by loss
            adjusted_weights = {char: count / (loss + 1e-5) for char, count in true_dist.items()}
            
            # Select the next character based on the adjusted probabilities
            next_char = random.choices(list(adjusted_weights.keys()), weights=adjusted_weights.values())[-1]
            generated_chars.append(next_char)
        
        return "".join(generated_chars)

# Example usage:
generator = CharSemanticGenerator(context_window=3)  # Set context size
KB_limit = -1

with open(input("Enter filename: "), 'r', encoding='utf-8') as f:
    text = f.read()[:KB_limit]

# Preprocess text to handle any encoding issues
text = ''.join(char for char in text if char.isprintable())

generator.train(text)

while True:
    print()
    print("AI:", generator.generate_text(input("Enter prompt: "), 2500))