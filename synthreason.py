import random
import math
from collections import defaultdict, Counter
from itertools import islice

class SemanticGenerator:
    def __init__(self, context_window=2):
        self.words = defaultdict(lambda: defaultdict(Counter))
        self.context_window = context_window  # Defines how many words influence the next choice
        self.prev_probs = defaultdict(Counter)
        self.distal_probs = defaultdict(Counter)
        self.future_probs = defaultdict(Counter)
        self.proximal_probs = defaultdict(Counter)

    def _get_context(self, tokens, i):
        """Get the context as a tuple of previous words up to `context_window` size."""
        return tuple(tokens[max(0, i - self.context_window):i])

    def train(self, text):
        """Trains the model with given text, using n-gram context."""
        tokens = text.split()
        for i in range(1, len(tokens)):
            context = self._get_context(tokens, i)  # Get previous `context_window` words
            next_word = tokens[i]
            category = "global"
            self.words[context][category][next_word] += 1  # Store transition probabilities

            # Update previous, distal, future, and proximal probabilities
            if i > 0:
                self.prev_probs[tokens[i-1]][next_word] += 1
            if i > 1:
                self.distal_probs[tokens[i-2]][next_word] += 1
            if i < len(tokens) - 1:
                self.future_probs[next_word][tokens[i+1]] += 1
            if i < len(tokens) - 2:
                self.proximal_probs[next_word][tokens[i+2]] += 1

    def _normalize_probabilities(self, category_counts):
        """Convert raw frequency counts into probabilities."""
        total_count = sum(category_counts.values())
        return {word: count / total_count for word, count in category_counts.items()}

    def _cross_entropy_loss(self, true_dist, predicted_dist):
        """Compute cross-entropy loss between two probability distributions."""
        epsilon = 1e-10  # Avoid log(0)
        return -sum(true_dist[word] * math.log(predicted_dist.get(word, epsilon)) for word in true_dist)

    def generate_text(self, seed, length=10):
        """Generates text starting with a seed phrase using context-based selection."""
        seed_tokens = seed.split()
        if len(seed_tokens) < self.context_window:
            return "[ERROR] Seed must have at least {} words.".format(self.context_window)

        generated_words = list(seed_tokens)

        for _ in range(length - len(seed_tokens)):
            current_context = tuple(generated_words[-self.context_window:])  # Get last `context_window` words
            category = "global"
            if current_context not in self.words or not self.words[current_context][category]:
                break  # Stop if context is unknown

            true_dist = self._normalize_probabilities(self.words[current_context][category])
            predicted_dist = self._normalize_probabilities(Counter(generated_words))
            loss = self._cross_entropy_loss(true_dist, predicted_dist)

            # Combine probabilities from previous, distal, future, and proximal contexts
            adjusted_weights = defaultdict(float)
            for word, count in true_dist.items():
                prev_prob = self.prev_probs[generated_words[-2]][word] if len(generated_words) > 0 else 0
                distal_prob = self.distal_probs[generated_words[-1]][word] if len(generated_words) > 1 else 0
                future_prob = sum(self.future_probs[word].values())
                proximal_prob = sum(self.proximal_probs[word].values())
                adjusted_weights[word] = (count / (loss + 1e-5)) * (1 + prev_prob + distal_prob + future_prob + proximal_prob)

            next_word = random.choices(list(adjusted_weights.keys()), weights=adjusted_weights.values())[-1]
            generated_words.append(next_word)

        return " ".join(generated_words)

# Example usage:
generator = SemanticGenerator(context_window=2)  # Set context size

KB_limit = -1
with open(input("Enter filename: "), 'r', encoding='utf-8') as f:
    text = ' '.join(f.read().split()[:KB_limit])

generator.train(text)

while True:
    print()
    print("AI:", generator.generate_text(input("Enter prompt: "), 250))