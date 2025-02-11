import random
import math
from collections import defaultdict, Counter
from itertools import islice

class SemanticGenerator:
    def __init__(self, context_window=2):
        self.words = defaultdict(lambda: defaultdict(Counter))
        self.word_counts = Counter()  # Count of each word in the entire corpus
        self.context_counts = Counter()  # Count of each context (sequence of words)
        self.context_window = context_window  # Defines how many words influence the next choice
    
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
            
            # Store word and context counts
            self.word_counts[next_word] += 1
            self.context_counts[context] += 1
            self.words[context][category][next_word] += 1  # Store transition probabilities
    
    def _normalize_probabilities(self, category_counts):
        """Convert raw frequency counts into probabilities."""
        total_count = sum(category_counts.values())
        return {word: count / total_count for word, count in category_counts.items()}
    
    def _cross_entropy_loss(self, true_dist, predicted_dist):
        """Compute cross-entropy loss between two probability distributions."""
        epsilon = 1e-10  # Avoid log(0)
        return -sum(true_dist[word] * math.log(predicted_dist.get(word, epsilon)) for word in true_dist)
    
    def _get_prior_probability(self, word):
        """Get the prior probability of a word based on its frequency in the entire corpus."""
        total_word_count = sum(self.word_counts.values())
        return self.word_counts[word] / total_word_count if total_word_count > 0 else 0
    
    def _get_prior_context_probability(self, context):
        """Get the prior probability of a context (sequence of words)."""
        total_context_count = sum(self.context_counts.values())
        return self.context_counts[context] / total_context_count if total_context_count > 0 else 0
    
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

            # Compute cross-entropy loss
            loss = self._cross_entropy_loss(true_dist, predicted_dist)
            
            # Adjust the weights by loss
            adjusted_weights = {word: count / (loss + 1e-5) for word, count in true_dist.items()}

            # Incorporating prior probabilities of words and context into the generation process
            weighted_adjusted_weights = {}
            for word, weight in adjusted_weights.items():
                prior_word_prob = self._get_prior_probability(word)  # Get prior probability of word
                prior_context_prob = self._get_prior_context_probability(current_context)  # Get prior probability of context
                # Combine adjusted weight, prior word probability, and prior context probability
                weighted_adjusted_weights[word] = weight * (prior_word_prob + prior_context_prob)
            
            # Normalize the weights again
            total_weight = sum(weighted_adjusted_weights.values())
            if total_weight > 0:
                normalized_weights = {word: weight / total_weight for word, weight in weighted_adjusted_weights.items()}
            else:
                normalized_weights = adjusted_weights

            # Select the next word based on the adjusted probabilities
            next_word = random.choices(list(normalized_weights.keys()), weights=normalized_weights.values())[-1]
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
