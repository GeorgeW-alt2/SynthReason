import random
import math
from collections import defaultdict, Counter
from itertools import islice

class SyntacticSemanticGenerator:
    def __init__(self, context_window=2):
        self.words = defaultdict(lambda: defaultdict(Counter))
        self.pos_tags = defaultdict(Counter)  # Part of speech patterns
        self.word_pos = defaultdict(Counter)  # Word to POS mappings
        self.pos_transitions = defaultdict(Counter)  # POS transition patterns
        self.context_window = context_window
        
    def _get_context(self, tokens, i):
        """Get the context as a tuple of previous words."""
        return tuple(tokens[max(0, i - self.context_window):i])
    
    def _infer_pos_tag(self, word, context):
        """Infer part of speech tag based on context and patterns."""
        if not context:
            return "UNKNOWN"
            
        # Use word endings and patterns to guess POS
        if word.endswith(('ly',)):
            return "ADV"
        elif word.endswith(('ing',)):
            return "VERB"
        elif word.endswith(('ed', 'en')):
            return "VERB_PAST"
        elif word.endswith(('s',)) and not word.endswith(('ss', 'us', 'is')):
            return "NOUN_PLURAL"
        elif word.endswith(('able', 'ible')):
            return "ADJ"
        
        # Use contextual clues
        prev_word = context[-1] if context else ""
        if prev_word in {"the", "a", "an"}:
            return "NOUN"
        elif prev_word in {"very", "quite", "rather"}:
            return "ADJ"
        
        # Return most common POS for this word
        return self.word_pos[word].most_common(1)[0][-1] if self.word_pos[word] else "UNKNOWN"
    
    def _analyze_syntax(self, tokens):
        """Analyze syntactic patterns in text."""
        pos_sequence = []
        for i, word in enumerate(tokens):
            context = self._get_context(tokens, i)
            pos = self._infer_pos_tag(word, context)
            pos_sequence.append(pos)
            
            # Update POS statistics
            self.word_pos[word][pos] += 1
            if i > 0:
                self.pos_transitions[pos_sequence[-2]][pos] += 1
                
        return pos_sequence
    
    def train(self, text):
        """Train the model with both semantic and syntactic patterns."""
        tokens = text.split()
        pos_sequence = self._analyze_syntax(tokens)
        
        for i in range(1, len(tokens)-1):
            context = self._get_context(tokens, i)
            next_word = tokens[i]
            category = "global"
            
            # Store transition probabilities
            self.words[context][category][next_word] += 1
            
            # Store POS patterns
            context_pos = tuple(pos_sequence[max(0, i - self.context_window):i])
            next_pos = pos_sequence[i+1]
            self.pos_tags[context_pos][next_pos] += 1
    
    def _normalize_probabilities(self, category_counts):
        """Normalize frequency counts to probabilities."""
        total_count = sum(category_counts.values())
        return {word: count / total_count for word, count in category_counts.items()}
    
    def _cross_entropy_loss(self, true_dist, predicted_dist):
        """Compute cross-entropy loss between distributions."""
        epsilon = 1e-10
        return -sum(true_dist[word] * math.log(predicted_dist.get(word, epsilon)) 
                   for word in true_dist)
    
    def _get_syntactic_score(self, word, context, pos_sequence):
        """Calculate syntactic coherence score for a word in context."""
        context_pos = tuple(pos_sequence[-self.context_window:])
        word_pos = self._infer_pos_tag(word, context)
        
        # Get probability of this POS following the context
        pos_prob = (self.pos_tags[context_pos][word_pos] + 1) / \
                  (sum(self.pos_tags[context_pos].values()) + len(self.pos_tags))
                  
        # Get probability of word having this POS
        word_pos_prob = (self.word_pos[word][word_pos] + 1) / \
                       (sum(self.word_pos[word].values()) + len(self.word_pos))
                       
        return pos_prob * word_pos_prob
    
    def generate_text(self, seed, length=10):
        """Generate text using both semantic and syntactic patterns."""
        seed_tokens = seed.split()
        if len(seed_tokens) < self.context_window:
            return f"[ERROR] Seed must have at least {self.context_window} words."
        
        generated_words = list(seed_tokens)
        pos_sequence = self._analyze_syntax(generated_words)
        
        for _ in range(length - len(seed_tokens)):
            current_context = tuple(generated_words[-self.context_window:])
            category = "global"
            
            if current_context not in self.words or not self.words[current_context][category]:
                break
            
            # Get semantic probabilities
            true_dist = self._normalize_probabilities(self.words[current_context][category])
            predicted_dist = self._normalize_probabilities(Counter(generated_words))
            loss = self._cross_entropy_loss(true_dist, predicted_dist)
            
            # Combine semantic and syntactic scores
            adjusted_weights = {}
            for word, sem_prob in true_dist.items():
                syn_score = self._get_syntactic_score(word, current_context, pos_sequence)
                combined_score = sem_prob * syn_score / (loss + 1e-5)
                adjusted_weights[word] = combined_score
            
            # Normalize weights
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                normalized_weights = {w: p/total_weight for w, p in adjusted_weights.items()}
            else:
                normalized_weights = adjusted_weights
            
            # Select next word
            next_word = random.choices(
                list(normalized_weights.keys()),
                weights=normalized_weights.values()
            )[-1]
            
            generated_words.append(next_word)
            pos_sequence.append(self._infer_pos_tag(next_word, current_context))
        
        return " ".join(generated_words)

# Example usage:
if __name__ == "__main__":
    generator = SyntacticSemanticGenerator(context_window=2)
    KB_limit = -1
    
    with open(input("Enter filename: "), 'r', encoding='utf-8') as f:
        text = ' '.join(f.read().split()[:KB_limit])
    
    generator.train(text)
    
    while True:
        print()
        print("AI:", generator.generate_text(input("Enter prompt: "), 250))