
from collections import defaultdict
from typing import Optional, List, Dict, Tuple
import numpy as np
import random

class TrigramPredictor:
    def __init__(self, contemplative_prob: float = 0.05):
        # Store trigram frequencies
        self.trigram_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Store word frequencies for random sampling
        self.word_frequencies: Dict[str, int] = defaultdict(int)
        self.contemplative_prob = contemplative_prob
        
        # Multi-word contemplative phrases
        self.contemplative_phrases = [
            # Uncertainty and reflection
            "perhaps it is", "perhaps we should", "perhaps one might",
            "it might be that", "it seems possible that",
            "one might consider that", "it could well be that",
            
            # Analysis and consideration
            "when one considers", "upon reflection", "in light of this",
            "it stands to reason", "it bears noting that",
            "taking this into account", "given this context",
            
            # Speculation and possibility
            "we might suppose that", "one could imagine that",
            "it is conceivable that", "let us consider that",
            "it is worth examining", "we might ponder whether",
            
            # Observation and insight
            "it appears that", "one notices that", "it becomes clear that",
            "looking more closely", "upon examination", "as we can see",
            
            # Transitions and implications
            "this suggests that", "which implies that", "from this we see",
            "this naturally leads to", "this brings us to", "consequently we find",
            
            # Extended contemplation
            "it is worth noting that", "we must consider whether",
            "it is interesting to observe", "one cannot overlook that",
            "it becomes apparent that", "we should recognize that"
        ]
        
    def _get_trigrams(self, words: List[str]) -> List[Tuple[str, str]]:
        """Generate trigrams from a list of words."""
        trigrams = []
        for i in range(len(words) - 2):
            context = f"{words[i]} {words[i+1]}"
            target = words[i+2]
            trigrams.append((context, target))
        return trigrams
    
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
    
    def predict(self, sequence: str) -> List[Tuple[str, float]]:
        """Predict next word given a sequence using trigram probabilities."""
        words = sequence.lower().split()
        
        # Need at least 2 words for context
        if len(words) < 2:
            return []
        
        # Get last two words for context
        context = f"{words[-2]} {words[-1]}"
        
        if context not in self.trigram_counts:
            return []
        
        # Calculate probabilities
        counts = self.trigram_counts[context]
        total = sum(counts.values())
        probs = [(word, count/total) for word, count in counts.items()]
        
        return sorted(probs, key=lambda x: x[1], reverse=True)

    def _sample_next_word(self, predictions: List[Tuple[str, float]], temperature: float = 1.0) -> str:
        """Sample next word from predictions using temperature, with contemplative phrases."""
        # Randomly inject contemplative phrases
        if random.random() < self.contemplative_prob:
            phrase = random.choice(self.contemplative_phrases)
            phrase_words = phrase.split()
            # Return first word and store rest for next iterations
            self._pending_words = phrase_words[1:]
            return phrase_words[0]
            
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

    def generate_text(self, seed: str, length: int = 50, temperature: float = 1.0, update_frequencies: bool = True) -> str:
        """Generate text starting from a seed sequence."""
        # Initialize sequence with seed
        current_sequence = seed.lower().split()
        generated_text = current_sequence.copy()
        
        # Clear any pending words at start of generation
        self._pending_words = []
        
        # Ensure we have at least 2 words for context
        if len(current_sequence) < 2:
            # Pad with random words if needed
            words = list(self.word_frequencies.keys())
            frequencies = list(self.word_frequencies.values())
            total = sum(frequencies)
            probs = [f/total for f in frequencies]
            padding = list(np.random.choice(words, size=max(0, 2-len(current_sequence)), p=probs))
            current_sequence = padding + current_sequence
        
        # Track new frequencies during generation
        new_frequencies = defaultdict(int)
        
        for _ in range(length):
            # Get predictions for current sequence
            predictions = self.predict(" ".join(current_sequence[-2:]))
            
            # Sample next word
            next_word = self._sample_next_word(predictions, temperature)
            
            # Update frequencies
            if update_frequencies:
                new_frequencies[next_word] += 1
            
            # Add to generated text and update sequence
            generated_text.append(next_word)
            current_sequence.append(next_word)
            
            # Keep only the last 2 words for context
            current_sequence = current_sequence[-2:]
        
        # Update model's word frequencies if requested
        if update_frequencies:
            for word, freq in new_frequencies.items():
                self.word_frequencies[word] += freq
                
            # Update trigram counts with generated sequences
            for context, target in self._get_trigrams(generated_text):
                self.trigram_counts[context][target] += 1
        
        return " ".join(generated_text)

# Example usage
if __name__ == "__main__":
    predictor = TrigramPredictor(contemplative_prob=0.05)
    
    # Training text
    try:
        with open(input("Enter filename: "), 'r', encoding='utf-8') as f:
            training_text = f.read().strip()
        
        # Train the model
        predictor.learn(training_text)
        
        # Interactive generation loop
        while True:
            seed = input("USER: ").strip()
            if not seed:  # Exit on empty input
                break
            print("AI:", predictor.generate_text(
                seed=seed,
                length=250,
                temperature=0.7  # Using a reasonable temperature value
            ))
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
