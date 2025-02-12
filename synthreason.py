import random
import math
from collections import defaultdict, Counter

class TextGenerator:
    def __init__(self, context_window=3, traceback_depth=5):
        self.words = defaultdict(lambda: defaultdict(Counter))
        self.word_transitions = defaultdict(Counter)  # Track word-to-word transitions
        self.sequence_scores = defaultdict(float)     # Track sequence scores
        self.context_window = context_window
        self.traceback_depth = traceback_depth
        self.history = []  # Track generation history
        
    def _get_context(self, sequence, position):
        """Get context at a specific position."""
        start = max(0, position - self.context_window)
        context = sequence[start:position]
        return tuple(context)
    
    def train(self, text):
        """Train the model on input text."""
        words = text.split()
        self.history.extend(words)  # Add to history
        
        # Build transition probabilities
        for i in range(len(words)):
            # Get context
            context = self._get_context(words, i)
            word = words[i]
            
            # Store transitions
            self.words[context]["global"][word] += 1
            
            # Track word-to-word transitions
            if i > 0:
                prev_word = words[i-1]
                self.word_transitions[prev_word][word] += 1
            
            # Track n-gram sequences
            for n in range(2, self.context_window + 1):
                if i >= n:
                    sequence = tuple(words[i-n:i])
                    self.sequence_scores[sequence] += 1
                    
    def _score_sequence(self, sequence, position):
        """Score a sequence at a position looking backward."""
        score = 0.0
        
        # Look back through traceback window
        start = max(0, position - self.traceback_depth)
        for i in range(start, position + 1):
            # Get local context
            context = self._get_context(sequence, i)
            if i < len(sequence):
                word = sequence[i]
                
                # Score based on context probability
                if context in self.words:
                    total = sum(self.words[context]["global"].values())
                    if total > 0:
                        prob = self.words[context]["global"][word] / total
                        score += math.log(prob + 1e-10)
                
                # Score based on transitions
                if i > 0:
                    prev_word = sequence[i-1]
                    if prev_word in self.word_transitions:
                        total = sum(self.word_transitions[prev_word].values())
                        if total > 0:
                            prob = self.word_transitions[prev_word][word] / total
                            score += math.log(prob + 1e-10)
                
                # Score based on n-gram frequency
                for n in range(2, min(self.context_window + 1, i + 1)):
                    sequence_slice = tuple(sequence[i-n:i])
                    if sequence_slice in self.sequence_scores:
                        score += math.log(self.sequence_scores[sequence_slice] + 1)
        
        return score
    
    def _traceback_optimize(self, sequence, position):
        """Optimize sequence by looking back and trying alternatives."""
        best_sequence = sequence
        best_score = self._score_sequence(sequence, position)
        
        # Look back through traceback window
        start = max(0, position - self.traceback_depth)
        for i in range(start, position):
            context = self._get_context(sequence, i)
            if context in self.words:
                # Get top alternative words at this position
                alternatives = self.words[context]["global"].most_common(5)
                
                for alt_word, _ in alternatives:
                    # Create alternative sequence
                    new_sequence = sequence[:i] + [alt_word] + sequence[i+1:]
                    new_score = self._score_sequence(new_sequence, position)
                    
                    # Keep if better
                    if new_score > best_score:
                        best_sequence = new_sequence
                        best_score = new_score
        
        return best_sequence
    
    def generate_with_traceback(self, seed, length=100):
        """Generate text using traceback optimization."""
        # Initialize with seed
        sequence = seed.split()
        
        while len(sequence) < length:
            # Get current context
            context = self._get_context(sequence, len(sequence))
            
            if context not in self.words:
                # Backoff to smaller context
                while len(context) > 0 and context not in self.words:
                    context = context[1:]
                if not context:
                    break
            
            # Get probabilities for next word
            word_counts = self.words[context]["global"]
            total = sum(word_counts.values())
            
            if total == 0:
                break
                
            # Select next word
            probs = {word: count/total for word, count in word_counts.items()}
            words = list(probs.keys())
            weights = list(probs.values())
            next_word = random.choices(words, weights=weights)[0]
            
            # Add word to sequence
            sequence.append(next_word)
            
            # Periodically optimize using traceback
            if len(sequence) % 5 == 0:
                sequence = self._traceback_optimize(sequence, len(sequence) - 1)
            
            # After each word, look back and try to improve coherence
            if len(sequence) > self.traceback_depth:
                # Score current sequence
                current_score = self._score_sequence(sequence, len(sequence) - 1)
                
                # Try to improve recent segments
                improved_sequence = self._traceback_optimize(sequence, len(sequence) - 1)
                improved_score = self._score_sequence(improved_sequence, len(sequence) - 1)
                
                # Keep improvements
                if improved_score > current_score:
                    sequence = improved_sequence
        
        return " ".join(sequence)

def main():
    # Initialize generator
    print("Initializing text generator...")
    generator = TextGenerator(context_window=3, traceback_depth=5)
    
    # Get training file
    filename = input("Enter training file path: ")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Train model
    print("\nTraining model...")
    generator.train(text)
    print("Training complete!")
    
    # Generation loop
    while True:
        try:
            seed = input("Enter seed text: ")

            length = 250

            # Generate text
            result = generator.generate_with_traceback(seed, length)
            print("\nGenerated text:")
            print(result)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    main()