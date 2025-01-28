import numpy as np
import random
import re
import pickle
import math
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any, Optional

class ErrorAwareSemanticGenerator:
    """
    A text generator that combines semantic analysis with error convergence monitoring.
    Incorporates both probabilistic text generation and theoretical error bounds.
    """
    
    def __init__(self, decay_rate: float = 0.95, convergence_threshold: float = 1e-6):
        self.words = defaultdict(lambda: defaultdict(Counter))
        self.error_history: List[Tuple[float, int]] = []  # (magnitude, time)
        self.memory_indices: List[int] = []
        self.error_magnitudes: List[float] = []
        self.decay_rate = decay_rate
        self.convergence_threshold = convergence_threshold
        self.is_converged = False
        self.total_epochs = 0

    def clean_text(self, text: str) -> str:
        """Clean and normalize input text."""
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
        text = re.sub(r'\S+@\S+', '', text)  # Remove email addresses
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = re.sub(r'[^A-Za-z0-9\s.,!?\'-]', '', text)  # Keep basic punctuation
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        # Capitalize first letter and ensure proper ending
        if text and text[0].isalpha():
            text = text[0].upper() + text[1:]
        if text and text[-1] not in '.!?':
            text += '.'
            
        return text

    def _categorize_word(self, word: str) -> str:
        """Categorize word into linguistic categories."""
        if word.endswith('ly'):
            return 'adverbs'
        elif word.endswith('ed') or word.endswith('ing'):
            return 'verbs'
        elif word in {'the', 'a', 'an', 'this', 'that', 'these', 'those'}:
            return 'determiners'
        elif word in {'in', 'on', 'at', 'to', 'from', 'with', 'without', 'by', 'near', 'far'}:
            return 'prepositions'
        else:
            return 'nouns'

    def verify_error_bounds(self, time_steps: int) -> Tuple[float, float]:
        """Verify error function remains bounded over time steps."""
        max_error = float('-inf')
        min_error = float('inf')
        
        for t in range(time_steps):
            error_sum = 0.0
            for idx, magnitude in zip(self.memory_indices, self.error_magnitudes):
                # Calculate spatial and temporal components
                spatial_term = 1.0 / (1.0 + abs(idx))
                temporal_term = self.decay_rate ** t
                error_sum += magnitude * spatial_term * temporal_term
                
            max_error = max(max_error, error_sum)
            min_error = min(min_error, error_sum)
            
        return min_error, max_error

    def verify_entropy_bounds(self, access_patterns: List[int]) -> float:
        """Calculate and verify entropy remains within theoretical bounds."""
        if not access_patterns:
            return 0.0
            
        counts = np.bincount(access_patterns)
        probabilities = counts[counts > 0] / len(access_patterns)
        empirical_entropy = -np.sum(probabilities * np.log2(probabilities))
        max_entropy = math.log2(len(self.words))
        
        assert 0 <= empirical_entropy <= max_entropy, \
            "Entropy violates theoretical bounds"
            
        return empirical_entropy

    def check_convergence(self) -> bool:
        """Check if training has converged based on recent error history."""
        if len(self.error_history) < 10:
            return False
            
        recent_errors = [error for error, _ in self.error_history[-10:]]
        diffs = [abs(recent_errors[i] - recent_errors[i-1]) 
                for i in range(1, len(recent_errors))]
                
        return max(diffs) < self.convergence_threshold

    def train_until_convergence(self, text: str, max_epochs: int = 100) -> List[float]:
        """
        Train the model until error convergence or maximum epochs reached.
        Returns list of error values per epoch.
        """
        text = self.clean_text(text)
        sentences = text.split('.')
        epoch_errors = []
        
        while self.total_epochs < max_epochs and not self.is_converged:
            epoch_error = 0.0
            num_samples = 0
            current_time = self.total_epochs
            
            for sentence in sentences:
                words = sentence.lower().split()
                prev_word = None
                
                for i, word in enumerate(words):
                    word = word.strip('.,!?')
                    if word:
                        category = self._categorize_word(word)
                        context = prev_word if prev_word else 'start'
                        
                        # Track memory access and calculate error
                        self.memory_indices.append(i)
                        error_magnitude = 1.0 if context not in self.words else 0.5
                        self.error_magnitudes.append(error_magnitude)
                        
                        # Apply temporal decay
                        decayed_error = error_magnitude * (self.decay_rate ** self.total_epochs)
                        self.error_history.append((decayed_error, current_time))
                        
                        # Update model
                        self.words[context][category][word] += 1
                        
                        epoch_error += decayed_error
                        num_samples += 1
                        prev_word = word
            
            # Calculate average error for epoch
            avg_epoch_error = epoch_error / num_samples if num_samples > 0 else 0
            epoch_errors.append(avg_epoch_error)
            
            # Check convergence
            self.is_converged = self.check_convergence()
            self.total_epochs += 1
            
            if self.total_epochs % 10 == 0:
                print(f"Epoch {self.total_epochs}: Average Error = {avg_epoch_error:.6f}")
                
        print(f"\nTraining completed after {self.total_epochs} epochs")
        print(f"Converged: {self.is_converged}")
        return epoch_errors

    def verify_convergence(self, time_horizon: int) -> bool:
        """Verify convergence conditions for error distribution."""
        series_terms = []
        for magnitude, t_i in self.error_history:
            series = []
            for t in range(t_i, time_horizon):
                term = magnitude * (self.decay_rate ** (t - t_i))
                series.append(abs(term))
            series_terms.append(sum(series))
        
        total_sum = sum(series_terms)
        is_bounded = np.isfinite(total_sum)
        
        if len(series_terms) > 1:
            ratios = [series_terms[i+1]/series_terms[i] 
                     for i in range(len(series_terms)-1)]
            converges_geometrically = all(r < 1 for r in ratios)
        else:
            converges_geometrically = True
            
        return is_bounded and converges_geometrically

    def generate_text(self, num_words: int = 50) -> str:
        """Generate text using the trained model."""
        if not self.is_converged:
            print("Warning: Model has not converged. Results may be unreliable.")
            
        generated_words = []
        current_word = input("USER: ")
        attempts = 0
        max_attempts = num_words * 2  # Prevent infinite loops
        
        while len(generated_words) < num_words and attempts < max_attempts:
            attempts += 1
            
            # If current word has no following words, try starting over
            if current_word not in self.words or not self.words[current_word]:
                current_word = 'start'
                continue
                
            # Get all categories and their total counts for current word
            categories = []
            weights = []
            for category, word_counts in self.words[current_word].items():
                if word_counts:  # Only include non-empty categories
                    categories.append(category)
                    weights.append(sum(word_counts.values()))
            
            if not categories:
                current_word = 'start'
                continue
                
            # Select category based on weights
            category = random.choices(categories, weights=weights)[0]
            
            # Get all words and their counts in the selected category
            available_words = []
            word_weights = []
            for word, count in self.words[current_word][category].items():
                available_words.append(word)
                word_weights.append(count)
            
            if not available_words:
                current_word = 'start'
                continue
                
            # Select word based on weights
            word = random.choices(available_words, weights=word_weights)[0]
            
            if len(generated_words) == 0 or word != generated_words[-1]:  # Avoid repetition
                generated_words.append(word)
                current_word = word
            
        return ' '.join(generated_words).capitalize() + '.'

    def save_model(self, model: str):
        """Save the model state to a file."""
        model_state = {
            'words': dict(self.words),
            'error_history': self.error_history,
            'memory_indices': self.memory_indices,
            'error_magnitudes': self.error_magnitudes,
            'decay_rate': self.decay_rate,
            'convergence_threshold': self.convergence_threshold,
            'is_converged': self.is_converged,
            'total_epochs': self.total_epochs
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_state, f)

    def load_model(self, model: str):
        """Load model state from a file."""
        with open(filename, 'rb') as f:
            model_state = pickle.load(f)
        
        # Reconstruct defaultdict structure
        self.words = defaultdict(lambda: defaultdict(Counter))
        for context, categories in model_state['words'].items():
            for category, counter in categories.items():
                self.words[context][category].update(counter)
                
        # Load other attributes
        self.error_history = model_state['error_history']
        self.memory_indices = model_state['memory_indices']
        self.error_magnitudes = model_state['error_magnitudes']
        self.decay_rate = model_state['decay_rate']
        self.convergence_threshold = model_state['convergence_threshold']
        self.is_converged = model_state['is_converged']
        self.total_epochs = model_state['total_epochs']
model = "model.pkl"
def main():
    """Main function demonstrating usage of the generator."""
    print("Error-Aware Semantic Text Generator")
    print("==================================")
    
    generator = ErrorAwareSemanticGenerator(
        decay_rate=0.35,
        convergence_threshold=1e-6
    )
    
    while True:
        print("\nOptions:")
        print("1. Train on new text")
        print("2. Generate text")
        print("3. Analyze error convergence")
        print("4. Save model")
        print("5. Load model")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            text = input("Enter text to train on (or 'file:' followed by filename): ")
            if text.startswith('file:'):
                filename = text[5:].strip()
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        text = f.read()
                except FileNotFoundError:
                    print("File not found!")
                    continue
            
            print("\nTraining model...")
            errors = generator.train_until_convergence(text)
            print(f"Final error: {errors[-1]:.6f}")
            
        elif choice == "2":
            try:
                num_words = 250
                while True:
                    generated_text = generator.generate_text(num_words)
                    print("\nGenerated text:")
                    print(generated_text)
            except ValueError:
                print("Please enter a valid number!")
                
        elif choice == "3":
            try:
                time_steps = int(input("Enter number of time steps to analyze: "))
                min_error, max_error = generator.verify_error_bounds(time_steps)
                print(f"\nError bounds: [{min_error:.6f}, {max_error:.6f}]")
                
                # Verify convergence
                converged = generator.verify_convergence(time_steps)
                print(f"Convergence status: {'Converged' if converged else 'Not converged'}")
                
                # Calculate entropy if we have access patterns
                if generator.memory_indices:
                    entropy = generator.verify_entropy_bounds(generator.memory_indices)
                    print(f"Empirical entropy: {entropy:.6f}")
            except ValueError:
                print("Please enter a valid number!")
                
        elif choice == "4":
            filename = input("Enter filename to save model: ")
            generator.save_model(model)
            print("Model saved successfully!")
            
        elif choice == "5":
            filename = input("Enter filename to load model: ")
            try:
                generator.load_model(model)
                print("Model loaded successfully!")
            except FileNotFoundError:
                print("File not found!")
                
        elif choice == "6":
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()