import numpy as np
import random
import re
import pickle
import math
import os
from collections import defaultdict, Counter, deque
from typing import List, Tuple, Dict, Any, Optional, Deque

class ContextWindow:
    def __init__(self, size: int = 3):
        self.size = size
        self.window: Deque[str] = deque(maxlen=size)
    
    def add(self, word: str):
        self.window.append(word)
    
    def add_multiple(self, words: List[str]):
        for word in words[-self.size:]:
            self.window.append(word)
    
    def get_context(self) -> str:
        return ' '.join(self.window)
    
    def clear(self):
        self.window.clear()

class ErrorAwareSemanticGenerator:
    def __init__(self, decay_rate: float = 0.95, convergence_threshold: float = 1e-6, context_size: int = 5):
        self.words = defaultdict(lambda: defaultdict(Counter))
        self.context_transitions = defaultdict(Counter)
        self.error_history: List[Tuple[float, int]] = []
        self.memory_indices: List[int] = []
        self.error_magnitudes: List[float] = []
        self.sigma_history: List[float] = []  # Track standard deviations
        self.running_mean: float = 0.0
        self.running_variance: float = 0.0
        self.decay_rate = decay_rate
        self.convergence_threshold = convergence_threshold
        self.is_converged = False
        self.total_epochs = 0
        self.context_window = ContextWindow(context_size)
        self.context_size = context_size
        self.window_size = 20  # Size of window for rolling statistics

    def clean_text(self, text: str) -> str:
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^A-Za-z0-9\s.,!?\'-]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        if text and text[0].isalpha():
            text = text[0].upper() + text[1:]
        if text and text[-1] not in '.!?':
            text += '.'
            
        return text

    def _categorize_word(self, word: str) -> str:
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

    def update_running_statistics(self, new_error: float) -> None:
        """Update running mean and variance using Welford's online algorithm."""
        n = len(self.error_history)
        if n == 0:
            self.running_mean = new_error
            self.running_variance = 0.0
        else:
            old_mean = self.running_mean
            self.running_mean += (new_error - old_mean) / (n + 1)
            self.running_variance += (new_error - old_mean) * (new_error - self.running_mean)

    def calculate_rolling_sigma(self) -> float:
        """Calculate rolling standard deviation over recent errors."""
        if len(self.error_history) < self.window_size:
            return float('inf')
        
        recent_errors = [error for error, _ in self.error_history[-self.window_size:]]
        mean = sum(recent_errors) / len(recent_errors)
        variance = sum((x - mean) ** 2 for x in recent_errors) / len(recent_errors)
        return math.sqrt(variance)

    def check_convergence(self) -> bool:
        if len(self.error_history) < self.window_size:
            return False
            
        # Calculate rolling statistics
        sigma = self.calculate_rolling_sigma()
        self.sigma_history.append(sigma)
        
        # Check both absolute error and stability
        recent_errors = [error for error, _ in self.error_history[-self.window_size:]]
        error_mean = sum(recent_errors) / len(recent_errors)
        
        # Criteria for convergence:
        # 1. Mean error below threshold
        # 2. Standard deviation stable and small
        error_stable = sigma < self.convergence_threshold
        mean_acceptable = error_mean < self.convergence_threshold * 2
        
        # Check trend in standard deviation
        sigma_stable = False
        if len(self.sigma_history) >= 3:
            recent_sigmas = self.sigma_history[-3:]
            sigma_changes = [abs(recent_sigmas[i] - recent_sigmas[i-1]) 
                           for i in range(1, len(recent_sigmas))]
            sigma_stable = all(change < self.convergence_threshold/2 for change in sigma_changes)
        
        return sigma_stable

    def train_until_convergence(self, text: str, max_epochs: int = 100) -> List[float]:
        """Train the model until convergence or maximum epochs reached."""
        text = self.clean_text(text)
        sentences = text.split('.')
        epoch_errors = []
        
        while self.total_epochs < max_epochs and not self.is_converged:
            epoch_error = 0.0
            epoch_errors_list = []  # Store all errors in epoch for variance calculation
            num_samples = 0
            current_time = self.total_epochs
            
            for sentence in sentences:
                words = sentence.lower().split()
                self.context_window.clear()
                
                for i, word in enumerate(words):
                    word = word.strip('.,!?')
                    if not word:
                        continue
                        
                    # Update context transitions
                    context = self.context_window.get_context()
                    if context:
                        self.context_transitions[context][word] += 1
                    
                    # Regular word-based training with error tracking
                    category = self._categorize_word(word)
                    prev_word = words[i-1].strip('.,!?') if i > 0 else 'start'
                    
                    self.memory_indices.append(i)
                    error_magnitude = 1.0 if prev_word not in self.words else 0.5
                    self.error_magnitudes.append(error_magnitude)
                    
                    # Calculate decayed error
                    decayed_error = error_magnitude * (self.decay_rate ** self.total_epochs)
                    self.error_history.append((decayed_error, current_time))
                    epoch_errors_list.append(decayed_error)
                    
                    # Update running statistics
                    self.update_running_statistics(decayed_error)
                    
                    self.words[prev_word][category][word] += 1
                    self.context_window.add(word)
                    
                    epoch_error += decayed_error
                    num_samples += 1
            
            # Calculate epoch statistics
            avg_epoch_error = epoch_error / num_samples if num_samples > 0 else 0
            epoch_errors.append(avg_epoch_error)
            
            # Calculate epoch standard deviation
            if epoch_errors_list:
                epoch_sigma = math.sqrt(sum((x - avg_epoch_error) ** 2 for x in epoch_errors_list) / len(epoch_errors_list))
            else:
                epoch_sigma = 0.0
                
            self.is_converged = self.check_convergence()
            self.total_epochs += 1
            
            if self.total_epochs % 10 == 0:
                print(f"Epoch {self.total_epochs}:")
                print(f"  Average Error = {avg_epoch_error:.6f}")
                print(f"  Sigma = {epoch_sigma:.6f}")
                print(f"  Rolling Sigma = {self.calculate_rolling_sigma():.6f}")
                
        print(f"\nTraining completed after {self.total_epochs} epochs")
        print(f"Converged: {self.is_converged}")
        print(f"Final Rolling Sigma: {self.calculate_rolling_sigma():.6f}")
        return epoch_errors

    def generate_text(self, num_words: int = 50) -> str:
        if not self.is_converged:
            print("Warning: Model not trained or loaded. Results may be unreliable.")
            
        generated_words = []
        self.context_window.clear()
        print()
        input_text = input("Enter starting words (space-separated): ").lower().strip()
        initial_words = [word.strip('.,!?') for word in input_text.split()]
        
        self.context_window.add_multiple(initial_words)
        generated_words.extend(initial_words)
        
        current_word = initial_words[-1] if initial_words else 'start'
        attempts = 0
        max_attempts = num_words * 2
        
        # Calculate current sigma state for adaptive selection
        current_sigma = self.calculate_rolling_sigma()
        sigma_factor = 1.0 / (1.0 + current_sigma) if current_sigma != float('inf') else 0.5
        
        while len(generated_words) < num_words and attempts < max_attempts:
            attempts += 1
            
            context = self.context_window.get_context()
            if context in self.context_transitions and self.context_transitions[context]:
                candidates = list(self.context_transitions[context].items())
                words, counts = zip(*candidates)
                
                # Adjust weights based on sigma stability
                adjusted_counts = [count * sigma_factor for count in counts]
                next_word = random.choices(words, weights=adjusted_counts)[0]
            else:
                if current_word not in self.words:
                    current_word = random.choice(generated_words[-3:] or ['start'])
                    continue
                
                categories = []
                weights = []
                category_counts = defaultdict(float)
                
                # Calculate category weights with sigma consideration
                for category, word_counts in self.words[current_word].items():
                    if word_counts:
                        total_count = sum(word_counts.values())
                        category_counts[category] = total_count
                        
                        # Apply sigma-based adjustment
                        adjusted_weight = total_count * sigma_factor
                        categories.append(category)
                        categories.append(category)

                        weights.append(adjusted_weight)
                        weights.append(total_count * current_sigma)

                if not categories:
                    current_word = random.choice(generated_words[-3:] or ['start'])
                    continue
                
                # Calculate stability metric for the current choices
                mean_weight = sum(weights) / len(weights) if weights else 0
                weight_variance = sum((w - mean_weight) ** 2 for w in weights) / len(weights) if weights else 0
                choice_stability = 1.0 / (1.0 + math.sqrt(weight_variance))
                
                # Apply stability-adjusted selection
                adjusted_weights = [w * choice_stability for w in weights]
                category = random.choices(categories, weights=adjusted_weights)[0]
                
                available_words = []
                word_weights = []
                
                # Select words within the chosen category
                for word, count in self.words[current_word][category].items():
                    available_words.append(word)
                    # Apply both sigma and category stability factors
                    adjusted_weight = count * sigma_factor * choice_stability
                    word_weights.append(adjusted_weight)
                
                if not available_words:
                    current_word = random.choice(generated_words[-3:] or ['start'])
                    continue
                
                next_word = random.choices(available_words, weights=word_weights)[0]
            
            if next_word != generated_words[-1]:
                generated_words.append(next_word)
                self.context_window.add(next_word)
                current_word = next_word
            
                # Update sigma factor periodically
                if len(generated_words) % 10 == 0:
                    current_sigma = self.calculate_rolling_sigma()
                    sigma_factor = 1.0 / (1.0 + current_sigma) if current_sigma != float('inf') else 0.5
            
            if next_word.endswith('.'):
                self.context_window.clear()
                if len(generated_words) < num_words:
                    self.context_window.add_multiple(generated_words[-min(3, len(generated_words)):])
        
        text = ' '.join(generated_words)
        sentences = text.split('.')
        formatted_sentences = []
        
        for sentence in sentences:
            if sentence.strip():
                formatted = sentence.strip()
                formatted = formatted[0].upper() + formatted[1:] + '.'
                formatted_sentences.append(formatted)
        
        return ' '.join(formatted_sentences)

    def save_model(self, filepath: str):
        try:
            model_state = {
                'words': {
                    context: {
                        category: dict(counter)
                        for category, counter in categories.items()
                    }
                    for context, categories in self.words.items()
                },
                'context_transitions': {
                    context: dict(transitions)
                    for context, transitions in self.context_transitions.items()
                },
                'error_history': self.error_history,
                'memory_indices': self.memory_indices,
                'error_magnitudes': self.error_magnitudes,
                'sigma_history': self.sigma_history,
                'running_mean': self.running_mean,
                'running_variance': self.running_variance,
                'decay_rate': self.decay_rate,
                'convergence_threshold': self.convergence_threshold,
                'is_converged': self.is_converged,
                'total_epochs': self.total_epochs,
                'context_size': self.context_size,
                'window_size': self.window_size
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(model_state, f)
            
        except Exception as e:
            raise Exception(f"Failed to save model: {str(e)}")

    def load_model(self, filepath: str):
        try:
            with open(filepath, 'rb') as f:
                model_state = pickle.load(f)
            
            self.words = defaultdict(lambda: defaultdict(Counter))
            for context, categories in model_state['words'].items():
                for category, words in categories.items():
                    self.words[context][category].update(words)
            
            self.context_transitions = defaultdict(Counter)
            for context, transitions in model_state['context_transitions'].items():
                self.context_transitions[context].update(transitions)
            
            self.error_history = model_state['error_history']
            self.memory_indices = model_state['memory_indices']
            self.error_magnitudes = model_state['error_magnitudes']
            self.sigma_history = model_state.get('sigma_history', [])
            self.running_mean = model_state.get('running_mean', 0.0)
            self.running_variance = model_state.get('running_variance', 0.0)
            self.decay_rate = model_state['decay_rate']
            self.convergence_threshold = model_state['convergence_threshold']
            self.is_converged = model_state['is_converged']
            self.total_epochs = model_state['total_epochs']
            self.context_size = model_state.get('context_size', 5)
            self.window_size = model_state.get('window_size', 10)
            
            self.context_window = ContextWindow(self.context_size)
            
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")

def main():
    print("Multi-Word Context-Aware Semantic Text Generator")
    print("=============================================")
    
    generator = ErrorAwareSemanticGenerator(
        decay_rate=0.25,
        convergence_threshold=1e-6,
        context_size=15
    )
    
    if not os.path.exists('models'):
        os.makedirs('models')
    
    while True:
        print("\nOptions:")
        print("1. Train model")
        print("2. Generate text")
        print("3. Save model")
        print("4. Load model")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            try:
                filename = input("Enter filename: ")
                with open(filename, 'r', encoding='utf-8') as f:
                    text = f.read()
                print("\nTraining model...")
                errors = generator.train_until_convergence(text)
                print(f"Final error: {errors[-1]:.6f}")
                
                model_path = os.path.join('models', 'auto_saved_model.pkl')
                generator.save_model(model_path)
                print(f"Model automatically saved to {model_path}")
            except FileNotFoundError:
                print("Error: Training file not found!")
            except Exception as e:
                print(f"Error during training: {e}")
                
        elif choice == "2":
            if not generator.is_converged:
                print("Warning: Model not trained or loaded. Results may be unreliable.")
            try:
                num_words = 500
                while True:
                    generated_text = generator.generate_text(num_words)
                    print("\nGenerated text:")
                    print(generated_text)
                    
            except ValueError as e:
                print(f"Error: {e}")
                
        elif choice == "3":
            try:
                model_name = input("Enter model name to save (will be saved in 'models' directory): ")
                if not model_name.endswith('.pkl'):
                    model_name += '.pkl'
                model_path = os.path.join('models', model_name)
                
                generator.save_model(model_path)
                print(f"Model saved successfully to {model_path}")
            except Exception as e:
                print(f"Error saving model: {e}")
                
        elif choice == "4":
            try:
                models = [f for f in os.listdir('models') if f.endswith('.pkl')]
                if not models:
                    print("No saved models found in 'models' directory.")
                    continue
                    
                print("\nAvailable models:")
                for i, model in enumerate(models, 1):
                    print(f"{i}. {model}")
                
                choice = input("\nEnter model number to load (or filename): ").strip()
                
                if choice.isdigit() and 1 <= int(choice) <= len(models):
                    model_name = models[int(choice) - 1]
                else:
                    model_name = choice if choice.endswith('.pkl') else choice + '.pkl'
                
                model_path = os.path.join('models', model_name)
                generator.load_model(model_path)
                print(f"Model loaded successfully from {model_path}")
            except FileNotFoundError:
                print("Error: Model file not found!")
            except Exception as e:
                print(f"Error loading model: {e}")
                
        elif choice == "5":
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
