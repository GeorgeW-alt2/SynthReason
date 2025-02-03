# SynthReason Version 4.0
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
        self.window = deque(maxlen=size)
    
    def add(self, word: str):
        self.window.append(word)
    
    def add_multiple(self, words: List[str]):
        for word in words[-self.size:]:
            self.window.append(word)
    
    def get_context(self) -> str:
        return ' '.join(list(self.window))
    
    def clear(self):
        self.window.clear()

class ErrorAwareSemanticGenerator:
    def __init__(self, decay_rate: float = 0.95, probability_threshold: float = 0.01, context_size: int = 5):
        self.words = defaultdict(lambda: defaultdict(Counter))
        self.context_transitions = defaultdict(Counter)
        self.transition_probabilities = defaultdict(dict)
        self.prev_probabilities = defaultdict(dict)
        self.decay_rate = decay_rate
        self.probability_threshold = probability_threshold
        self.is_converged = False
        self.total_epochs = 0
        self.context_window = ContextWindow(context_size)
        self.context_size = context_size
        
    def _is_valid_word(self, word: str) -> bool:
        valid_one_letter = ['a', 'i']
        valid_two_letter = [
            'am', 'an', 'as', 'at', 'be', 'by', 'do', 'go', 'he', 'hi',
            'if', 'in', 'is', 'it', 'me', 'my', 'no', 'of', 'on', 'or',
            'so', 'to', 'up', 'us', 'we'
        ]
        
        clean_word = word.strip('.,!?')
        
        if len(clean_word) == 1:
            return clean_word.lower() in valid_one_letter
        elif len(clean_word) == 2:
            return clean_word.lower() in valid_two_letter
        
        return True
        
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
        elif word in ['the', 'a', 'an', 'this', 'that', 'these', 'those']:
            return 'determiners'
        elif word in ['in', 'on', 'at', 'to', 'from', 'with', 'without', 'by', 'near', 'far']:
            return 'prepositions'
        else:
            return 'nouns'

    def _calculate_transition_probabilities(self):
        """Calculate transition probabilities for all word pairs."""
        self.prev_probabilities = self.transition_probabilities.copy()
        self.transition_probabilities.clear()
        
        for prev_word, categories in self.words.items():
            total_transitions = sum(sum(counter.values()) for counter in categories.values())
            
            if total_transitions > 0:
                for category, counter in categories.items():
                    for word, count in counter.items():
                        prob = count / total_transitions
                        if prev_word not in self.transition_probabilities:
                            self.transition_probabilities[prev_word] = {}
                        self.transition_probabilities[prev_word][word] = prob

    def _compare_probabilities(self) -> bool:
        """Compare current and previous transition probabilities."""
        if not self.prev_probabilities:
            return False
            
        total_diff = 0
        count = 0
        
        for prev_word, transitions in self.transition_probabilities.items():
            if prev_word in self.prev_probabilities:
                for word, prob in transitions.items():
                    if word in self.prev_probabilities[prev_word]:
                        diff = abs(prob - self.prev_probabilities[prev_word][word])
                        total_diff += diff
                        count += 1
        
        if count == 0:
            return False
            
        avg_diff = total_diff / count
        return avg_diff < self.probability_threshold

    def train_until_convergence(self, text: str, max_epochs: int = 10) -> List[float]:
        """Train the model until transition probabilities converge."""
        text = self.clean_text(text)
        sentences = text.split('.')
        epoch_diffs = []
        
        while self.total_epochs < max_epochs:
            for sentence in sentences:
                words = sentence.lower().split()
                self.context_window.clear()
                
                for i, word in enumerate(words):
                    word = word.strip('.,!?')
                    if not word or not self._is_valid_word(word):
                        continue
                    if self.is_converged:    
                        context = self.context_window.get_context()
                        if context:
                            self.context_transitions[context][word] += 1
                    
                    prev_word = words[i-1].strip('.,!?') if i > 0 else 'start'
                    category = ()
                    if word in self.prev_probabilities[prev_word]:
                        category = abs(i - self.prev_probabilities[prev_word][word])

                    if i > 0 and not self._is_valid_word(prev_word):
                        prev_word = 'start'
                    
                    self.words[prev_word][category][word] += 1
                    self.context_window.add(word)
            if self._compare_probabilities():
                self._calculate_transition_probabilities()
            self.is_converged = self._compare_probabilities()
            self.total_epochs += 1
            
            if self.total_epochs % 10 == 0:
                print(f"Epoch {self.total_epochs}")
                
        print(f"\nTraining completed after {self.total_epochs} epochs")
        print(f"Converged: {self.is_converged}")
        return epoch_diffs

    def generate_text(self, num_words: int = 50) -> str:
        """Generate text based on learned probabilities."""
        generated_words = []
        self.context_window.clear()
        
        input_text = input("Enter starting words (space-separated): ").lower().strip()
        initial_words = [word.strip('.,!?') for word in input_text.split() 
                        if self._is_valid_word(word)]
        
        if not initial_words:
            initial_words = ['start']
        
        self.context_window.add_multiple(initial_words)
        generated_words.extend(initial_words)
        
        current_word = initial_words[-1] if initial_words else 'start'
        attempts = 0
        max_attempts = num_words * 2
        
        while len(generated_words) < num_words and attempts < max_attempts:
            attempts += 1
            
            context = self.context_window.get_context()
            if context in self.context_transitions and self.context_transitions[context]:
                # Use context-based transitions
                candidates = [(word, count) for word, count in self.context_transitions[context].items() 
                            if self._is_valid_word(word)]
                if candidates:
                    words, counts = zip(*candidates)
                    next_word = random.choices(words, weights=counts)[0]
                else:
                    continue
            else:
                # Use word-based transitions
                if current_word not in self.words:
                    valid_words = [word for word in generated_words[-3:] 
                                 if self._is_valid_word(word)]
                    current_word = random.choice(valid_words if valid_words else ['start'])
                    continue
                
                categories = []
                weights = []
                
                for category, word_counts in self.words[current_word].items():
                    if word_counts:
                        total_count = sum(word_counts.values())
                        categories.append(category)
                        weights.append(total_count)
                
                if not categories:
                    current_word = random.choice([word for word in generated_words[-3:] 
                                               if self._is_valid_word(word)] or ['start'])
                    continue
                
                category = random.choices(categories, weights=weights)[0]
                
                valid_words = [(word, count) for word, count in self.words[current_word][category].items()
                             if self._is_valid_word(word)]
                
                if not valid_words:
                    continue
                    
                words, counts = zip(*valid_words)
                next_word = random.choices(words, weights=counts)[0]
            
            if next_word != (generated_words[-1] if generated_words else ''):
                generated_words.append(next_word)
                self.context_window.add(next_word)
                current_word = next_word
            
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
        """Save the model state to a file."""
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
                'transition_probabilities': dict(self.transition_probabilities),
                'prev_probabilities': dict(self.prev_probabilities),
                'decay_rate': self.decay_rate,
                'probability_threshold': self.probability_threshold,
                'is_converged': self.is_converged,
                'total_epochs': self.total_epochs,
                'context_size': self.context_size
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(model_state, f)
            
        except Exception as e:
            raise Exception(f"Failed to save model: {str(e)}")

    def load_model(self, filepath: str):
        """Load the model state from a file."""
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
            
            self.transition_probabilities = defaultdict(dict, model_state['transition_probabilities'])
            self.prev_probabilities = defaultdict(dict, model_state['prev_probabilities'])
            self.decay_rate = model_state['decay_rate']
            self.probability_threshold = model_state['probability_threshold']
            self.is_converged = model_state['is_converged']
            self.total_epochs = model_state['total_epochs']
            self.context_size = model_state['context_size']
            self.context_window = ContextWindow(self.context_size)
            
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")

def main():
    print("Probability-Based Context-Aware Semantic Text Generator")
    print("================================================")
    
    generator = ErrorAwareSemanticGenerator(
        decay_rate=0.95,
        probability_threshold=0.01,
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
                generator.train_until_convergence(text)
                
                model_path = os.path.join('models', 'auto_saved_model.pkl')
                generator.save_model(model_path)
                print(f"Model automatically saved to {model_path}")
            except FileNotFoundError:
                print("Error: Training file not found!")
            except Exception as e:
                print(f"Error during training: {e}")
                
        elif choice == "2":
            while True:
                num_words = 250
                generated_text = generator.generate_text(num_words)
                print("\nGenerated text:")
                print(generated_text)
       
                
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
