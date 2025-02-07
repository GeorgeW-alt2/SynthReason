# SynthReason Version 7.0
import numpy as np
import random
import re
import pickle
import math
import os
from collections import defaultdict, Counter, deque
from typing import List, Tuple, Dict, Any, Optional, Deque

KB_limit = -1 # -1 for unlimited
STAGE0 = 1000
STAGE1 = 1000000
class ProgressBar:
    def __init__(self, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.iteration = 0

    def print(self, iteration):
        self.iteration = iteration
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (iteration / float(self.total)))
        filled_length = int(self.length * iteration // self.total)
        bar = self.fill * filled_length + '-' * (self.length - filled_length)
        print(f'\r{self.prefix} |{bar}| {percent}% {self.suffix}', end='')
        if iteration == self.total:
            print()
            
class ContextWindow:
    def __init__(self, block_size: int, num_blocks: int, num_layers: int = 3, layer_depth: int = 2):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.layer_depth = layer_depth
        
        # Initialize 3D window structure: [layers][blocks][words]
        self.window = [[[''] * block_size for _ in range(num_blocks)] for _ in range(num_layers)]
        
        # Tracking current position in each layer
        self.current_positions = [(0, 0) for _ in range(num_layers)]  # (block, index)
        
        # Track semantic relationships between layers
        self.layer_connections = defaultdict(lambda: defaultdict(float))

    def add(self, word: str, layer: int = 0):
        if not (0 <= layer < self.num_layers):
            print(f"Error: layer {layer} out of bounds.")
            return
            
        current_block, current_index = self.current_positions[layer]
        
        # Add word to current position
        self.window[layer][current_block][current_index] = word
        
        # Update position
        current_index += 1
        if current_index >= self.block_size:
            current_index = 0
            current_block = (current_block + 1) % self.num_blocks
            
        self.current_positions[layer] = (current_block, current_index)
        
        # Propagate to higher layers based on semantic relationships
        if layer < self.num_layers - 1:
            self._propagate_to_higher_layers(word, layer)

    def _propagate_to_higher_layers(self, word: str, current_layer: int):
        for higher_layer in range(current_layer + 1, min(current_layer + self.layer_depth, self.num_layers)):
            connection_strength = self.layer_connections[current_layer][higher_layer]
            if connection_strength > 0.5:  # Threshold for propagation
                self.add(word, higher_layer)

    def add_multiple(self, words: list, layer: int = 0):
        for word in words:
            self.add(word, layer)

    def get_context(self, layer: int = None) -> str:
        if layer is not None:
            if not (0 <= layer < self.num_layers):
                return ""
            flat_layer = [word for block in self.window[layer] for word in block if word]
            return ' '.join(flat_layer)
        
        # Combine context from all layers with decreasing weight
        contexts = []
        for l in range(self.num_layers):
            layer_context = self.get_context(l)
            if layer_context:
                weight = 1.0 / (l + 1)  # Higher layers have less weight
                contexts.append((layer_context, weight))
        
        return ' '.join(context for context, _ in contexts)

    def get_layer_context(self, start_layer: int, depth: int) -> str:
        if not (0 <= start_layer < self.num_layers):
            return ""
            
        max_depth = min(depth, self.num_layers - start_layer)
        contexts = []
        
        for l in range(start_layer, start_layer + max_depth):
            layer_context = self.get_context(l)
            if layer_context:
                contexts.append(layer_context)
                
        return ' '.join(contexts)

    def clear(self, layer: int = None):
        if layer is not None:
            if 0 <= layer < self.num_layers:
                self.window[layer] = [[''] * self.block_size for _ in range(self.num_blocks)]
                self.current_positions[layer] = (0, 0)
            return
            
        self.window = [[[''] * self.block_size for _ in range(self.num_blocks)] for _ in range(self.num_layers)]
        self.current_positions = [(0, 0) for _ in range(self.num_layers)]
        self.layer_connections.clear()

    def update_layer_connection(self, layer1: int, layer2: int, strength: float):
        if 0 <= layer1 < self.num_layers and 0 <= layer2 < self.num_layers:
            self.layer_connections[layer1][layer2] = max(0.0, min(1.0, strength))

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
        self.context_size = context_size
        
        # Enhanced context window initialization with two configurations
        self.standard_context_window = ContextWindow(
            block_size=100,
            num_blocks=50,
            num_layers=3,
            layer_depth=2
        )
        
        self.high_dim_context_window = ContextWindow(
            block_size=100,
            num_blocks=150,
            num_layers=7,
            layer_depth=3
        )
        
        # Use standard window by default
        self.context_window = self.standard_context_window
        
        # Layer-specific transition tracking
        self.layer_transitions = [defaultdict(Counter) for _ in range(7)]  # Increased to 7 layers
        self.semantic_categories = defaultdict(str)

    def train_until_convergence(self, text: str, max_epochs: int = 1000) -> List[float]:
        """Train the model with two phases: standard and high dimensionality."""
        text = self.clean_text(text)
        words = text.split()
        
        # Phase 1: 
        phase1_words = ' '.join(words[:STAGE0])
        phase1_sentences = phase1_words.split('.')
        
        print(f"\nPhase 1: Training first {STAGE0} words with standard dimensionality...")
        progress1 = ProgressBar(max_epochs, prefix='Phase 1:', suffix='Complete', length=50)
        
        # Use standard context window for phase 1
        self.context_window = self.standard_context_window
        
        # Train phase 1
        for epoch in range(max_epochs):
            progress1.print(epoch)
            
            for sentence in phase1_sentences:
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
                    category = self._categorize_word(word)
                    
                    if i > 0 and not self._is_valid_word(prev_word):
                        prev_word = 'start'
                    
                    self.words[prev_word][category][word] += 1
                    self.context_window.add(word)
                    
            self._calculate_transition_probabilities()
            
        progress1.print(max_epochs)
        print("\nPhase 1 completed")
        
        # Phase 2: 
        phase2_words = ' '.join(words[STAGE0:STAGE1])
        phase2_sentences = phase2_words.split('.')
        
        print(f"\nPhase 2: Training next {STAGE1} words with higher dimensionality...")
        progress2 = ProgressBar(max_epochs, prefix='Phase 2:', suffix='Complete', length=50)
        
        # Switch to high dimensionality context window for phase 2
        self.context_window = self.high_dim_context_window
        
        # Train phase 2
        for epoch in range(max_epochs):
            progress2.print(epoch)
            
            for sentence in phase2_sentences:
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
                    category = self._categorize_word(word)
                    
                    if i > 0 and not self._is_valid_word(prev_word):
                        prev_word = 'start'
                    
                    # Add words to multiple layers for higher dimensionality
                    self.words[prev_word][category][word] += 1
                    for layer in range(min(3, len(self.layer_transitions))):
                        self.context_window.add(word, layer)
                        
                    # Calculate semantic connections between layers
                    if i > 0:
                        prev_category = self._categorize_word(prev_word)
                        connection_strength = self._calculate_semantic_connection(prev_category, category)
                        self.context_window.update_layer_connection(0, 1, connection_strength)
                        self.context_window.update_layer_connection(1, 2, connection_strength)
            
            self._calculate_transition_probabilities()
            self.is_converged = self._compare_probabilities()
            
        progress2.print(max_epochs)
        print(f"\nTraining completed after {max_epochs * 2} total epochs")
        print(f"Converged: {self.is_converged}")
        
        # Reset to high dimensionality context window for future use
        self.context_window = self.high_dim_context_window
        return []
    
    def _calculate_semantic_connection(self, category1: str, category2: str) -> float:
        # Define semantic relationship strengths between categories
        relationship_matrix = {
            ('verbs', 'adverbs'): 0.8,
            ('determiners', 'nouns'): 0.7,
            ('prepositions', 'nouns'): 0.6
        }
        
        return relationship_matrix.get((category1, category2), 0.3)
        
    def _format_generated_text(self, words: List[str]) -> str:
        text = ' '.join(words)
        sentences = text.split('.')
        formatted_sentences = []
        
        for sentence in sentences:
            if sentence.strip():
                formatted = sentence.strip()
                formatted = formatted[0].upper() + formatted[1:] + '.'
                formatted_sentences.append(formatted)
        
        return ' '.join(formatted_sentences)
        
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
                        self.context_window.add_multiple(category)

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
        
def save_model(generator, filename):
    """Save the model state to a file."""
    model_state = {
        'words': dict(generator.words),
        'context_transitions': dict(generator.context_transitions),
        'transition_probabilities': dict(generator.transition_probabilities),
        'prev_probabilities': dict(generator.prev_probabilities),
        'total_epochs': generator.total_epochs,
        'is_converged': generator.is_converged,
        'layer_transitions': [dict(lt) for lt in generator.layer_transitions],
        'semantic_categories': dict(generator.semantic_categories)
    }
    
    with open(f'models/{filename}.pkl', 'wb') as f:
        pickle.dump(model_state, f)
    print(f"\nModel saved to models/{filename}.pkl")

def load_model(generator, filename):
    """Load the model state from a file."""
    try:
        with open(f'models/{filename}.pkl', 'rb') as f:
            model_state = pickle.load(f)
            
        generator.words = defaultdict(lambda: defaultdict(Counter))
        generator.words.update(model_state['words'])
        
        generator.context_transitions = defaultdict(Counter)
        generator.context_transitions.update(model_state['context_transitions'])
        
        generator.transition_probabilities = defaultdict(dict)
        generator.transition_probabilities.update(model_state['transition_probabilities'])
        
        generator.prev_probabilities = defaultdict(dict)
        generator.prev_probabilities.update(model_state['prev_probabilities'])
        
        generator.total_epochs = model_state['total_epochs']
        generator.is_converged = model_state['is_converged']
        
        generator.layer_transitions = []
        for lt in model_state['layer_transitions']:
            layer_trans = defaultdict(Counter)
            layer_trans.update(lt)
            generator.layer_transitions.append(layer_trans)
            
        generator.semantic_categories = defaultdict(str)
        generator.semantic_categories.update(model_state['semantic_categories'])
        
        print(f"\nModel loaded from models/{filename}.pkl")
        print(f"Model state: {generator.total_epochs} epochs, converged: {generator.is_converged}")
        return True
    except FileNotFoundError:
        print(f"\nNo model file found at models/{filename}.pkl")
        return False
    except Exception as e:
        print(f"\nError loading model: {str(e)}")
        return False
        
def main():
    print("Probability-Based Context-Aware Semantic Text Generator")
    print("================================================")
    
    # Initialize generator with optimized parameters
    generator = ErrorAwareSemanticGenerator(
        decay_rate=0.95,
        probability_threshold=0.01,
        context_size=15
    )
    
    # Create models directory if it doesn't exist
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
                filename = input("Enter training file path: ")
                with open(filename, 'r', encoding='utf-8') as f:
                    text = ' '.join(f.read().split()[:KB_limit])
                print("\nTraining model...")
                generator.train_until_convergence(text)
            except FileNotFoundError:
                print(f"Error: File '{filename}' not found")
            except Exception as e:
                print(f"Error during training: {str(e)}")
                
        elif choice == "2":
            try:
                while True:
                    num_words = 250
                    generated_text = generator.generate_text(num_words)
                    print("\nGenerated text:")
                    print("AI:",generated_text)
            except Exception as e:
                print(f"Error generating text: {str(e)}")
                
        elif choice == "3":
            try:
                model_name = input("Enter model name to save: ")
                save_model(generator, model_name)
            except Exception as e:
                print(f"Error saving model: {str(e)}")
                
        elif choice == "4":
            try:
                model_name = input("Enter model name to load: ")
                load_model(generator, model_name)
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                
        elif choice == "5":
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()
