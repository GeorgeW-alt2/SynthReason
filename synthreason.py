# SynthReason Version 10.0
import numpy as np
import random
import re
import pickle
import math
import os
import time
from datasets import load_dataset
from collections import defaultdict, Counter, deque
from typing import List, Tuple, Dict, Any, Optional, Deque

KB_limit = 10000000
STAGE0 = -1
out_length = 250

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
        self.context_size = context_size
        
        # Enhanced context window initialization with two configurations
        self.standard_context_window = ContextWindow(
            block_size=50,
            num_blocks=70,
            num_layers=3,
            layer_depth=2
        )
        
        self.high_dim_context_window = ContextWindow(
            block_size=120,
            num_blocks=110,
            num_layers=7,
            layer_depth=3
        )
        
        # Use standard window by default
        self.context_window = self.standard_context_window
        
        # Layer-specific transition tracking
        self.layer_transitions = [defaultdict(Counter) for _ in range(7)]  # Increased to 7 layers
        self.semantic_categories = defaultdict(str)

    def train_until_convergence(self, text: str, max_epochs: int = 10) -> List[float]:
        """Train the model with two phases: standard and high dimensionality."""
        text = self.clean_text(text)
        words = text.split()
        
        # Phase 1: 
        phase1_words = ' '.join(words[:STAGE0])
        phase1_sentences = phase1_words.split('.')
        
        print(f"\nTraining")
        progress1 = ProgressBar(max_epochs, prefix='Phase 1:', suffix='Complete', length=50)
        
        # Use standard context window for phase 1
        self.context_window = self.standard_context_window
        
        # Train phase 1
        for epoch in range(max_epochs):
            # Update to show current progress as percentage of max_epochs
            progress1.print(epoch + 1)  # Changed from epoch to epoch + 1
            
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
                
            # Phase 2: 
            phase2_words = ' '.join(words[STAGE0:-1])
            phase2_sentences = phase2_words.split()
            
            # Switch to high dimensionality context window for phase 2
            self.context_window = self.high_dim_context_window
            
            # Train phase 2
            for epoch in range(max_epochs):
                
                for sentence in phase2_sentences:
                    words = sentence.lower().split()
                    
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

                            self.context_window.update_layer_connection(2, 3, connection_strength)

                            self.context_window.update_layer_connection(3, 4, connection_strength)

                            self.context_window.update_layer_connection(4, 5, connection_strength)

                            self.context_window.update_layer_connection(5, 6, connection_strength)

                            self.context_window.update_layer_connection(6, 7, connection_strength)

                
                self._calculate_transition_probabilities()
                self.is_converged = self._compare_probabilities()
            
        # Ensure progress bar shows 100% at completion
        
        print(f"\nTraining complete.")
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
        
        return relationship_matrix.get((category1, category2), 0.9)
        
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
        print()
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
        
        generator.is_converged = model_state['is_converged']
        
        generator.layer_transitions = []
        for lt in model_state['layer_transitions']:
            layer_trans = defaultdict(Counter)
            layer_trans.update(lt)
            generator.layer_transitions.append(layer_trans)
            
        generator.semantic_categories = defaultdict(str)
        generator.semantic_categories.update(model_state['semantic_categories'])
        
        print(f"\nModel loaded from models/{filename}.pkl")
        return True
    except FileNotFoundError:
        print(f"\nNo model file found at models/{filename}.pkl")
        return False
    except Exception as e:
        print(f"\nError loading model: {str(e)}")
        return False

def download_huggingface_dataset(dataset_name: str, subset: str = None, split: str = 'train', 
                               text_column: str = 'text', num_examples: int = 1000) -> str:
    """
    Download and process text from a Hugging Face dataset.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face
        subset: Optional subset/configuration of the dataset
        split: Which split to use ('train', 'test', 'validation')
        text_column: Name of the column containing the text
        num_examples: Number of examples to download
    
    Returns:
        str: Combined text from the dataset
    """
    try:
        # Load dataset
        if subset:
            dataset = load_dataset(dataset_name, subset, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        
        # Limit to specified number of examples
        if len(dataset) > num_examples:
            dataset = dataset.select(range(num_examples))
        
        # Extract text from the specified column
        texts = dataset[text_column]
        
        # Filter out empty texts and join
        texts = [text for text in texts if text and isinstance(text, str)]
        return "\n\n".join(texts)
        
    except Exception as e:
        raise Exception(f"Error downloading dataset {dataset_name}: {str(e)}")

def download_huggingface_datasets(output_dir: str = 'training_data',
                                progress_callback = None) -> None:
    """
    Download multiple curated datasets from Hugging Face.
    """
    # Dictionary of dataset configurations
    datasets = {
        'wikitext': {
            'name': 'wikitext',
            'subset': 'wikitext-103-v1',
            'text_column': 'text',
            'num_examples': KB_limit
        }
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (dataset_key, config) in enumerate(datasets.items(), 1):
        try:
            print(f"\nDownloading {dataset_key} dataset...")
            text = download_huggingface_dataset(
                config['name'],
                subset=config.get('subset'),
                text_column=config['text_column'],
                num_examples=config['num_examples']
            )
            
            # Save to file
            output_file = os.path.join(output_dir, f"{dataset_key}.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            if progress_callback:
                progress_callback(i)
            
            # Small delay between downloads
            time.sleep(1)
            
        except Exception as e:
            print(f"Error downloading {dataset_key}: {str(e)}")
            continue        
def main():
    """Main function for the Text Generator application."""
    generator = ErrorAwareSemanticGenerator(
        decay_rate=0.95,
        probability_threshold=0.01,
        context_size=15
    )
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('training_data', exist_ok=True)
    
    menu_options = {
        '1': 'Train model',
        '2': 'Generate text',
        '3': 'Save model',
        '4': 'Load model',
        '5': 'Download datasets',
        '6': 'View model stats',
        '7': 'Exit'
    }
    
    print("\nProbability-Based Context-Aware Semantic Text Generator")
    print("================================================")
    
    while True:
        try:
            print("\nOptions:")
            for key, value in menu_options.items():
                print(f"{key}. {value}")
                
            choice = input("\nEnter choice (1-7): ").strip()
            
            if choice == '1':  # Train model
                print("\nSelect training source:")
                print("1. Local file")
                print("2. Downloaded dataset")
                source = input("Choose source (1-2): ").strip()
                
                if source == '1':
                    filepath = input("Enter filepath: ").strip()
                    if not os.path.exists(filepath):
                        print("Error: File not found")
                        continue
                        
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = ' '.join(f.read().split()[:KB_limit])
                        
                elif source == '2':
                    data_dir = 'training_data'
                    datasets = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
                    
                    if not datasets:
                        print("No datasets found. Please download them first.")
                        continue
                        
                    print("\nAvailable datasets:")
                    for idx, dataset in enumerate(datasets, 1):
                        print(f"{idx}. {dataset[:-4]}")
                        
                    try:
                        dataset_idx = int(input("Choose dataset number: "))
                        if not 1 <= dataset_idx <= len(datasets):
                            print("Invalid selection")
                            continue
                            
                        filepath = os.path.join(data_dir, datasets[dataset_idx-1])
                        with open(filepath, 'r', encoding='utf-8') as f:
                            text = ' '.join(f.read().split()[:KB_limit])
                    except ValueError:
                        print("Please enter a valid number")
                        continue
                else:
                    print("Invalid choice")
                    continue
                
                print("\nStarting training...")
                generator.train_until_convergence(text)
                
            elif choice == '2':  # Generate text
                if not generator.words:
                    print("Error: Model not trained. Please train or load a model first.")
                    continue
                    
                while True:
                    try:
                        
                        print("\nGenerated text:")
                        print("-" * 50)
                        print(generator.generate_text(out_length))
                        print("-" * 50)
                        
                       
                            
                    except ValueError:
                        print("Please enter a valid number")
                        
            elif choice == '3':  # Save model
                if not generator.words:
                    print("Error: No model to save. Please train a model first.")
                    continue
                    
                name = input("Enter model name: ").strip()
                if name:
                    save_model(generator, name)
                else:
                    print("Invalid model name")
                    
            elif choice == '4':  # Load model
                models = [f[:-4] for f in os.listdir('models') if f.endswith('.pkl')]
                
                if not models:
                    print("No saved models found")
                    continue
                    
                print("\nAvailable models:")
                for idx, model in enumerate(models, 1):
                    print(f"{idx}. {model}")
                    
                try:
                    model_idx = int(input("Choose model number: "))
                    if not 1 <= model_idx <= len(models):
                        print("Invalid selection")
                        continue
                        
                    load_model(generator, models[model_idx-1])
                except ValueError:
                    print("Please enter a valid number")
                    
            elif choice == '5':  # Download datasets
                print("\nDownloading options:")
                print("1. Download all datasets")
                print("2. Download specific dataset")
                download_choice = input("Choose option (1-2): ").strip()
                
                if download_choice == '1':
                    progress = ProgressBar(5, prefix='Downloading:', suffix='Complete', length=50)
                    download_huggingface_datasets(progress_callback=progress.print)
                    
                elif download_choice == '2':
                    datasets = ['wikitext', 'bookcorpus', 'oscar', 'c4', 'pile']
                    print("\nAvailable datasets:")
                    for idx, dataset in enumerate(datasets, 1):
                        print(f"{idx}. {dataset}")
                        
                    try:
                        dataset_idx = int(input("Choose dataset number: "))
                        if not 1 <= dataset_idx <= len(datasets):
                            print("Invalid selection")
                            continue
                            
                        dataset = datasets[dataset_idx-1]
                        progress = ProgressBar(1, prefix='Downloading:', suffix='Complete', length=50)
                        
                        if dataset == 'wikitext':
                            text = download_huggingface_dataset('wikitext', 'wikitext-103-v1')
                        elif dataset == 'bookcorpus':
                            text = download_huggingface_dataset('bookcorpus')
                        elif dataset == 'oscar':
                            text = download_huggingface_dataset('oscar', 'unshuffled_deduplicated_en')
                        elif dataset == 'c4':
                            text = download_huggingface_dataset('c4', 'en')
                        elif dataset == 'pile':
                            text = download_huggingface_dataset('the_pile')
                            
                        with open(f'training_data/{dataset}.txt', 'w', encoding='utf-8') as f:
                            f.write(text)
                            
                        progress.print(1)
                        print(f"\n{dataset} dataset downloaded successfully!")
                        
                    except ValueError:
                        print("Please enter a valid number")
                else:
                    print("Invalid choice")
                    
            elif choice == '6':  # View model stats
                if not generator.words:
                    print("Error: No model loaded. Please train or load a model first.")
                    continue
                    
                unique_words = set()
                for word_dict in generator.words.values():
                    for category_dict in word_dict.values():
                        unique_words.update(category_dict.keys())
                        
                print("\nModel Statistics")
                print("-" * 20)
                print(f"Unique words: {len(unique_words)}")
                print(f"Context transitions: {len(generator.context_transitions)}")
                print(f"Model converged: {generator.is_converged}")
                
                # Show top words by category
                categories = ['verbs', 'nouns', 'adverbs', 'determiners', 'prepositions']
                print("\nTop 5 words by category:")
                for category in categories:
                    words = []
                    for word_dict in generator.words.values():
                        if category in word_dict:
                            words.extend(word_dict[category].items())
                            
                    if words:
                        sorted_words = sorted(words, key=lambda x: x[1], reverse=True)[:5]
                        print(f"\n{category.capitalize()}:")
                        for word, count in sorted_words:
                            print(f"  {word}: {count}")
                            
                input("\nPress Enter to continue...")
                
            elif choice == '7':  # Exit
                print("\nGoodbye!")
                break
                
            else:
                print("Invalid choice")
                
        except KeyboardInterrupt:
            print("\nOperation cancelled")
            continue
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            continue

if __name__ == "__main__":
    main()
