
# SynthReason Version 9.0
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

KB_limit = 1000000
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
            

class ErrorAwareSemanticGenerator:
    def __init__(self, decay_rate: float = 0.95, probability_threshold: float = 0.01):
        self.words = defaultdict(lambda: defaultdict(Counter))
        self.prev_probabilities = defaultdict(dict)
        self.decay_rate = decay_rate
        self.probability_threshold = probability_threshold
        
        # Use standard window by default
        

    def train_until_convergence(self, text: str, max_epochs: int = 10) -> List[float]:
        """Train the model with two phases: standard and high dimensionality."""
        text = self.clean_text(text)
        words = text.split()
        
        # Phase 1: 
        phase1_words = ' '.join(words[:STAGE0])
        phase1_sentences = phase1_words.split(".")
        
        print(f"\nTraining")
        progress1 = ProgressBar(max_epochs, prefix='Phase 1:', suffix='Complete', length=50)
        
        # Use standard context window for phase 1
        
        # Train phase 1
        for epoch in range(max_epochs):
            # Update to show current progress as percentage of max_epochs
            progress1.print(epoch + 1)  # Changed from epoch to epoch + 1
            
            for sentence in phase1_sentences:
                words = sentence.lower().split()

                for i, word in enumerate(words):
                    word = word.strip('.,!?')
                    if not word or not self._is_valid_word(word):
                        continue
                        
                    prev_word = words[i-1].strip('.,!?') if i > 0 else 'start'
                    
                    if i > 0 and not self._is_valid_word(prev_word):
                        prev_word = 'start'
                    category = self._categorize_word(prev_word)

                    self.words[prev_word][category][word] += 1
                    temp = self.words[prev_word][category][word]   

                    phase2_words = ' '.join(words[STAGE0:-1])
                    phase2_sentences = phase2_words.split()
                    
                    for epoch2 in range(max_epochs):
                        
                        for sentence in phase2_sentences:
                            words = sentence.lower().split()
                            
                            for i, word in enumerate(words[::3]):
                                word = word.strip('.,!?')
                                if not word or not self._is_valid_word(word):
                                    continue

                                prev_word = words[i-1].strip('.,!?') if i > 0 else 'start'
                                category = self._categorize_word(word)
                                
                                if i > 0 and not self._is_valid_word(prev_word):
                                    prev_word = 'start'
                                
                                self.words[prev_word][category][word] += 1
        print(f"\nTraining complete.")
        
        return []
        
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

    def _is_valid_word(self, word):
        """Check if a word is valid (extend as needed)."""
        return word.isalpha()

    def generate_text(self, num_words: int = 50) -> str:
        """Generate text with autoisomorphic composure."""
        generated_words = []
        print()
        input_text = input("Enter starting words (space-separated): ").lower().strip()
        initial_words = [word.strip('.,!?') for word in input_text.split() if self._is_valid_word(word)]
        
        if not initial_words:
            initial_words = ['start']

        current_word = initial_words[-1]
        attempts = 0
        max_attempts = num_words * 2
        word_patterns = {}  # Memory for word recurrence
        lambda_factor = 0.5  # Initial influence of self-similarity

        while len(generated_words) < num_words and attempts < max_attempts:
            attempts += 1

            if current_word not in self.words:
                valid_words = [word for word in generated_words[-3:] if self._is_valid_word(word)]
                current_word = random.choice(valid_words if valid_words else ['start'])
                continue

            categories = list(self.words[current_word].keys())
            if not categories:
                continue

            # Compute category selection probabilities
            category = random.choices(categories, 
                                      weights=[sum(self.words[current_word][cat].values()) for cat in categories])[-1]

            valid_words = [(word, count) for word, count in self.words[current_word][category].items()
                           if self._is_valid_word(word)]
            if not valid_words:
                continue

            words, counts = zip(*valid_words)
            
            # Compute autoisomorphic recurrence weight
            recurrence_weights = [self._compute_recurrence_weight(word, generated_words, lambda_factor) for word in words]
            
            # Compute final selection probabilities
            combined_weights = [c + r for c, r in zip(counts, recurrence_weights)]
            next_word = random.choices(words, weights=combined_weights)[-1]

            # Store word occurrence pattern
            if current_word not in word_patterns:
                word_patterns[current_word] = []
            word_patterns[current_word].append(next_word)

            if next_word != (generated_words[-1] if generated_words else ''):
                generated_words.append(next_word)
                current_word = next_word

            # Dynamically adjust lambda based on text progression
            lambda_factor = min(2.0, lambda_factor * 1.05)  # Increase self-similarity influence

        return ' '.join(generated_words)

    def _compute_recurrence_weight(self, word, generated_words, lambda_factor):
        """Compute the recurrence weight for a word."""
        weight = 0.0
        for i, prev_word in enumerate(generated_words):
            if word != prev_word:
                decay = math.exp(-0.1 * (len(generated_words) - i))
                weight += lambda_factor * decay
        return weight
    def _calculate_weight(self, epoch, max_epochs):
        """Calculates the weight for the current epoch.  You can customize this."""
        # Example: Linear decay
        return (max_epochs - epoch) / max_epochs  # Weight decreases with each epoch

        # Example: Exponential decay
        # return math.exp(-epoch / max_epochs)

        # Example: No weight (all epochs equal)
        # return 1
def save_model(generator, filename):
    """Save the model state to a file."""
    model_state = {
        'words': dict(generator.words),
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
        probability_threshold=0.01
    )
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('training_data', exist_ok=True)
    
    menu_options = {
        '1': 'Train model',
        '2': 'Generate text',
        '3': 'Save model',
        '4': 'Load model',
        '5': 'Download datasets',
        '6': 'Exit'
    }
    
    print("\nProbability-Based Context-Aware Semantic Text Generator")
    print("================================================")
    
    while True:
        try:
            print("\nOptions:")
            for key, value in menu_options.items():
                print(f"{key}. {value}")
                
            choice = input("\nEnter choice (1-6): ").strip()
            
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
                    datasets = ['wikitext']
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
    
                        
                        with open(f'training_data/{dataset}.txt', 'w', encoding='utf-8') as f:
                            f.write(text)
                            
                        progress.print(1)
                        print(f"\n{dataset} dataset downloaded successfully!")
                        
                    except ValueError:
                        print("Please enter a valid number")
                else:
                    print("Invalid choice")
                    
            elif choice == '6':  # Exit
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
