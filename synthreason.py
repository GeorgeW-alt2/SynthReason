import numpy as np
import random
import re
import pickle
import math
import os
import time
from typing import List, Tuple, Dict, Optional, Any, Deque
from collections import defaultdict, Counter, deque

# Constants
STAGE0 = 1000
OUT_LENGTH = 250

class ProgressBar:
    def __init__(self, total: int, prefix: str = '', suffix: str = '', decimals: int = 1, 
                 length: int = 50, fill: str = '█'):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        
    def print(self, iteration: int):
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (iteration / float(self.total)))
        filled_length = int(self.length * iteration // self.total)
        bar = self.fill * filled_length + '-' * (self.length - filled_length)
        print(f'\r{self.prefix} |{bar}| {percent}% {self.suffix}', end='\r')
        if iteration == self.total:
            print()

class HybridTextGenerator:
    def __init__(self, temperature: float = 0.7, memory_length: int = 5,
                 decay_rate: float = 0.95, probability_threshold: float = 0.01):
        """Initialize the hybrid text generator combining neural and semantic approaches.
        
        Args:
            temperature (float): Controls randomness in generation (0.0-1.0)
            memory_length (int): Number of previous words to consider for context
            decay_rate (float): Rate at which probabilities decay over time
            probability_threshold (float): Minimum probability threshold
        """
        # Neural components
        self.temperature = temperature
        self.memory_length = memory_length
        self.attention_weights = {}
        self.context_memory = []
        
        # Semantic components
        self.words = defaultdict(lambda: defaultdict(Counter))
        self.prev_probabilities = defaultdict(dict)
        self.decay_rate = decay_rate
        self.probability_threshold = probability_threshold
        self.word_categories = {}

    def clean_text(self, text: str) -> str:
        """Clean and normalize input text."""
        # Remove URLs and emails
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Convert to lowercase and normalize whitespace
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize sentence endings and punctuation
        text = re.sub(r'([.!?])\s*([A-Za-z])', r'\1 \2', text)
        text = re.sub(r'[^A-Za-z0-9\s.,!?\'-]', '', text)
        
        # Ensure proper sentence ending
        if text and text[0].isalpha():
            text = text[0].upper() + text[1:]
        if text and text[-1] not in '.!?':
            text += '.'
            
        return text

    def _is_valid_word(self, word: str) -> bool:
        """Check if a word is valid using enhanced validation rules."""
        valid_one_letter = ['a', 'i']
        valid_two_letter = [
            'am', 'an', 'as', 'at', 'be', 'by', 'do', 'go', 'he', 'hi',
            'if', 'in', 'is', 'it', 'me', 'my', 'no', 'of', 'on', 'or',
            'so', 'to', 'up', 'us', 'we'
        ]
        
        clean_word = word.strip('.,!?')
        if not clean_word:
            return False
            
        if len(clean_word) == 1:
            return clean_word.lower() in valid_one_letter
        elif len(clean_word) == 2:
            return clean_word.lower() in valid_two_letter
            
        return any(c.isalnum() for c in clean_word)

    def _categorize_word(self, word: str) -> str:
        """Categorize words using both semantic and syntactic features."""
        if word not in self.word_categories:
            if word == 'start':
                category = 'START'
            elif word.endswith('ly'):
                category = 'ADVERB'
            elif word.endswith('ed') or word.endswith('ing'):
                category = 'VERB'
            elif word in ['the', 'a', 'an', 'this', 'that', 'these', 'those']:
                category = 'DETERMINER'
            elif word in ['in', 'on', 'at', 'to', 'from', 'with', 'by']:
                category = 'PREPOSITION'
            elif any(c.isdigit() for c in word):
                category = 'NUMERIC'
            elif word.istitle():
                category = 'TITLE'
            elif all(c.isupper() for c in word if c.isalpha()):
                category = 'UPPERCASE'
            else:
                category = 'STANDARD'
                
            self.word_categories[word] = category
            
        return self.word_categories[word]

    def train_until_convergence(self, text: str, max_epochs: int = 10) -> List[float]:
        """Train the model using both neural and semantic approaches."""
        losses = []
        text = self.clean_text(text)
        words = text.split()
        
        # Prepare data for both phases
        phase1_words = ' '.join(words[:STAGE0])
        phase2_words = ' '.join(words[STAGE0:-1])
        
        phase1_sentences = [s.strip() for s in phase1_words.split(".") if s.strip()]
        phase2_sentences = [s.strip() for s in phase2_words.split(".") if s.strip()]
        
        print("\nTraining")
        progress1 = ProgressBar(max_epochs, prefix='Phase 1:', suffix='Complete', length=50)
        progress2 = ProgressBar(max_epochs, prefix='Phase 2:', suffix='Complete', length=50)
        
        # Phase 1: Standard training with neural attention
        epoch_loss = 0.0
        for epoch in range(max_epochs):
            progress1.print(epoch + 1)
            
            for sentence in phase1_sentences:
                words = sentence.lower().split()
                epoch_loss += self._process_sentence(words)
                
            losses.append(epoch_loss)
            epoch_loss = 0.0
            
        print("\nPhase 1 complete. Starting Phase 2...")
        
        # Phase 2: High dimensionality training with semantic weighting
        for epoch in range(max_epochs):
            progress2.print(epoch + 1)
            
            for sentence in phase2_sentences:
                words = sentence.lower().split()
                epoch_loss += self._process_sentence(words, phase=2)
                
            losses.append(epoch_loss)
            epoch_loss = 0.0
            
        print("\nTraining complete.")
        return losses



    def generate_text(self, num_words: int = 50) -> str:
        """Generate text using combined neural and semantic approaches."""
        generated_words = []
        print()
        input_text = input("Enter starting words (space-separated): ").lower().strip()
        initial_words = [word.strip('.,!?') for word in input_text.split() 
                        if self._is_valid_word(word)]
        
        if not initial_words:
            initial_words = ['start']
            
        self.context_memory = initial_words[-self.memory_length:]
        current_word = initial_words[-1] if initial_words else 'start'
        
        attempts = 0
        max_attempts = num_words * 2
        
        while len(generated_words) < num_words and attempts < max_attempts:
            attempts += 1
            
            if current_word not in self.words:
                context_words = self._get_context_words(generated_words)
                current_word = self._apply_attention(context_words)
                continue
                
            categories, weights = self._get_weighted_categories(current_word)
            
            if not categories:
                continue
                
            weights = self._apply_temperature(weights)
            category = random.choices(categories, weights=weights)[0]
            
            valid_words = self._get_valid_words(current_word, category)
            
            if not valid_words:
                continue
                
            words, probs = self._calculate_neural_probabilities(valid_words, generated_words)
            next_word = random.choices(words, weights=probs)[0]
            
            if next_word != (generated_words[-1] if generated_words else ''):
                generated_words.append(next_word)
                self._update_context_memory(next_word)
                self._update_attention_weights(current_word, next_word)
                current_word = next_word
        
        return self._format_generated_text(generated_words)
    def _process_sentence(self, words: List[str], phase: int = 1) -> float:
        """Process a sentence applying both neural and semantic mechanisms."""
        sentence_loss = 0.0
        
        for i, word in enumerate(words):
            word = word.strip('.,!?')
            if not word or not self._is_valid_word(word):
                continue
                
            prev_word = words[i-1].strip('.,!?') if i > 0 else 'start'
            if i > 0 and not self._is_valid_word(prev_word):
                prev_word = 'start'
                
            category = self._categorize_word(prev_word)
            
            try:
                if phase == 1:
                    # Neural phase with attention
                    self.words[prev_word][category][word] += 1
                    self._update_attention_weights(prev_word, word)
                    # Add small epsilon to prevent log(0)
                    word_count = max(self.words[prev_word][category][word], 1e-10)
                    sentence_loss += math.log(word_count)
                else:
                    # Semantic phase with position-based scaling
                    scale_factor = max((i + 1) / len(words), 1e-10)  # Ensure non-zero
                    semantic_weight = max(self._calculate_neural_weight(prev_word, category), 1e-10)
                    
                    # Ensure the product is positive and non-zero
                    current_count = max(self.words[prev_word][category][word], 1e-10)
                    scaled_value = current_count * scale_factor * semantic_weight
                    
                    # Update with the new scaled value
                    self.words[prev_word][category][word] = scaled_value
                    
                    # Calculate loss using the updated value
                    sentence_loss += math.log(max(scaled_value, 1e-10))
                    
            except ValueError as e:
                print(f"Warning: Value error processing word '{word}': {str(e)}")
                continue
                
        return sentence_loss

    def _calculate_neural_weight(self, word: str, category: str) -> float:
        """Calculate neural weight based on context and history."""
        base_weight = 1.0
        if word in self.attention_weights and category in self.attention_weights[word]:
            # Ensure weight is positive and non-zero
            weight = max(self.attention_weights[word][category], 1e-10)
            base_weight += weight
        return base_weight

    def _apply_temperature(self, weights: List[float]) -> List[float]:
        """Apply temperature scaling for controlled randomness."""
        if not weights:
            return []
            
        # Ensure all weights are positive and non-zero
        weights = [max(w, 1e-10) for w in weights]
        weights = np.array(weights)
        
        try:
            # Safe log computation with clipping
            log_weights = np.log(weights)
            scaled_weights = np.exp(log_weights / max(self.temperature, 1e-10))
            return scaled_weights.tolist()
        except Exception as e:
            print(f"Warning: Error in temperature scaling: {str(e)}")
            return weights.tolist()  # Return original weights if scaling fails

    def _get_context_words(self, generated_words: List[str]) -> List[str]:
        """Get recent context words with memory mechanism."""
        return generated_words[-self.memory_length:] if generated_words else ['start']

    def _apply_attention(self, context_words: List[str]) -> str:
        """Apply neural attention mechanism to select next word."""
        attention_scores = {}
        for word in context_words:
            if word in self.attention_weights:
                for next_word, weight in self.attention_weights[word].items():
                    attention_scores[next_word] = attention_scores.get(next_word, 0) + weight
        
        if attention_scores:
            words = list(attention_scores.keys())
            scores = list(attention_scores.values())
            return random.choices(words, weights=scores)[0]
        return 'start'

    def _get_weighted_categories(self, word: str) -> Tuple[List[str], List[float]]:
        """Get categories with combined neural and semantic weighting."""
        categories = []
        weights = []
        
        for category, word_counts in self.words[word].items():
            if word_counts:
                total_count = sum(word_counts.values())
                neural_weight = self._calculate_neural_weight(word, category)
                semantic_weight = self._calculate_semantic_weight(word, category)
                categories.append(category)
                weights.append(total_count * neural_weight * semantic_weight)
                
        return categories, weights

    def _calculate_neural_weight(self, word: str, category: str) -> float:
        """Calculate neural weight based on context and history."""
        base_weight = 1.0
        if word in self.attention_weights and category in self.attention_weights[word]:
            base_weight += self.attention_weights[word][category]
        return base_weight

    def _calculate_semantic_weight(self, word: str, category: str) -> float:
        """Calculate semantic weight based on word properties."""
        if category == 'START':
            return 1.2  # Boost sentence starters
        elif category in ['VERB', 'NOUN']:
            return 1.1  # Slight boost for content words
        return 1.0

    def _get_valid_words(self, word: str, category: str) -> List[Tuple[str, float]]:
        """Get valid next words with their base probabilities."""
        return [(w, c) for w, c in self.words[word][category].items()
                if self._is_valid_word(w)]

    def _calculate_neural_probabilities(self, valid_words: List[Tuple[str, float]], 
                                      context: List[str]) -> Tuple[List[str], List[float]]:
        """Calculate combined neural and semantic probabilities."""
        words, base_probs = zip(*valid_words)
        
        adjusted_probs = []
        for word, prob in zip(words, base_probs):
            context_boost = self._get_context_boost(word, context)
            semantic_boost = self._get_semantic_boost(word)
            adjusted_probs.append(prob * context_boost * semantic_boost)
            
        return words, adjusted_probs

    def _get_context_boost(self, word: str, context: List[str]) -> float:
        """Calculate context-based probability boost."""
        boost = 1.0
        recent_context = context[-self.memory_length:]
        
        for ctx_word in recent_context:
            if ctx_word in self.attention_weights and word in self.attention_weights[ctx_word]:
                boost *= (1 + self.attention_weights[ctx_word][word])
        
        return boost

    def _get_semantic_boost(self, word: str) -> float:
        """Calculate semantic-based probability boost."""
        category = self._categorize_word(word)
        if category in ['VERB', 'NOUN', 'ADJECTIVE']:
            return 1.2  # Boost content words
        return 1.0

    def _update_context_memory(self, word: str):
        """Update neural context memory."""
        self.context_memory.append(word)
        if len(self.context_memory) > self.memory_length:
            self.context_memory.pop(0)

    def _update_attention_weights(self, current_word: str, next_word: str):
        """Update neural attention weights with semantic influence."""
        if current_word not in self.attention_weights:
            self.attention_weights[current_word] = {}
        
        if next_word not in self.attention_weights[current_word]:
            self.attention_weights[current_word][next_word] = 0.1
            
        # Apply semantic-aware strengthening
        category = self._categorize_word(next_word)
        semantic_modifier = 1.2 if category in ['VERB', 'NOUN', 'ADJECTIVE'] else 1.1
        self.attention_weights[current_word][next_word] *= semantic_modifier

    def _format_generated_text(self, words: List[str]) -> str:
        """Format generated text with enhanced formatting rules."""
        if not words:
            return ""
            
        text = ' '.join(words)
        sentences = text.split('.')
        formatted_sentences = []
        
        for sentence in sentences:
            if sentence.strip():
                sentence = sentence.strip()
                # Ensure proper capitalization
                sentence = sentence[0].upper() + sentence[1:] if sentence else ''
                # Add period if missing
                if sentence[-1] not in '.!?':
                    sentence += '.'
                formatted_sentences.append(sentence)
                
        return ' '.join(formatted_sentences)

def save_model(generator: HybridTextGenerator, filename: str):
    """Save the model state to a file."""
    model_state = {
        'words': dict(generator.words),
        'attention_weights': generator.attention_weights,
        'word_categories': generator.word_categories,
        'temperature': generator.temperature,
        'memory_length': generator.memory_length,
        'decay_rate': generator.decay_rate,
        'probability_threshold': generator.probability_threshold
    }
    
    os.makedirs('models', exist_ok=True)
    with open(f'models/{filename}.pkl', 'wb') as f:
        pickle.dump(model_state, f)
    print(f"\nModel saved to models/{filename}.pkl")

def load_model(filename: str) -> Optional[HybridTextGenerator]:
    """Load the model state from a file."""
    try:
        with open(f'models/{filename}.pkl', 'rb') as f:
            model_state = pickle.load(f)
            
        generator = HybridTextGenerator(
            temperature=model_state.get('temperature', 0.7),
            memory_length=model_state.get('memory_length', 5),
            decay_rate=model_state.get('decay_rate', 0.95),
            probability_threshold=model_state.get('probability_threshold', 0.01)
        )
        
        # Restore model state
        generator.words = defaultdict(lambda: defaultdict(Counter))
        generator.words.update(model_state['words'])
        generator.attention_weights = model_state['attention_weights']
        generator.word_categories = model_state['word_categories']
        
        print(f"\nModel loaded from models/{filename}.pkl")
        return generator
    except FileNotFoundError:
        print(f"\nNo model file found at models/{filename}.pkl")
        return None
    except Exception as e:
        print(f"\nError loading model: {str(e)}")
        return None

def main():
    """Main function for the Hybrid Text Generator application."""
    print("\nHybrid Neural-Semantic Text Generator")
    print("====================================")
    
    generator = HybridTextGenerator(
        temperature=0.7,
        memory_length=5,
        decay_rate=0.95,
        probability_threshold=0.01
    )
    
    os.makedirs('models', exist_ok=True)
    
    while True:
        print("\nOptions:")
        print("1. Train model")
        print("2. Generate text")
        print("3. Save model")
        print("4. Load model")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        try:
            if choice == '1':

                filepath = input("Enter filepath: ").strip()
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                except Exception as e:
                    print(f"Error reading file: {str(e)}")
                    continue
               
                
                max_epochs = int(input("Enter number of training epochs (default 10): ") or "10")
                losses = generator.train_until_convergence(text, max_epochs)
                
            elif choice == '2':
                if not generator.words:
                    print("Error: Model not trained. Please train or load a model first.")
                    continue
                
                print("\nGenerated text:")
                print("-" * 50)
                print(generator.generate_text(OUT_LENGTH))
                print("-" * 50)
                
            elif choice == '3':
                if not generator.words:
                    print("Error: No model to save. Please train a model first.")
                    continue
                
                name = input("Enter model name: ").strip()
                if name:
                    save_model(generator, name)
                else:
                    print("Invalid model name")
                    
            elif choice == '4':
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
                    
                    loaded_generator = load_model(models[model_idx-1])
                    if loaded_generator:
                        generator = loaded_generator
                except ValueError:
                    print("Please enter a valid number")
                    
            elif choice == '5':
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
