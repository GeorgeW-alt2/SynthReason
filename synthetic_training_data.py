import numpy as np
import random
import re
import pickle
import math
import os
from collections import defaultdict, Counter, deque
from typing import List, Tuple, Dict, Any, Optional, Deque
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

KB_LIMIT = 9999

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

class TextProcessor:
    @staticmethod
    def clean_text(text: str) -> str:
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
    
    @staticmethod
    def post_process_text(text: str) -> str:
        # Remove multiple periods
        text = re.sub(r'\.+', '.', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        text = re.sub(r'([.,!?])(?=[A-Za-z])', r'\1 ', text)
        
        # Ensure proper capitalization
        sentences = text.split('.')
        formatted = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                formatted.append(sentence + '.')
        
        return ' '.join(formatted)
    
    @staticmethod
    def categorize_word(word: str) -> str:
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

class BaseGenerator:
    def __init__(self, 
                 decay_rate: float = 0.95,
                 convergence_threshold: float = 1e-6,
                 context_size: int = 5):
        self.decay_rate = decay_rate
        self.convergence_threshold = convergence_threshold
        self.is_converged = False
        self.total_epochs = 0
        self.context_size = context_size
        self.context_window = ContextWindow(context_size)
        
    def check_convergence(self, errors: List[float]) -> bool:
        if len(errors) < 10:
            return False
            
        recent_errors = errors[-10:]
        diffs = [abs(recent_errors[i] - recent_errors[i-1]) 
                for i in range(1, len(recent_errors))]
                
        return max(diffs) < self.convergence_threshold

class ErrorAwareGenerator(BaseGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.words = defaultdict(lambda: defaultdict(Counter))
        self.context_transitions = defaultdict(Counter)
        self.error_history: List[Tuple[float, int]] = []
        
    def train(self, text: str, max_epochs: int = 100) -> List[float]:
        text = TextProcessor.clean_text(text)
        sentences = text.split('.')
        epoch_errors = []
        
        while self.total_epochs < max_epochs and not self.is_converged:
            epoch_error = 0.0
            num_samples = 0
            
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
                    
                    # Regular word-based training
                    category = TextProcessor.categorize_word(word)
                    prev_word = words[i-1].strip('.,!?') if i > 0 else 'start'
                    
                    error_magnitude = 1.0 if prev_word not in self.words else 0.5
                    decayed_error = error_magnitude * (self.decay_rate ** self.total_epochs)
                    self.error_history.append((decayed_error, self.total_epochs))
                    
                    self.words[prev_word][category][word] += 1
                    self.context_window.add(word)
                    
                    epoch_error += decayed_error
                    num_samples += 1
            
            avg_epoch_error = epoch_error / num_samples if num_samples > 0 else 0
            epoch_errors.append(avg_epoch_error)
            
            self.is_converged = self.check_convergence(epoch_errors)
            self.total_epochs += 1
            
            if self.total_epochs % 10 == 0:
                print(f"Epoch {self.total_epochs}: Average Error = {avg_epoch_error:.6f}")
                
        print(f"\nTraining completed after {self.total_epochs} epochs")
        print(f"Converged: {self.is_converged}")
        return epoch_errors
    
    def generate(self, seed_text: str = "", num_words: int = 50) -> str:
        if not self.is_converged:
            print("Warning: Model not trained or loaded. Results may be unreliable.")
            
        generated_words = []
        self.context_window.clear()
        
        initial_words = [word.strip('.,!?') for word in seed_text.lower().split()]
        if not initial_words:
            initial_words = ["the"]
            
        self.context_window.add_multiple(initial_words)
        generated_words.extend(initial_words)
        
        current_word = initial_words[-1] if initial_words else 'start'
        attempts = 0
        max_attempts = num_words * 2
        
        while len(generated_words) < num_words and attempts < max_attempts:
            attempts += 1
            
            context = self.context_window.get_context()
            if context in self.context_transitions and self.context_transitions[context]:
                candidates = list(self.context_transitions[context].items())
                words, counts = zip(*candidates)
                next_word = random.choices(words, weights=counts)[0]
            else:
                if current_word not in self.words:
                    current_word = random.choice(generated_words[-3:] or ['start'])
                    continue
                    
                categories = []
                weights = []
                for category, word_counts in self.words[current_word].items():
                    if word_counts:
                        categories.append(category)
                        weights.append(sum(word_counts.values()))
                
                if not categories:
                    current_word = random.choice(generated_words[-3:] or ['start'])
                    continue
                    
                category = random.choices(categories, weights=weights)[0]
                available_words = []
                word_weights = []
                for word, count in self.words[current_word][category].items():
                    available_words.append(word)
                    word_weights.append(count)
                
                if not available_words:
                    current_word = random.choice(generated_words[-3:] or ['start'])
                    continue
                    
                next_word = random.choices(available_words, weights=word_weights)[0]
            
            if next_word != generated_words[-1]:
                generated_words.append(next_word)
                self.context_window.add(next_word)
                current_word = next_word
            
            if next_word.endswith('.'):
                self.context_window.clear()
                if len(generated_words) < num_words:
                    self.context_window.add_multiple(generated_words[-min(3, len(generated_words)):])
        
        return TextProcessor.post_process_text(' '.join(generated_words))

    def save(self, filepath: str):
        try:
            state = {
                'words': dict(self.words),
                'context_transitions': dict(self.context_transitions),
                'error_history': self.error_history,
                'is_converged': self.is_converged,
                'total_epochs': self.total_epochs,
                'context_size': self.context_size
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
                
        except Exception as e:
            raise Exception(f"Failed to save model: {str(e)}")
    
    def load(self, filepath: str):
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.words = defaultdict(lambda: defaultdict(Counter))
            for context, categories in state['words'].items():
                for category, words in categories.items():
                    self.words[context][category].update(words)
            
            self.context_transitions = defaultdict(Counter)
            for context, transitions in state['context_transitions'].items():
                self.context_transitions[context].update(transitions)
            
            self.error_history = state['error_history']
            self.is_converged = state['is_converged']
            self.total_epochs = state['total_epochs']
            self.context_size = state['context_size']
            self.context_window = ContextWindow(self.context_size)
            
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")

class Attention:
    def __init__(self, dim: int = 64, num_heads: int = 4):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Initialize weights
        self.W_q = np.random.randn(dim, dim) * 0.02
        self.W_k = np.random.randn(dim, dim) * 0.02
        self.W_v = np.random.randn(dim, dim) * 0.02
        self.W_o = np.random.randn(dim, dim) * 0.02
    
    def split_heads(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len = x.shape[:2]
        return x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
    
    def apply_causal_mask(self, scores: np.ndarray) -> np.ndarray:
        seq_len = scores.shape[-1]
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        scores = scores.copy()
        scores[..., mask.astype(bool)] = float('-inf')
        return scores
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len = x.shape[:2]
        
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / self.scale
        scores = self.apply_causal_mask(scores)
        
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights /= np.sum(attn_weights, axis=-1, keepdims=True) + 1e-9
        
        context = np.matmul(attn_weights, V)
        context = context.reshape(batch_size, seq_len, self.dim)
        
        return context @ self.W_o

class CausalGenerator(BaseGenerator):
    def __init__(self, dim: int = 64, num_heads: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.embedding = None
        self.attention = Attention(dim, num_heads)
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2
        
    def _build_vocabulary(self, text: str):
        word_counts = Counter(text.lower().split())
        
        for word in word_counts:
            if word not in self.word_to_idx:
                self.word_to_idx[word] = self.vocab_size
                self.idx_to_word[self.vocab_size] = word
                self.vocab_size += 1
        
        if self.embedding is None or self.embedding.shape[0] != self.vocab_size:
            new_embedding = np.random.randn(self.vocab_size, self.dim) * 0.02
            if self.embedding is not None:
                old_size = min(self.embedding.shape[0], self.vocab_size)
                new_embedding[:old_size] = self.embedding[:old_size]
            self.embedding = new_embedding
    
    def _encode_sequence(self, words: List[str]) -> np.ndarray:
        indices = [self.word_to_idx.get(word.lower(), 1) for word in words]
        return np.array([self.embedding[idx] for idx in indices])
    
    def train(self, text: str, max_epochs: int = 100) -> List[float]:
        text = TextProcessor.clean_text(text)
        print("Building vocabulary...")
        self._build_vocabulary(text)
        print(f"Vocabulary size: {self.vocab_size}")
        
        sentences = text.split('.')
        epoch_errors = []
        
        while self.total_epochs < max_epochs and not self.is_converged:
            epoch_error = 0.0
            num_samples = 0
            
            for sentence in sentences:
                words = sentence.lower().split()
                if len(words) < 2:
                    continue
                
                # Encode sequence
                encoded_seq = self._encode_sequence(words)
                encoded_seq = encoded_seq.reshape(1, len(words), self.dim)
                
                # Forward pass
                attention_output = self.attention.forward(encoded_seq)
                
                # Calculate error
                for i in range(1, len(words)):
                    target_encoding = self._encode_sequence([words[i]])[0]
                    error = np.mean(np.abs(attention_output[0, i-1] - target_encoding))
                    decayed_error = error * (self.decay_rate ** self.total_epochs)
                    epoch_error += decayed_error
                    num_samples += 1
            
            avg_epoch_error = epoch_error / num_samples if num_samples > 0 else 0
            epoch_errors.append(avg_epoch_error)
            
            self.is_converged = self.check_convergence(epoch_errors)
            self.total_epochs += 1
            
            if self.total_epochs % 10 == 0:
                print(f"Epoch {self.total_epochs}: Average Error = {avg_epoch_error:.6f}")
        
        print(f"\nTraining completed after {self.total_epochs} epochs")
        print(f"Converged: {self.is_converged}")
        return epoch_errors
    
    def generate(self, seed_text: str = "", num_words: int = 50, temperature: float = 0.7) -> str:
        if not self.is_converged:
            print("Warning: Model not trained or loaded. Results may be unreliable.")
        
        generated_words = []
        self.context_window.clear()
        
        initial_words = [word.strip('.,!?') for word in seed_text.lower().split()]
        if not initial_words:
            initial_words = ["the"]
        
        self.context_window.add_multiple(initial_words)
        generated_words.extend(initial_words)
        
        while len(generated_words) < num_words:
            context_seq = self._encode_sequence(generated_words[-self.context_size:])
            context_seq = context_seq.reshape(1, -1, self.dim)
            
            attention_output = self.attention.forward(context_seq)
            last_output = attention_output[0, -1]
            
            # Calculate word probabilities
            similarities = np.dot(self.embedding, last_output)
            similarities = np.exp(similarities / temperature)
            probabilities = similarities / np.sum(similarities)
            
            # Sample next word (excluding special tokens)
            valid_indices = np.arange(2, self.vocab_size)
            probabilities = probabilities[valid_indices]
            probabilities /= np.sum(probabilities)
            
            next_idx = valid_indices[np.random.choice(len(valid_indices), p=probabilities)]
            next_word = self.idx_to_word[next_idx]
            
            generated_words.append(next_word)
            self.context_window.add(next_word)
        
        return TextProcessor.post_process_text(' '.join(generated_words))
    
    def save(self, filepath: str):
        try:
            state = {
                'embedding': self.embedding,
                'attention_weights': {
                    'W_q': self.attention.W_q,
                    'W_k': self.attention.W_k,
                    'W_v': self.attention.W_v,
                    'W_o': self.attention.W_o
                },
                'word_to_idx': self.word_to_idx,
                'idx_to_word': self.idx_to_word,
                'vocab_size': self.vocab_size,
                'is_converged': self.is_converged,
                'total_epochs': self.total_epochs,
                'context_size': self.context_size
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
                
        except Exception as e:
            raise Exception(f"Failed to save model: {str(e)}")
    
    def load(self, filepath: str):
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.embedding = state['embedding']
            self.attention.W_q = state['attention_weights']['W_q']
            self.attention.W_k = state['attention_weights']['W_k']
            self.attention.W_v = state['attention_weights']['W_v']
            self.attention.W_o = state['attention_weights']['W_o']
            
            self.word_to_idx = state['word_to_idx']
            self.idx_to_word = state['idx_to_word']
            self.vocab_size = state['vocab_size']
            self.is_converged = state['is_converged']
            self.total_epochs = state['total_epochs']
            self.context_size = state['context_size']
            self.context_window = ContextWindow(self.context_size)
            
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")

class TextGenerator:
    def __init__(self,
                 causal_dim: int = 64,
                 num_heads: int = 4,
                 decay_rate: float = 0.25,
                 convergence_threshold: float = 1e-6,
                 context_size: int = 15):
        
        self.causal_gen = CausalGenerator(
            dim=causal_dim,
            num_heads=num_heads,
            decay_rate=decay_rate,
            convergence_threshold=convergence_threshold,
            context_size=context_size
        )
        
        self.error_gen = ErrorAwareGenerator(
            decay_rate=decay_rate,
            convergence_threshold=convergence_threshold,
            context_size=context_size
        )
        
        self.generation_cache = {}
        self.cache_size_limit = 1000
        self.batch_size = 5
    
    def train(self, text: str, max_epochs: int = 100) -> Tuple[List[float], List[float]]:
        print("Training Causal Generator...")
        causal_errors = self.causal_gen.train(text, max_epochs)
        
        print("\nTraining Error-Aware Generator...")
        error_errors = self.error_gen.train(text, max_epochs)
        
        return causal_errors, error_errors
    
    def _generate_batch(self, seed_text: str, batch_size: int, 
                       num_words: int = 100, temperature: float = 0.7) -> List[str]:
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = []
            for _ in range(batch_size):
                futures.append(executor.submit(
                    lambda: self.causal_gen.generate(seed_text, num_words, temperature)
                ))
            return [future.result() for future in futures]

    def _enhance_text(self, base_text: str, target_length: int) -> str:
        cache_key = (base_text[:100], target_length)
        if cache_key in self.generation_cache:
            return self.generation_cache[cache_key]
        
        enhanced = self.error_gen.generate(base_text, target_length)
        
        if len(self.generation_cache) > self.cache_size_limit:
            num_to_remove = len(self.generation_cache) - self.cache_size_limit + 1
            for _ in range(num_to_remove):
                self.generation_cache.pop(next(iter(self.generation_cache)))
        
        self.generation_cache[cache_key] = enhanced
        return enhanced

    def generate(self,
                seed_text: str = "The",
                num_iterations: int = 60,
                base_words: int = 100,
                final_words: int = 500,
                temperature: float = 0.7) -> str:
        combined_text = ""
        
        print("\nGenerating text...")
        with tqdm(total=num_iterations) as pbar:
            for i in range(0, num_iterations, self.batch_size):
                batch_size = min(self.batch_size, num_iterations - i)
                base_texts = self._generate_batch(seed_text, batch_size, base_words, temperature)
                
                for base_text in base_texts:
                    enhanced_text = self._enhance_text(base_text, final_words)
                    combined_text += enhanced_text + " "
                    pbar.update(1)
        
        return TextProcessor.post_process_text(combined_text)
    
    def save(self, filepath: str):
        try:
            base_path = os.path.splitext(filepath)[0]
            self.causal_gen.save(f"{base_path}_causal.pkl")
            self.error_gen.save(f"{base_path}_error.pkl")
        except Exception as e:
            raise Exception(f"Failed to save models: {str(e)}")
    
    def load(self, filepath: str):
        try:
            base_path = os.path.splitext(filepath)[0]
            self.causal_gen.load(f"{base_path}_causal.pkl")
            self.error_gen.load(f"{base_path}_error.pkl")
        except Exception as e:
            raise Exception(f"Failed to load models: {str(e)}")

def main():
    print("Enhanced Text Generation System")
    print("==============================")
    
    generator = TextGenerator(
        causal_dim=64,
        num_heads=4,
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
                filename = input("Enter training file name: ")
                with open(filename, 'r', encoding='utf-8') as f:
                    text = ' '.join(f.read().split(".")[:KB_LIMIT])
                print("\nTraining models...")
                causal_errors, error_errors = generator.train(text)
                print(f"\nTraining complete:")
                print(f"Causal final error: {causal_errors[-1]:.6f}")
                print(f"Error-aware final error: {error_errors[-1]:.6f}")
                
                # Auto-save
                model_path = os.path.join('models', 'auto_saved_model.pkl')
                generator.save(model_path)
                print(f"Model automatically saved to {model_path}")
                
            except FileNotFoundError:
                print("Error: Training file not found!")
            except Exception as e:
                print(f"Error during training: {e}")
        
        elif choice == "2":
            try:
                iterations = 60
                base_words = 5
                final_words = 150
                temperature = 0.7
                while True:
                    seed_text = input("Enter seed text: ")
                    
                    generated_text = generator.generate(
                        seed_text=seed_text,
                        num_iterations=iterations,
                        base_words=base_words,
                        final_words=final_words,
                        temperature=temperature
                    )
                
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
                
                generator.save(model_path)
                print(f"Model saved successfully to {model_path}")
            except Exception as e:
                print(f"Error saving model: {e}")
        
        elif choice == "4":
            try:
                models = [f for f in os.listdir('models') if f.endswith('_causal.pkl')]
                if not models:
                    print("No saved models found in 'models' directory.")
                    continue
                
                print("\nAvailable models:")
                for i, model in enumerate(models, 1):
                    base_name = model.replace('_causal.pkl', '')
                    print(f"{i}. {base_name}")
                
                choice = input("\nEnter model number to load (or name): ").strip()
                
                if choice.isdigit() and 1 <= int(choice) <= len(models):
                    model_name = models[int(choice) - 1].replace('_causal.pkl', '')
                else:
                    model_name = choice if choice.endswith('.pkl') else choice + '.pkl'
                    model_name = model_name.replace('.pkl', '')
                
                model_path = os.path.join('models', model_name)
                generator.load(model_path)
                print(f"Model loaded successfully from {model_path}")
            except FileNotFoundError:
                print("Error: Model files not found!")
            except Exception as e:
                print(f"Error loading model: {e}")
        
        elif choice == "5":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
