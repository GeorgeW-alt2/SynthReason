import random
import re
import pickle
from typing import List, Dict, Tuple, Any
from collections import Counter, defaultdict
import numpy as np

KB_MEMORY = -1

def clean_text(text: str) -> str:
    """Clean text using regular expressions."""
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text) 
    
    # Remove phone numbers
    text = re.sub(r'\+?\d{1,3}[-.\s]?\(?\d{1,3}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}', '', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^A-Za-z0-9\s.,!?\'"-]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove extra dots
    text = re.sub(r'\.{2,}', '.', text)
    
    # Fix spacing around punctuation
    text = re.sub(r'\s([.,!?])', r'\1', text)  
    text = re.sub(r'([.,!?])([A-Za-z])', r'\1 \2', text)
    
    # Fix contractions spacing
    text = re.sub(r'\s\'', '\'', text)
    
    # Fix quote spacing
    text = re.sub(r'\s"', '"', text)
    text = re.sub(r'"\s', '"', text)
    
    # Fix dash spacing
    text = re.sub(r'\s-\s', '-', text)
    
    # Normalize multiple consecutive spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Ensure first character is uppercase
    if text and text[0].isalpha():
        text = text[0].upper() + text[1:]
    
    # Ensure there's a period at the end if not already ended with punctuation
    if text and not text[-1] in '.!?':
        text += '.'
        
    return text

class SemanticGenerator:
    def __init__(self):
        self.templates = []
        self.words = {
            'nouns': Counter({
                'cat': 1, 'dog': 1, 'house': 1, 'tree': 1, 'car': 1, 
                'book': 1, 'computer': 1, 'phone': 1, 'person': 1,
                'time': 1, 'day': 1, 'year': 1, 'city': 1, 'world': 1,
                'family': 1, 'friend': 1, 'food': 1, 'water': 1, 'life': 1
            }),
            'verbs': Counter({
                'run': 1, 'walk': 1, 'jump': 1, 'sit': 1, 'stand': 1,
                'eat': 1, 'drink': 1, 'sleep': 1, 'wake': 1, 'work': 1,
                'play': 1, 'study': 1, 'read': 1, 'write': 1, 'speak': 1
            }),
            'adjectives': Counter({
                'big': 1, 'small': 1, 'tall': 1, 'short': 1, 'long': 1,
                'wide': 1, 'narrow': 1, 'thick': 1, 'thin': 1, 'heavy': 1,
                'light': 1, 'fast': 1, 'slow': 1, 'hot': 1, 'cold': 1
            }),
            'adverbs': Counter({
                'quickly': 1, 'slowly': 1, 'carefully': 1, 'carelessly': 1,
                'quietly': 1, 'loudly': 1, 'happily': 1, 'sadly': 1,
                'easily': 1, 'hardly': 1, 'completely': 1, 'partially': 1
            }),
            'prepositions': Counter({
                'in': 1, 'on': 1, 'at': 1, 'to': 1, 'from': 1,
                'with': 1, 'without': 1, 'by': 1, 'near': 1, 'far': 1
            }),
            'determiners': Counter({
                'the': 1, 'a': 1, 'an': 1, 'this': 1, 'that': 1,
                'these': 1, 'those': 1, 'my': 1, 'your': 1, 'his': 1
            })
        }
        self.endings = {'.', '!', '?'}
        
    def weighted_choice(self, counter: Counter) -> str:
        """Select a word based on its frequency weight."""
        if not counter:
            return ""
        words = list(counter.keys())
        weights = list(counter.values())
        total = sum(weights)
        probabilities = [w/total for w in weights]
        return random.choices(words, weights=probabilities, k=1)[0]
        
    def learn_from_text(self, text: str) -> None:
        """Learn patterns and words from input text."""
        sentences = self._split_into_sentences(text)
        for sentence in sentences:
            template = self._extract_template(sentence)
            if template:
                self.templates.append(template)
            words = sentence.lower().split()
            self._categorize_words(words)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        text = ' '.join(text.split())
        sentences = []
        current = []
        
        words = text.split()
        for i, word in enumerate(words):
            current.append(word)
            if any(word.endswith(end) for end in self.endings):
                if not (len(word) <= 3 and word[0].isupper()):
                    sentences.append(' '.join(current))
                    current = []
                    
        if current:
            sentences.append(' '.join(current))
            
        return sentences
    
    def _extract_template(self, sentence: str) -> List[Tuple[str, str]]:
        """Extract a template from a sentence."""
        template = []
        words = sentence.split()
        
        i = 0
        while i < len(words):
            word = words[i].lower()
            clean_word = word.rstrip('.,!?')
            
            if clean_word in self.words['determiners']:
                template.append(('DET', word))
                self.words['determiners'][clean_word] += 1
            elif clean_word.endswith(('ly')) and len(clean_word) > 2:
                template.append(('ADV', word))
                self.words['adverbs'][clean_word] += 1
            elif clean_word.endswith(('ed', 'ing')):
                template.append(('VERB', word))
                self.words['verbs'][clean_word] += 1
            elif clean_word in self.words['prepositions']:
                template.append(('PREP', word))
                self.words['prepositions'][clean_word] += 1
            elif i > 0 and words[i-1].lower() in self.words['determiners']:
                template.append(('NOUN', word))
                self.words['nouns'][clean_word] += 1
            elif i > 0 and template and template[-1][0] == 'NOUN':
                template.append(('VERB', word))
                self.words['verbs'][clean_word] += 1
            else:
                template.append(('WORD', word))
            
            i += 1
            
        return template
    
    def _categorize_words(self, words: List[str]) -> None:
        """Categorize words based on patterns and context."""
        for i, word in enumerate(words):
            clean_word = word.lower().rstrip('.,!?')
            
            max_freq = 0
            word_category = None
            for category, counter in self.words.items():
                if clean_word in counter:
                    max_freq = counter[clean_word]
                    word_category = category
            
            if max_freq > 0:
                if word_category:
                    self.words[word_category][clean_word] += 1
                continue
                
            if word.endswith(('ly')) and len(word) > 2:
                self.words['adverbs'][clean_word] += 1
            elif word.endswith(('ed', 'ing')):
                self.words['verbs'][clean_word] += 1
            elif i > 0 and words[i-1].lower() in self.words['determiners']:
                self.words['nouns'][clean_word] += 1
    
    def generate_sentence(self) -> str:
        """Generate a new sentence using learned patterns."""
        if not self.templates:
            return "No patterns learned yet."
            
        template = random.choice(self.templates)
        sentence_parts = []
        
        for word_type, original in template:
            if word_type == 'DET':
                word = self.weighted_choice(self.words['determiners'])
                sentence_parts.append(word if word else original)
            elif word_type == 'NOUN':
                word = self.weighted_choice(self.words['nouns'])
                sentence_parts.append(word if word else original)
            elif word_type == 'VERB':
                word = self.weighted_choice(self.words['verbs'])
                sentence_parts.append(word if word else original)
            elif word_type == 'ADV':
                word = self.weighted_choice(self.words['adverbs'])
                sentence_parts.append(word if word else original)
            elif word_type == 'PREP':
                word = self.weighted_choice(self.words['prepositions'])
                sentence_parts.append(word if word else original)
            else:
                sentence_parts.append(original)
        
        sentence = ' '.join(sentence_parts)
        sentence = sentence[0].upper() + sentence[1:]
        if not any(sentence.endswith(end) for end in self.endings):
            sentence += '.'
            
        return sentence
    
    def generate_text(self, num_sentences: int = 3) -> str:
        """Generate multiple sentences."""
        return ' '.join(self.generate_sentence() for _ in range(num_sentences))
    
    def dump_probabilities(self, filename: str = 'semantic_generator_probabilities.pkl'):
        """Dump word probabilities and templates to a pickle file."""
        probabilities_dict = {}
        
        for category, counter in self.words.items():
            total = sum(counter.values())
            category_list = []
            for word, count in counter.items():
                probability = count / total
                index = list(counter.keys()).index(word)
                category_list.append((word, probability, index))
            
            category_list.sort(key=lambda x: x[2])
            probabilities_dict[category] = category_list
        
        probabilities_dict['templates'] = self.templates
        
        with open(filename, 'wb') as f:
            pickle.dump(probabilities_dict, f)
        
        print(f"Probabilities dumped to {filename}")
        
    def _build_language_models(self):
        """Build language models and save to dump file."""
        from tqdm import tqdm
        
        print("\nBuilding language models...")
        
        # Collect words from templates
        all_words = []
        for template in tqdm(self.templates, desc="Processing templates"):
            template_words = [word.lower() for _, word in template]
            all_words.extend(template_words)
        
        # Build unigram counts
        print("\nCalculating unigram probabilities...")
        unigram_counts = defaultdict(int)
        for word in tqdm(all_words):
            unigram_counts[word] += 1
        total_words = sum(unigram_counts.values())
        
        unigram_probs = {
            word: count / total_words 
            for word, count in tqdm(unigram_counts.items(), desc="Computing unigram probabilities")
        }
        
        # Initialize n-gram structures
        bigram_counts = defaultdict(lambda: defaultdict(int))
        context_counts = defaultdict(lambda: defaultdict(int))
        
        # Build context and bigram counts
        print("\nBuilding context models...")
        context_window = 10
        for i in tqdm(range(len(all_words) - context_window), desc="Processing word contexts"):
            context = tuple(all_words[i:i + context_window])
            next_word = all_words[i + context_window]
            context_counts[context][next_word] += 1
            
            prev_word = all_words[i]
            curr_word = all_words[i + 1]
            bigram_counts[prev_word][curr_word] += 1
        
        # Calculate context probabilities and divergence
        print("\nCalculating context probabilities...")
        context_data = {}
        for context, next_words in tqdm(context_counts.items(), desc="Computing context probabilities"):
            total = sum(next_words.values())
            context_probs = {
                word: count / total 
                for word, count in next_words.items()
            }
            
            # Calculate KL divergence
            epsilon = 1e-10
            p = np.array(list(context_probs.values()))
            q = np.array([unigram_probs.get(w, epsilon) for w in context_probs.keys()])
            p = p / np.sum(p)
            q = q / np.sum(q)
            divergence = np.sum(np.where(p > epsilon, p * np.log(p / q), 0))
            
            context_data[context] = {
                'probabilities': context_probs,
                'divergence': divergence
            }
        
        # Calculate transition probabilities
        print("\nCalculating transition probabilities...")
        transition_probs = {}
        for prev_word, next_words in tqdm(bigram_counts.items(), desc="Computing transitions"):
            total = sum(next_words.values())
            transition_probs[prev_word] = {
                word: count / total 
                for word, count in next_words.items()
            }
        
        # Create model dump
        print("\nPreparing model dump...")
        model_dump = {
            'unigram_probs': unigram_probs,
            'bigram_counts': dict(bigram_counts),
            'context_data': context_data,
            'transition_probs': transition_probs
        }
        
        # Save to file
        with open('language_model_dump.pkl', 'wb') as f:
            pickle.dump(model_dump, f)
            
        print("\nLanguage models successfully dumped to language_model_dump.pkl")

class NaturalTextGenerator:
    def __init__(self, prob_file='semantic_generator_probabilities.pkl', model_file='language_model_dump.pkl'):
        """Initialize generator from probability and model dumps."""
        # Load probabilities
        with open(prob_file, 'rb') as f:
            self.probabilities = pickle.load(f)
        
        # Load language model
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
        
        # Initialize components
# Set up models
        self.ngram_models = {
            'unigram': model_data['unigram_probs'],
            'bigram': model_data['bigram_counts']
        }
        
        # Context parameters
        self.context_window = 4
        self.divergence_threshold = 0.3
        
        # Initialize from dump
        self.forward_context = defaultdict(lambda: defaultdict(float))
        self.context_divergence = defaultdict(float)
        self.transition_probabilities = defaultdict(lambda: defaultdict(float))
        
        for context, data in model_data['context_data'].items():
            self.forward_context[context] = data['probabilities']
            self.context_divergence[context] = data['divergence']
        
        self.transition_probabilities = model_data['transition_probs']
        
        # Initialize categories
        self.categories = {}
        for category, category_list in self.probabilities.items():
            if category == 'templates':
                continue
            words = [item[0] for item in category_list]
            probs = [item[1] for item in category_list]
            self.categories[category] = {
                'words': words,
                'probabilities': probs
            }

    def calculate_kl_divergence(self, p_dist, q_dist):
        """Calculate KL divergence between two probability distributions."""
        # Create unified vocabulary
        all_words = set(p_dist.keys()) | set(q_dist.keys())
        
        # Initialize with small epsilon
        epsilon = 1e-10
        p = np.array([p_dist.get(word, epsilon) for word in all_words])
        q = np.array([q_dist.get(word, epsilon) for word in all_words])
        
        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        return np.sum(np.where(p > epsilon, p * np.log(p / q), 0))

    def weighted_choice(self, category):
        """Choose word from category based on probabilities."""
        if category not in self.categories:
            return None
            
        cat_data = self.categories[category]
        if not cat_data['words'] or not cat_data['probabilities']:
            return None
        
        try:
            return random.choices(
                cat_data['words'],
                weights=cat_data['probabilities'],
                k=1
            )[0]
        except ValueError as e:
            print(f"Error in weighted_choice: {e}")
            print(f"Category: {category}")
            print(f"Words length: {len(cat_data['words'])}")
            print(f"Probabilities length: {len(cat_data['probabilities'])}")
            return None

    def generate_word_from_context(self, prev_word, temperature):
        """Generate word based on context with temperature."""
        if prev_word in self.transition_probabilities:
            probs = self.transition_probabilities[prev_word]
            
            # Apply temperature scaling
            adjusted_probs = {
                word: np.power(prob, 1/temperature)
                for word, prob in probs.items()
            }
            
            # Normalize
            total = sum(adjusted_probs.values())
            if total > 0:
                adjusted_probs = {
                    word: prob/total 
                    for word, prob in adjusted_probs.items()
                }
                
                return random.choices(
                    list(adjusted_probs.keys()),
                    weights=list(adjusted_probs.values())
                )[0]
        
        # Fallback to random category
        categories = list(self.categories.keys())
        if categories:
            category = random.choice(categories)
            return self.weighted_choice(category)
        return None

    def continue_text(self, input_text, num_words=10, temperature=0.7):
        """Continue text using context and divergence."""
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        if not input_text:
            print("Warning: Empty input text")
            return ""
            
        words = input_text.lower().split()
        generated_words = []
        
        for _ in range(num_words):
            try:
                # Get dynamic context window
                context_window = min(random.randint(1, self.context_window), len(words))
                context_start = max(0, len(words) - context_window)
                recent_context = tuple(words[context_start:])
                
                if recent_context in self.forward_context:
                    next_word_probs = self.forward_context[recent_context]
                    
                    # Get divergence score
                    divergence = self.context_divergence.get(recent_context, 0)
                    
                    # Adjust temperature
                    local_temp = temperature
                    if divergence > self.divergence_threshold:
                        local_temp *= (1 + divergence)
                    else:
                        local_temp *= max(0.5, 1 - divergence)
                    
                    # Apply sigmoid with noise
                    noise_factor = random.uniform(0.1, 0.2) * (1 + divergence)
                    sigmoid_probs = {
                        word: sigmoid(prob * local_temp * noise_factor)
                        for word, prob in next_word_probs.items()
                    }
                    
                    # Boost low probabilities
                    boost_threshold = random.uniform(0.1, 0.2) * (1 + divergence)
                    for word, prob in sigmoid_probs.items():
                        if prob < boost_threshold and random.random() < divergence:
                            boost_factor = random.uniform(1.1, 2.0) * (1 + divergence)
                            sigmoid_probs[word] *= boost_factor
                    
                    # Temperature scaling
                    adjusted_probs = {
                        word: np.power(prob, 1/(local_temp * (1 + divergence)))
                        for word, prob in sigmoid_probs.items()
                    }
                    
                    # Normalize
                    total = sum(adjusted_probs.values())
                    if total > 0:
                        adjusted_probs = {
                            word: prob/total 
                            for word, prob in adjusted_probs.items()
                        }
                        
                        # Select word based on divergence
                        if random.random() < divergence:
                            sorted_words = sorted(
                                adjusted_probs.items(), 
                                key=lambda x: x[1]
                            )
                            pool_size = max(1, int(len(sorted_words) * divergence))
                            selection_pool = sorted_words[:pool_size]
                            if selection_pool:
                                next_word = random.choice(selection_pool)[0]
                            else:
                                next_word = self.generate_word_from_context(
                                    words[-1] if words else None,
                                    local_temp
                                )
                        else:
                            next_word = random.choices(
                                list(adjusted_probs.keys()),
                                weights=list(adjusted_probs.values())
                            )[0]
                    else:
                        next_word = self.generate_word_from_context(
                            words[-1] if words else None,
                            local_temp
                        )
                else:
                    next_word = self.generate_word_from_context(
                        words[-1] if words else None,
                        temperature
                    )
                
                if next_word:
                    words.append(next_word)
                    generated_words.append(next_word)
                    
            except Exception as e:
                print(f"Warning: Error during word generation: {e}")
                continue
        
        # Format output
        continuation = ' '.join(generated_words)
        if not continuation:
            return input_text
            
        if input_text[-1] in '.!?':
            continuation = continuation[0].upper() + continuation[1:]
        
        return input_text + ' ' + continuation

def train_probs():
    """Train new model and create dumps."""
    generator = SemanticGenerator()
    
    print("Welcome to the Probabilistic Semantic Text Generator!")
    
    try:
        with open(input("Enter filename: "), 'r', encoding='iso-8859-1') as f:
            text = ' '.join(clean_text(f.read()).strip().split())[:KB_MEMORY]
            generator.learn_from_text(text)
            print("\nLoaded and processed base patterns from file.")
    except FileNotFoundError:
        print("\nNo base file found. Starting with empty patterns.")
    
    # Create both dumps
    generator.dump_probabilities()
    generator._build_language_models()
    return

def main():
    while True:
        choice = input("\nChoose an option:\n1. Train new model\n2. Continue with existing model\nChoice (1/2): ").strip()
        
        if choice == "1":
            train_probs()
        
        elif choice == "2":
            try:
                generator = NaturalTextGenerator()
                print("\nEntering text generation mode. Type your prompts (or 'quit'/'exit' to end):")
                
                while True:
                    try:
                        user_input = input("\nUser: ").strip()
                        if user_input.lower() in ['quit', 'exit']:
                            break
                        continued_text = generator.continue_text(
                            user_input, 
                            num_words=250, 
                            temperature=0.72
                        )
                        print("\nAI:", continued_text)
                    except KeyboardInterrupt:
                        print("\nExiting...")
                        break
                    except Exception as e:
                        print(f"Error occurred: {e}")
                        continue
                        
            except FileNotFoundError:
                print("\nError: Model dumps not found. Please train a model first (option 1).")
            except Exception as e:
                print(f"\nError loading model: {e}")
        
        else:
            print("Invalid choice. Please select 1 or 2.")
            continue
        
if __name__ == "__main__":
    main()