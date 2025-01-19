import random
import re
import pickle
from typing import List, Dict, Tuple, Any
from collections import Counter, defaultdict
import numpy as np

KB_MEMORY = -1
filename = "test.txt"
class SemanticGenerator:
    def __init__(self):
        self.templates = []
        self.words = {
            'nouns': Counter({
                'cat': 1, 'dog': 1, 'house': 1, 'tree': 1, 'car': 1, 
                'book': 1, 'computer': 1, 'phone': 1, 'person': 1,
                'time': 1, 'day': 1, 'year': 1, 'city': 1, 'world': 1,
                'family': 1, 'friend': 1, 'food': 1, 'water': 1, 'life': 1,
                'work': 1, 'school': 1, 'office': 1, 'home': 1, 'room': 1,
                'door': 1, 'window': 1, 'table': 1, 'chair': 1, 'bed': 1,
                'street': 1, 'road': 1, 'park': 1, 'garden': 1, 'store': 1,
                'weather': 1, 'sun': 1, 'moon': 1, 'star': 1, 'sky': 1
            }),
            
            'verbs': Counter({
                'run': 1, 'walk': 1, 'jump': 1, 'sit': 1, 'stand': 1,
                'eat': 1, 'drink': 1, 'sleep': 1, 'wake': 1, 'work': 1,
                'play': 1, 'study': 1, 'read': 1, 'write': 1, 'speak': 1,
                'listen': 1, 'watch': 1, 'look': 1, 'see': 1, 'hear': 1,
                'think': 1, 'know': 1, 'understand': 1, 'learn': 1, 'teach': 1,
                'make': 1, 'create': 1, 'build': 1, 'break': 1, 'fix': 1,
                'open': 1, 'close': 1, 'start': 1, 'finish': 1, 'continue': 1,
                'live': 1, 'die': 1, 'grow': 1, 'change': 1, 'move': 1
            }),
            
            'adjectives': Counter({
                'big': 1, 'small': 1, 'tall': 1, 'short': 1, 'long': 1,
                'wide': 1, 'narrow': 1, 'thick': 1, 'thin': 1, 'heavy': 1,
                'light': 1, 'fast': 1, 'slow': 1, 'hot': 1, 'cold': 1,
                'warm': 1, 'cool': 1, 'new': 1, 'old': 1, 'young': 1,
                'good': 1, 'bad': 1, 'happy': 1, 'sad': 1, 'angry': 1,
                'bright': 1, 'dark': 1, 'beautiful': 1, 'ugly': 1, 'clean': 1,
                'dirty': 1, 'rich': 1, 'poor': 1, 'strong': 1, 'weak': 1,
                'hard': 1, 'soft': 1, 'loud': 1, 'quiet': 1, 'sweet': 1
            }),
            
            'adverbs': Counter({
                'quickly': 1, 'slowly': 1, 'carefully': 1, 'carelessly': 1,
                'quietly': 1, 'loudly': 1, 'happily': 1, 'sadly': 1,
                'easily': 1, 'hardly': 1, 'completely': 1, 'partially': 1,
                'fully': 1, 'partly': 1, 'really': 1, 'actually': 1,
                'generally': 1, 'specifically': 1, 'usually': 1, 'rarely': 1,
                'always': 1, 'never': 1, 'often': 1, 'sometimes': 1,
                'early': 1, 'late': 1, 'today': 1, 'tomorrow': 1,
                'well': 1, 'badly': 1, 'fast': 1, 'slow': 1,
                'here': 1, 'there': 1, 'everywhere': 1, 'nowhere': 1,
                'inside': 1, 'outside': 1, 'forward': 1, 'backward': 1
            }),
            
            'prepositions': Counter({
                'in': 1, 'on': 1, 'at': 1, 'to': 1, 'from': 1,
                'with': 1, 'without': 1, 'by': 1, 'near': 1, 'far': 1,
                'through': 1, 'across': 1, 'between': 1, 'among': 1,
                'around': 1, 'about': 1, 'against': 1, 'along': 1,
                'behind': 1, 'beside': 1, 'besides': 1, 'during': 1,
                'except': 1, 'for': 1, 'into': 1, 'like': 1,
                'of': 1, 'off': 1, 'over': 1, 'since': 1,
                'under': 1, 'until': 1, 'up': 1, 'upon': 1,
                'above': 1, 'below': 1, 'within': 1, 'without': 1
            }),
            
            'determiners': Counter({
                'the': 1, 'a': 1, 'an': 1, 'this': 1, 'that': 1,
                'these': 1, 'those': 1, 'my': 1, 'your': 1, 'his': 1,
                'her': 1, 'its': 1, 'our': 1, 'their': 1, 'any': 1,
                'some': 1, 'many': 1, 'few': 1, 'several': 1, 'each': 1,
                'every': 1, 'all': 1, 'both': 1, 'either': 1, 'neither': 1,
                'no': 1, 'another': 1, 'such': 1, 'what': 1, 'which': 1,
                'whose': 1, 'enough': 1, 'much': 1, 'more': 1, 'most': 1
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
        
    def add_input_text(self, text: str) -> None:
        """Add new input text and learn from it."""
        self.learn_from_text(text)
        
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
        """Split text into sentences while handling common abbreviations."""
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
        """Extract a template from a sentence, marking word types."""
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
            
            # Skip words already strongly categorized
            max_freq = 0
            word_category = None
            for category, counter in self.words.items():
                if clean_word in counter:
                    if counter[clean_word] > max_freq:
                        max_freq = counter[clean_word]
                        word_category = category
            
            if max_freq > 0:
                if word_category:
                    self.words[word_category][clean_word] += 1
                continue
                
            # Apply categorization rules
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
        """
        Dump word probabilities and word indices to a pickle file.
        
        :param filename: Name of the pickle file to save probabilities
        """
        # Prepare a dictionary to store probabilities and indices
        probabilities_dict = {}
        
        for category, counter in self.words.items():
            # Calculate total count for the category
            total = sum(counter.values())
            
            # Create a list of (word, probability, index) tuples
            category_list = []
            for word, count in counter.items():
                probability = count / total
                index = list(counter.keys()).index(word)
                category_list.append((word, probability, index))
            
            # Sort the list by index to maintain original order
            category_list.sort(key=lambda x: x[2])
            
            probabilities_dict[category] = category_list
        
        # Dump templates as well
        probabilities_dict['templates'] = self.templates
        
        # Save to pickle file
        with open(filename, 'wb') as f:
            pickle.dump(probabilities_dict, f)
        
        print(f"Probabilities dumped to {filename}")
        return

def train_probs():
    """Run interactive text input session."""
    generator = SemanticGenerator()
    
    print("Welcome to the Probabilistic Semantic Text Generator!")
    
    # Try to load base text if available
    try:
        with open(filename, "r", encoding="utf-8") as f:
            text = ' '.join(f.read().split())[:KB_MEMORY]
            generator.learn_from_text(text)
            print("\nLoaded base patterns from file.")
    except FileNotFoundError:
        print("\nNo base file found. Starting with empty patterns.")
 
  
    print(generator.generate_text(10))
    
    # Optional: Dump probabilities after each input
    generator.dump_probabilities()
    return



class NaturalTextGenerator:
    def __init__(self, filename='semantic_generator_probabilities.pkl'):
        """
        Initialize an advanced text generator with natural language modeling
        
        :param filename: Path to the pickled probabilities file
        """
        # Load pickled probabilities
        with open(filename, 'rb') as f:
            self.probabilities = pickle.load(f)
        
        # Build advanced language models
        self.ngram_models = {}
        self.context_models = defaultdict(lambda: defaultdict(int))
        self.transition_probabilities = defaultdict(lambda: defaultdict(float))
        
        # Add forward continuation context
        self.forward_context = defaultdict(lambda: defaultdict(float))
        self.context_window = 3  # Number of words to consider for context
        
        # Preprocess categories and build language models
        self.categories = {}
        for category, category_list in self.probabilities.items():
            if category == 'templates':
                continue
            
            # Extract words and their probabilities
            words = [item[0] for item in category_list]
            probs = [item[1] for item in category_list]
            
            self.categories[category] = {
                'words': words,
                'probabilities': probs
            }
        
        # Build n-gram and context models from templates
        self._build_language_models()
    
    def _build_language_models(self):
        """
        Build advanced language models from templates including forward context
        """
        # Collect all words from templates
        all_words = []
        for template in self.probabilities.get('templates', []):
            template_words = [word.lower() for _, word in template]
            all_words.extend(template_words)
        
        # Unigram model
        unigram_counts = defaultdict(int)
        for word in all_words:
            unigram_counts[word] += 1
        total_words = sum(unigram_counts.values())
        
        # Unigram probabilities
        unigram_probs = {
            word: count / total_words 
            for word, count in unigram_counts.items()
        }
        
        # Bigram and context models
        bigram_counts = defaultdict(lambda: defaultdict(int))
        
        # Build forward context model
        for i in range(len(all_words) - self.context_window):
            context = tuple(all_words[i:i + self.context_window])
            next_word = all_words[i + self.context_window]
            self.forward_context[context][next_word] += 1
            
            # Also update bigram counts
            prev_word = all_words[i]
            curr_word = all_words[i + 1]
            bigram_counts[prev_word][curr_word] += 1
        
        # Normalize forward context probabilities
        for context, next_words in self.forward_context.items():
            total = sum(next_words.values())
            self.forward_context[context] = {
                word: count / total 
                for word, count in next_words.items()
            }
        
        # Calculate transition probabilities
        for prev_word, next_words in bigram_counts.items():
            total = sum(next_words.values())
            self.transition_probabilities[prev_word] = {
                word: count / total 
                for word, count in next_words.items()
            }
        
        # Store models
        self.ngram_models = {
            'unigram': unigram_probs,
            'bigram': dict(bigram_counts)
        }
    
    def weighted_choice(self, category):
        """
        Choose a word from a category based on its learned probability
        
        :param category: Word category to choose from
        :return: Selected word
        """
        cat_data = self.categories.get(category, None)
        if not cat_data or not cat_data['words']:
            return None
        
        return random.choices(
            cat_data['words'], 
            weights=cat_data['probabilities']
        )[0]

    def continue_text(self, input_text, num_words=10, temperature=0.7):
        """
        Continue given text using forward context mechanism
        
        :param input_text: Text to continue
        :param num_words: Number of words to generate
        :param temperature: Controls randomness (lower = more deterministic)
        :return: Continued text
        """
        # Tokenize input text
        words = input_text.lower().split()
        
        # Generate continuation
        generated_words = []
        
        for _ in range(num_words):
            # Get recent context
            context_start = max(0, len(words) - self.context_window)
            recent_context = tuple(words[context_start:])
            
            # If we have forward context for this sequence
            if recent_context in self.forward_context:
                next_word_probs = self.forward_context[recent_context]
                
                # Apply temperature
                adjusted_probs = {
                    word: np.power(prob, 1/temperature)
                    for word, prob in next_word_probs.items()
                }
                # Normalize
                total = sum(adjusted_probs.values())
                adjusted_probs = {
                    word: prob/total 
                    for word, prob in adjusted_probs.items()
                }
                
                # Choose next word
                next_word = random.choices(
                    list(adjusted_probs.keys()),
                    weights=list(adjusted_probs.values())
                )[0]
            
            # Fallback to regular generation if no forward context
            else:
                next_word = self.generate_word_from_context(words[-1] if words else None, temperature)
            
            words.append(next_word)
            generated_words.append(next_word)
        
        # Format the continuation
        continuation = ' '.join(generated_words)
        
        # If the original text ended with punctuation, start with uppercase
        if input_text[-1] in '.!?':
            continuation = continuation[0].upper() + continuation[1:]
        
        return input_text + ' ' + continuation

    def generate_word_from_context(self, prev_word, temperature):
        """
        Generate a word based on previous context
        
        :param prev_word: Previous word for context
        :param temperature: Temperature parameter for randomness
        :return: Generated word
        """
        if prev_word in self.transition_probabilities:
            probs = self.transition_probabilities[prev_word]
            
            # Apply temperature
            adjusted_probs = {
                word: np.power(prob, 1/temperature)
                for word, prob in probs.items()
            }
            # Normalize
            total = sum(adjusted_probs.values())
            adjusted_probs = {
                word: prob/total 
                for word, prob in adjusted_probs.items()
            }
            
            return random.choices(
                list(adjusted_probs.keys()),
                weights=list(adjusted_probs.values())
            )[0]
        
        # Fallback to random category word
        categories = list(self.categories.keys())
        category = random.choice(categories)
        return self.weighted_choice(category)
    
    def analyze_language_models(self):
        """
        Analyze and print details of language models
        """
        print("\nLanguage Model Analysis:")
        
        # Unigram analysis
        print("\nTop 10 Unigram Probabilities:")
        sorted_unigrams = sorted(
            self.ngram_models['unigram'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        for word, prob in sorted_unigrams:
            print(f"{word}: {prob:.4f}")
        
        # Bigram analysis
        print("\nSample Bigram Transition Probabilities:")
        sample_bigrams = list(self.transition_probabilities.items())[:5]
        for prev_word, transitions in sample_bigrams:
            print(f"\nWord: {prev_word}")
            sorted_transitions = sorted(
                transitions.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            for next_word, prob in sorted_transitions:
                print(f"  -> {next_word}: {prob:.4f}")
        
        # Forward context analysis
        print("\nSample Forward Context Probabilities:")
        sample_contexts = list(self.forward_context.items())[:3]
        for context, next_words in sample_contexts:
            print(f"\nContext: {' '.join(context)}")
            sorted_next = sorted(
                next_words.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            for word, prob in sorted_next:
                print(f"  -> {word}: {prob:.4f}")

def main():
    while True:
        choice = input("Choose an option:\n1. Train new model\n2. Continue with existing model\nChoice (1/2): ").strip()
        generator = NaturalTextGenerator()

        if choice == "1":
            train_probs()
            # After training, initialize the generator with the new probabilities
        elif choice == "2":
            # Main interaction loop - moved outside the if/elif blocks
            print("\nEntering text generation mode. Type your prompts:")
            while True:
                try:
                    user_input = input("User: ")
                    if user_input.lower() in ['quit', 'exit']:
                        break
                    continued_text = generator.continue_text(user_input, num_words=150, temperature=0.5)
                    print("AI:", continued_text)
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
                except Exception as e:
                    print(f"Error occurred: {e}")
                    continue
        else:
            print("Invalid choice. Exiting...")
            return
    
if __name__ == "__main__":
    main()
