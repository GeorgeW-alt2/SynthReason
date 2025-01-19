import pickle
import random
import numpy as np
from collections import defaultdict

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
    # Create advanced text generator
    generator = NaturalTextGenerator()
    while True:
        try:
            continued_text = generator.continue_text(input("User: "), num_words=150, temperature=0.5)
            print("AI:", continued_text)
        except:
            False
if __name__ == "__main__":
    main()
