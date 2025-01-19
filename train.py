import random
import re
import pickle
from typing import List, Dict, Tuple, Any
from collections import Counter

KB_MEMORY = -1

class SemanticGenerator:
    def __init__(self):
        self.templates = []
        self.words = {
            'nouns': Counter(),
            'verbs': Counter(),
            'adjectives': Counter(),
            'adverbs': Counter(),
            'prepositions': Counter({'in': 1, 'on': 1, 'at': 1, 'through': 1, 
                                   'under': 1, 'over': 1, 'with': 1, 'by': 1}),
            'determiners': Counter({'the': 1, 'a': 1, 'an': 1, 'this': 1, 
                                  'that': 1, 'these': 1, 'those': 1, 'my': 1, 
                                  'your': 1, 'their': 1})
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

def run_interactive():
    """Run interactive text input session."""
    generator = SemanticGenerator()
    
    print("Welcome to the Probabilistic Semantic Text Generator!")
    
    # Try to load base text if available
    try:
        with open("test.txt", "r", encoding="utf-8") as f:
            text = ' '.join(f.read().split())[:KB_MEMORY]
            generator.learn_from_text(text)
            print("\nLoaded base patterns from file.")
    except FileNotFoundError:
        print("\nNo base file found. Starting with empty patterns.")
 
  
    print(generator.generate_text(10))
    
    # Optional: Dump probabilities after each input
    generator.dump_probabilities()

if __name__ == "__main__":
    run_interactive()
