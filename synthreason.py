import random
import re
from typing import List, Dict, Set, Tuple
KB_MEMORY = 999999
class SemanticGenerator:
    def __init__(self):
        self.templates = []
        self.words = {
            'nouns': set(),
            'verbs': set(),
            'adjectives': set(),
            'adverbs': set(),
            'prepositions': set(['in', 'on', 'at', 'through', 'under', 'over', 'with', 'by']),
            'determiners': set(['the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'their'])
        }
        self.endings = {'.', '!', '?'}
        
    def learn_from_text(self, text: str) -> None:
        """Learn patterns and words from input text."""
        # Clean and split text into sentences
        sentences = self._split_into_sentences(text)
        
        for sentence in sentences:
            # Extract and store the sentence template
            template = self._extract_template(sentence)
            if template:
                self.templates.append(template)
            
            # Categorize and store words
            words = sentence.lower().split()
            self._categorize_words(words)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while handling common abbreviations."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Split on sentence endings while preserving them
        sentences = []
        current = []
        
        words = text.split()
        for i, word in enumerate(words):
            current.append(word)
            
            # Check if word ends with sentence ending
            if any(word.endswith(end) for end in self.endings):
                # Check if it's not an abbreviation
                if not (len(word) <= 3 and word[0].isupper()):
                    sentences.append(' '.join(current))
                    current = []
                    
        # Add any remaining text
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
            
            # Remove punctuation for matching
            clean_word = word.rstrip('.,!?')
            
            # Try to identify word type
            if clean_word in self.words['determiners']:
                template.append(('DET', word))
            elif clean_word.endswith(('ly')) and len(clean_word) > 2:
                template.append(('ADV', word))
                self.words['adverbs'].add(clean_word)
            elif clean_word.endswith(('ed', 'ing')):
                template.append(('VERB', word))
                self.words['verbs'].add(clean_word)
            elif clean_word in self.words['prepositions']:
                template.append(('PREP', word))
            elif i > 0 and words[i-1].lower() in self.words['determiners']:
                template.append(('NOUN', word))
                self.words['nouns'].add(clean_word)
            elif i > 0 and template[-1][0] == 'NOUN':
                template.append(('VERB', word))
                self.words['verbs'].add(clean_word)
            else:
                template.append(('WORD', word))
            
            i += 1
            
        return template
    
    def _categorize_words(self, words: List[str]) -> None:
        """Categorize words based on patterns and context."""
        for i, word in enumerate(words):
            clean_word = word.lower().rstrip('.,!?')
            
            # Skip already categorized words
            if any(clean_word in wordset for wordset in self.words.values()):
                continue
                
            # Apply basic categorization rules
            if word.endswith(('ly')) and len(word) > 2:
                self.words['adverbs'].add(clean_word)
            elif word.endswith(('ed', 'ing')):
                self.words['verbs'].add(clean_word)
            elif i > 0 and words[i-1].lower() in self.words['determiners']:
                self.words['nouns'].add(clean_word)
    
    def generate_sentence(self) -> str:
        """Generate a new sentence using learned patterns."""
        if not self.templates:
            return "No patterns learned yet."
            
        # Choose a random template
        template = random.choice(self.templates)
        
        # Generate sentence from template
        sentence_parts = []
        for word_type, original in template:
            if word_type == 'DET':
                sentence_parts.append(random.choice(list(self.words['determiners'])))
            elif word_type == 'NOUN':
                sentence_parts.append(random.choice(list(self.words['nouns'])))
            elif word_type == 'VERB':
                sentence_parts.append(random.choice(list(self.words['verbs'])))
            elif word_type == 'ADV':
                if self.words['adverbs']:
                    sentence_parts.append(random.choice(list(self.words['adverbs'])))
                else:
                    sentence_parts.append(original)
            elif word_type == 'PREP':
                sentence_parts.append(random.choice(list(self.words['prepositions'])))
            else:
                sentence_parts.append(original)
        
        sentence = ' '.join(sentence_parts)
        
        # Ensure proper capitalization and ending
        sentence = sentence[0].upper() + sentence[1:]
        if not any(sentence.endswith(end) for end in self.endings):
            sentence += '.'
            
        return sentence
    
    def generate_text(self, num_sentences: int = 3) -> str:
        """Generate multiple sentences."""
        return ' '.join(self.generate_sentence() for _ in range(num_sentences))

# Example usage
if __name__ == "__main__":
    # Create generator
    generator = SemanticGenerator()
    
    # Sample text to learn from
    with open("xaa", "r", encoding="utf-8") as f:
        text = ' '.join(f.read().split())[:KB_MEMORY]
    
    # Learn patterns
    generator.learn_from_text(text)
    
    # Generate new text
    print("\nGenerated text:")
    print(generator.generate_text(103))
