"""
Document-based Text Processor
---------------------------
Analyzes text from document files using:
S → D : Syntax implies Data match
L ↔ S : Lexical binding equivalent to Syntax
D ≡ XOR : Data matches XOR condition
∀ → D : All matches imply Data result
"""

import logging
from typing import Dict, Set, List, Optional, Union, Any
from pathlib import Path
import json

class DocumentProcessor:
    def __init__(self):
        self.data = ""  # D: The text data
        self.vocab_files = {}  # Dictionary to store multiple vocab files and their data
        self.lexical_bindings = {}  # L: Lexical bindings per vocab file
        self.combined_vocab = set()  # Combined vocabulary from all files
        self.syntax_patterns = set()  # S: User-defined syntax patterns
        self.results_history = []  # Store processing history
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_document(self, filename: str) -> bool:
        """Load the document to analyze"""
        try:
            with open(filename, 'r') as file:
                self.data = file.read().lower()
            self.logger.info(f"Successfully loaded document: {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading document {filename}: {e}")
            return False

    def load_vocab_file(self, filename: str, category: Optional[str] = None) -> bool:
        """Load vocabulary from file with optional category"""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                content = file.read()
                vocab_words = set(word.strip().lower() 
                                for word in content.split('\n') 
                                if word.strip())
            
            category = category or Path(filename).stem
            self.vocab_files[category] = {
                'filename': filename,
                'words': vocab_words,
                'count': len(vocab_words)
            }
            
            self.combined_vocab.update(vocab_words)
            self.logger.info(f"Loaded vocabulary file {filename} with {len(vocab_words)} words")
            return True
        except Exception as e:
            self.logger.error(f"Error loading vocabulary file {filename}: {e}")
            return False

    def interactive_syntax_input(self) -> None:
        """Get syntax patterns from user"""
        print("\nEnter syntax patterns to search in the document (one per line, press Enter twice to finish):")
        while True:
            pattern = input().strip()
            if not pattern:
                break
            self.add_syntax_pattern(pattern)
            print(f"Added pattern: {pattern}")

    def add_syntax_pattern(self, pattern: str) -> None:
        """Add a user-defined syntax pattern"""
        normalized_pattern = pattern.strip().lower()
        if normalized_pattern:
            self.syntax_patterns.add(normalized_pattern)
            self.logger.info(f"Added syntax pattern: {normalized_pattern}")

    def get_matching_sentences(self, pattern: Optional[str] = None) -> List[str]:
        """Get sentences matching a pattern or all sentences if no pattern provided"""
        sentences = [s.strip() for s in self.data.split('.') if s.strip()]
        if pattern:
            return [s for s in sentences if pattern.lower() in s.lower()]
        return sentences

    def process_document(self) -> Dict[str, Any]:
        """Process the loaded document with all patterns"""
        if not self.data:
            return {"error": "No document loaded"}

        results = {
            "patterns": {},
            "sentences": [],
            "summary": {
                "total_sentences": len(self.get_matching_sentences()),
                "patterns_found": 0,
                "vocab_matches": {}
            }
        }

        # Process each syntax pattern
        for pattern in self.syntax_patterns:
            matching_sentences = self.get_matching_sentences(pattern)
            if matching_sentences:
                results["patterns"][pattern] = {
                    "count": len(matching_sentences),
                    "sentences": matching_sentences
                }
                results["summary"]["patterns_found"] += 1

                # Process vocabulary matches for each sentence
                for sentence in matching_sentences:
                    sentence_result = self.process_sentence(sentence)
                    if sentence_result not in results["sentences"]:
                        results["sentences"].append(sentence_result)

        return results

    def process_sentence(self, sentence: str) -> Dict[str, Any]:
        """Process a single sentence"""
        words = set(sentence.split())
        lexical_matches = {}
        
        # Find vocabulary matches
        for category, vocab_info in self.vocab_files.items():
            matches = list(vocab_info['words'].intersection(words))
            if matches:
                lexical_matches[category] = matches

        # Find XOR patterns
        xor_matches = []
        sentence_words = sentence.split()
        for i in range(len(sentence_words) - 1):
            word1, word2 = sentence_words[i], sentence_words[i + 1]
            for category, vocab_words in self.vocab_files.items():
                word1_in = word1 in vocab_words['words']
                word2_in = word2 in vocab_words['words']
                if word1_in != word2_in:  # XOR condition
                    xor_matches.append({
                        "pair": (word1, word2),
                        "category": category
                    })

        return {
            "sentence": sentence,
            "lexical_matches": lexical_matches,
            "xor_patterns": xor_matches
        }

    def display_results(self, results: Dict[str, Any]) -> None:
        """Display processing results"""
        print("\nDocument Analysis Results:")
        print("=========================")
        
        print(f"\nTotal Sentences: {results['summary']['total_sentences']}")
        print(f"Patterns Found: {results['summary']['patterns_found']}")
        
        print("\nPattern Matches:")
        for pattern, info in results["patterns"].items():
            print(f"\nPattern '{pattern}' found in {info['count']} sentences:")
            for sentence in info["sentences"]:
                print(f"- {sentence}")
        
        print("\nSentence Analysis:")
        for sentence_result in results["sentences"]:
            print(f"\nSentence: {sentence_result['sentence']}")
            
            if sentence_result["lexical_matches"]:
                print("Vocabulary Matches:")
                for category, matches in sentence_result["lexical_matches"].items():
                    print(f"- {category}: {', '.join(matches)}")
            
            if sentence_result["xor_patterns"]:
                print("XOR Patterns:")
                for xor in sentence_result["xor_patterns"]:
                    print(f"- {xor['pair'][0]} <-> {xor['pair'][1]} ({xor['category']})")

def main():
    processor = DocumentProcessor()
    
    # Load document to analyze
    document_file = "test.txt"  # Replace with your document file
    if not processor.load_document(document_file):
        print(f"Failed to load document: {document_file}")
        return

    # Load vocabulary files
    vocab_files = {
        'descriptions.txt': 'descriptions',
        'actions.txt': 'actions'
    }
    
    for filename, category in vocab_files.items():
        if not processor.load_vocab_file(filename, category):
            print(f"Failed to load vocabulary file: {filename}")
            return

    # Get syntax patterns from user
    processor.interactive_syntax_input()

    # Process document
    results = processor.process_document()
    
    # Display results
    processor.display_results(results)

    # Save results
    try:
        with open('analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print("\nResults saved to analysis_results.json")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    main()