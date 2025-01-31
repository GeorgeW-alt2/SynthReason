import logging
from typing import Dict, Set, List, Optional, Any
from pathlib import Path
import json

class DocumentProcessor:
    def __init__(self):
        self.data = ""
        self.vocab_files = {}
        self.combined_vocab = set()
        self.syntax_patterns = set()
        self.category_filter = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def set_category_filter(self, category: str) -> None:
        self.category_filter = category

    def load_document(self, filename: str) -> bool:
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                self.data = file.read().lower()
            return True
        except Exception as e:
            self.logger.error(f"Error loading document {filename}: {e}")
            return False

    def load_vocab_file(self, filename: str, category: Optional[str] = None) -> bool:
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                vocab_words = set(word.strip().lower() for word in file if word.strip())
            
            category = category or Path(filename).stem
            self.vocab_files[category] = {'words': vocab_words, 'count': len(vocab_words)}
            self.combined_vocab.update(vocab_words)
            return True
        except Exception as e:
            self.logger.error(f"Error loading vocabulary: {e}")
            return False

    def add_syntax_pattern(self, pattern: str) -> None:
        self.syntax_patterns.add(pattern.strip().lower())

    def get_matching_sentences(self, pattern: Optional[str] = None) -> List[str]:
        sentences = [s.strip() for s in self.data.split('.') if s.strip()]
        return [s for s in sentences if not pattern or pattern.lower() in s.lower()]

    def process_sentence(self, sentence: str) -> Dict[str, Any]:
        words = set(sentence.split())
        if self.category_filter:
            if self.category_filter not in self.vocab_files:
                return {}
            vocab_info = self.vocab_files[self.category_filter]
            matches = list(vocab_info['words'].intersection(words))
            lexical_matches = {self.category_filter: matches} if matches else {}
        else:
            lexical_matches = {category: list(vocab_info['words'].intersection(words))
                             for category, vocab_info in self.vocab_files.items()}
            lexical_matches = {k: v for k, v in lexical_matches.items() if v}
        
        if not lexical_matches:
            return {}

        return {
            "sentence": sentence,
            "lexical_matches": lexical_matches,
            "matching_patterns": []  # Will be populated in process_document
        }

    def process_document(self) -> Dict[str, Any]:
        if not self.data:
            return {"error": "No document loaded"}

        results = {
            "matched_content": [],
            "summary": {
                "total_sentences": len(self.get_matching_sentences()),
                "total_patterns": len(self.syntax_patterns),
                "patterns_with_matches": 0
            }
        }

        processed_sentences = set()

        for pattern in self.syntax_patterns:
            matching_sentences = self.get_matching_sentences(pattern)
            if matching_sentences:
                results["summary"]["patterns_with_matches"] += 1
                
                for sentence in matching_sentences:
                    if sentence not in processed_sentences:
                        sentence_result = self.process_sentence(sentence)
                        if sentence_result:
                            ordered_result = {
                                "sentence": sentence_result["sentence"],
                                "lexical_matches": sentence_result["lexical_matches"],
                                "matching_patterns": [pattern]
                            }
                            results["matched_content"].append(ordered_result)
                            processed_sentences.add(sentence)
                    else:
                        for content in results["matched_content"]:
                            if content["sentence"] == sentence:
                                if "matching_patterns" not in content:
                                    content["matching_patterns"] = []
                                content["matching_patterns"].append(pattern)

        results["summary"]["sentences_with_matches"] = len(results["matched_content"])
        return results

    def save_new_vocab_files(self, results: Dict[str, Any]) -> None:
        """Create new vocabulary files from the lexical matches in the results, organized by pattern."""
        # Collect matches by pattern
        pattern_vocab: Dict[str, Dict[str, Set[str]]] = {}
        
        for content in results["matched_content"]:
            for pattern in content["matching_patterns"]:
                if pattern not in pattern_vocab:
                    pattern_vocab[pattern] = {}
                for category, matches in content["lexical_matches"].items():
                    if category not in pattern_vocab[pattern]:
                        pattern_vocab[pattern][category] = set()
                    pattern_vocab[pattern][category].update(matches)

        # Save each pattern's matches to a file
        for pattern in pattern_vocab:
            output_filename = f"{pattern}.txt"
            try:
                with open(output_filename, 'w', encoding='utf-8') as f:
                    # Write matches by category
                    for category, words in pattern_vocab[pattern].items():
                        for word in sorted(words):
                            f.write(f"{word}\n")
                        f.write("\n")
                self.logger.info(f"Created vocabulary file for pattern: {output_filename}")
            except Exception as e:
                self.logger.error(f"Error saving vocabulary file {output_filename}: {e}")

def main():
    processor = DocumentProcessor()
    if not all([
        processor.load_document("kb.txt"),
        processor.load_vocab_file("descriptions.txt", "what"),
        processor.load_vocab_file("actions.txt", "how")
    ]):
        return
        
    print("Enter context (blank line to finish):")
    while pattern := input().strip():
        processor.add_syntax_pattern(pattern)
        
    category = input("Enter adjunct (what/how) or press Enter for all: ").strip()
    if category:
        processor.set_category_filter(category)

    results = processor.process_document()
    
    try:
        
        # Create new vocabulary files
        processor.save_new_vocab_files(results)
        
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    main()
