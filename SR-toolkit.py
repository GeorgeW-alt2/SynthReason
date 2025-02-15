from tqdm import tqdm
import threading
from queue import Queue
from typing import Dict, Set, List, Tuple
import os
from collections import deque, defaultdict

KB_limit = -1
BUFFER_SIZE = 999999

translation_dict = {
    "what": "descriptions.txt",  # nouns (can be subjects or objects)
    "how": "actions.txt",       # adverbs
    "do": "verbs.txt",         # verbs
    "describe": "picturable.txt", # articles/determiners
    "grade": "adj.txt",        # adjectives
    "form": "prep.txt"         # prepositions
}

class SVOPattern:
    def __init__(self):
        self.subjects = defaultdict(set)       # subject -> verb
        self.verbs = defaultdict(set)          # verb -> object
        self.objects = defaultdict(set)        # object -> subject
        self.subject_object = defaultdict(set) # subject -> object

    def add_pattern(self, subject: str, verb: str, obj: str):
        self.subjects[subject].add(verb)
        self.verbs[verb].add(obj)
        self.objects[obj].add(subject)
        self.subject_object[subject].add(obj)

    def get_verbs_for_subject(self, subject: str) -> Set[str]:
        return self.subjects[subject]

    def get_objects_for_verb(self, verb: str) -> Set[str]:
        return self.verbs[verb]

    def get_subjects_for_object(self, obj: str) -> Set[str]:
        return self.objects[obj]
        
    def save_to_file(self, filename: str):
        with open(filename, "w", encoding="utf-8") as f:
            for subject in self.subjects:
                verbs = self.subjects[subject]
                for verb in verbs:
                    objects = self.verbs[verb]
                    for obj in objects:
                        f.write(f"{subject} {verb} {obj}\n")
                        
    @classmethod
    def load_from_file(cls, filename: str) -> 'SVOPattern':
        pattern = cls()
        try:
            with open(filename, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        subject, verb, obj = parts
                        pattern.add_pattern(subject, verb, obj)
            return pattern
        except FileNotFoundError:
            print(f"File {filename} not found.")
            return None
        except Exception as e:
            print(f"Error loading patterns: {e}")
            return None

class VocabularyCache:
    def __init__(self, translation_dict: Dict[str, str]):
        self.vocab_cache: Dict[str, Set[str]] = {}
        self._load_vocabularies(translation_dict)
    
    def _load_vocabularies(self, translation_dict: Dict[str, str]) -> None:
        for category, filename in translation_dict.items():
            with open(filename, 'r', encoding='utf-8') as f:
                self.vocab_cache[category] = {line.strip() for line in f.readlines()}
    
    def get_vocabulary(self, category: str) -> Set[str]:
        return self.vocab_cache.get(category, set())

def process_sentence(sentence: str, vocab_cache: VocabularyCache, svo_patterns: SVOPattern = None) -> str:
    words = sentence.split()
    temp = "["
    
    # First pass: categorize words and track positions
    word_categories = {}
    for i, word in enumerate(words):
        for category, vocab in vocab_cache.vocab_cache.items():
            if word in vocab:
                word_categories[i] = (word, category)
                temp += f":{category}>{word}"
    
    # Second pass: identify SVO patterns
    if svo_patterns is not None:
        for i in range(len(words)-2):
            if i in word_categories and i+1 in word_categories and i+2 in word_categories:
                word1, cat1 = word_categories[i]
                word2, cat2 = word_categories[i+1]
                word3, cat3 = word_categories[i+2]
                
                # Check for SVO pattern
                if cat1 == "what" and cat2 == "do" and cat3 == "what":
                    svo_patterns.add_pattern(word1, word2, word3)
    
    temp += ":]\n"
    return temp if len(temp) > 3 else ""

import random

def generate_svo_sentence(svo_patterns: SVOPattern, vocab_cache: VocabularyCache, randomize: bool = False) -> str:
    if randomize:
        # Get all subjects that have associated verbs
        valid_subjects = [subj for subj in svo_patterns.subjects.keys() if svo_patterns.subjects[subj]]
        if not valid_subjects:
            return None
            
        # Randomly select a subject
        subject = random.choice(valid_subjects)
        
        # Get verbs associated with this subject and randomly select one
        possible_verbs = list(svo_patterns.get_verbs_for_subject(subject))
        verb = random.choice(possible_verbs)
        
        # Get objects associated with this verb and randomly select one
        possible_objects = list(svo_patterns.get_objects_for_verb(verb))
        obj = random.choice(possible_objects)
        
        return f"{subject} {verb} {obj}."
    else:
        # Pattern-based SVO generation
        for subject in svo_patterns.subjects:
            verbs_for_subject = svo_patterns.get_verbs_for_subject(subject)
            if verbs_for_subject:
                for verb in verbs_for_subject:
                    objects = svo_patterns.get_objects_for_verb(verb)
                    if objects:
                        obj = next(iter(objects))
                        return f"{subject} {verb} {obj}."
    
    return None

class ResultBuffer:
    def __init__(self, output_file: str, buffer_size: int = BUFFER_SIZE):
        self.output_file = output_file
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.lock = threading.Lock()
        self.flush_count = 0
    
    def add_result(self, result: str) -> None:
        with self.lock:
            self.buffer.append(result)
            if len(self.buffer) >= self.buffer_size:
                self.flush_buffer()
    
    def flush_buffer(self) -> None:
        if not self.buffer:
            return
            
        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                while self.buffer:
                    f.write(self.buffer.popleft())
            self.flush_count += 1
        except Exception as e:
            print(f"Error writing to file: {e}")
            self.buffer.extendleft(reversed(list(self.buffer)))
    
    def final_flush(self) -> None:
        with self.lock:
            self.flush_buffer()

def worker(sentence_queue: Queue, result_buffer: ResultBuffer, 
          vocab_cache: VocabularyCache, svo_patterns: SVOPattern,
          pbar: tqdm) -> None:
    while True:
        try:
            sentence = sentence_queue.get_nowait()
        except Queue.Empty:
            break
            
        if sentence is None:
            break
            
        result = process_sentence(sentence, vocab_cache, svo_patterns)
        if result:
            result_buffer.add_result(result)
        pbar.update(1)
        sentence_queue.task_done()

def build_memory_multithreaded(filename: str, num_threads: int = None) -> SVOPattern:
    if num_threads is None:
        num_threads = os.cpu_count() or 4
    
    print(f"\nBuilding memory using {num_threads} threads...")
    
    sentence_queue = Queue()
    result_buffer = ResultBuffer("memory.txt")
    svo_patterns = SVOPattern()
    vocab_cache = VocabularyCache(translation_dict)
    
    with open(filename, 'r', encoding='utf-8') as f:
        text = ' '.join(f.read().split()[:KB_limit])
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    
    for sentence in sentences:
        sentence_queue.put(sentence)
    
    for _ in range(num_threads):
        sentence_queue.put(None)
    
    pbar = tqdm(total=len(sentences), desc="Processing Sentences", unit="sentence")
    
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(
            target=worker,
            args=(sentence_queue, result_buffer, vocab_cache, svo_patterns, pbar)
        )
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()
    
    result_buffer.final_flush()
    pbar.close()
    
    print(f"\nMemory building complete. Buffer flushed {result_buffer.flush_count} times.")
    return svo_patterns

def main():
    print(translation_dict)
    svo_patterns = None
    vocab_cache = None
    
    while True:
        print("\nOptions:")
        print("1. Build memory")
        print("2. Execute queries")
        print("3. Generate learned SVO sentence")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ").strip()
        vocab_cache = VocabularyCache(translation_dict)

        if choice == "1":
            filename = input("Enter training file path: ")
            num_threads = input("Enter number of threads (press Enter for auto): ").strip()
            num_threads = int(num_threads) if num_threads else None
            svo_patterns = build_memory_multithreaded(filename, num_threads)
            svo_patterns.save_to_file("SVO.txt")

        elif choice == "2":
            while True:
                query = input("Enter command (or 'back' to return to main menu): ").split()
                if query[0].lower() == 'back':
                    break
                    
                if len(query) < 2:
                    print("Please enter a category (e.g., 'what') and a word (e.g., 'helped').")
                    continue

                category_to_search = query[0]
                search_words = query

                if category_to_search not in translation_dict:
                    print(f"Invalid category: {category_to_search}")
                    continue

                out = set()
                with open("memory.txt", "r", encoding="utf-8") as f:
                    data = f.readlines()

                with tqdm(total=len(data), desc="Processing data", unit="segment") as pbar:
                    for sub_text in data:
                        pbar.update(1)

                        lingual = sub_text.split(":")
                        words_in_entry = {}
                        for search_word in search_words:
                            for group in lingual:
                                parts = group.split(">")
                                if len(parts) > 1:
                                    element0 = parts[0].strip()
                                    element1 = parts[1].strip()

                                    if element0 and element1:
                                        if element0 not in words_in_entry:
                                            words_in_entry[element0] = set()
                                        words_in_entry[element0].add(element1)

                        relationship_mappings = [
                            ("what", "do"),
                            ("how", "do"),
                            ("describe", "what"),
                            ("grade", "what"),
                            ("what", "how"),
                            ("describe", "grade"),
                            ("how", "what"),
                            ("form", "what"),
                            ("form", "describe"),
                            ("form", "grade"),
                            ("form", "how")
                        ]

                        for source, target in relationship_mappings:
                            if search_word in words_in_entry.get(source, set()):
                                out.update(words_in_entry.get(target, set()))

                print("[", ' '.join(out), "]")
                print()
                
        elif choice == "3":
            svo_patterns = SVOPattern.load_from_file("SVO.txt")

            if not svo_patterns or not vocab_cache:
                print("Please build memory first (option 1)")
                continue
            
            print("\nGenerating SVO sentence...")
            num_sentences = int(input("How many random sentences would you like to generate? "))
            print("\nGenerating random SVO sentences...")
            for i in range(num_sentences):
                sentence = generate_svo_sentence(svo_patterns, vocab_cache, randomize=True)
                if sentence:
                    print(f"{i+1}. {sentence}")
                else:
                    print("Could not generate a valid SVO sentence from the learned patterns.")
           
        elif choice == "4":
            print("Exiting program...")
            break

if __name__ == "__main__":
    main()
