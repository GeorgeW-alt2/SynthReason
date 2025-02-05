from tqdm import tqdm
import threading
from queue import Queue
from typing import Dict, Set, List
import os
from collections import deque

KB_limit = -1
BUFFER_SIZE = 1000  # Number of results to buffer before writing
translation_dict = {
    "what": "descriptions.txt",
    "how": "actions.txt",
    "do": "verbs.txt",
    "describe": "picturable.txt",
    "grade": "adj.txt",
    "form": "prep.txt"
}

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

def process_sentence(sentence: str, vocab_cache: VocabularyCache) -> str:
    temp = "["
    for category in translation_dict.keys():
        vocab = vocab_cache.get_vocabulary(category)
        for word in sentence.split():
            if word in vocab:
                temp += f":{category}>{word}"
    temp += ":]\n"
    return temp if len(temp) > 3 else ""

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
            # Put the results back in the buffer
            self.buffer.extendleft(reversed(list(self.buffer)))
    
    def final_flush(self) -> None:
        with self.lock:
            self.flush_buffer()

def worker(sentence_queue: Queue, result_buffer: ResultBuffer, vocab_cache: VocabularyCache, pbar: tqdm) -> None:
    while True:
        try:
            sentence = sentence_queue.get_nowait()
        except Queue.Empty:
            break
            
        if sentence is None:  # Poison pill
            break
            
        result = process_sentence(sentence, vocab_cache)
        if result:
            result_buffer.add_result(result)
        pbar.update(1)
        sentence_queue.task_done()

def build_memory_multithreaded(filename: str, num_threads: int = None) -> None:
    if num_threads is None:
        num_threads = os.cpu_count() or 4
    
    print(f"\nBuilding memory using {num_threads} threads...")
    
    # Initialize queue and result buffer
    sentence_queue = Queue()
    result_buffer = ResultBuffer("memory.txt")
    
    # Load vocabularies once
    vocab_cache = VocabularyCache(translation_dict)
    
    # Read and queue sentences
    with open(filename, 'r', encoding='utf-8') as f:
        text = ' '.join(f.read().split()[:KB_limit])
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    
    # Queue all sentences first
    for sentence in sentences:
        sentence_queue.put(sentence)
    
    # Add poison pills
    for _ in range(num_threads):
        sentence_queue.put(None)
    
    # Initialize progress bar
    pbar = tqdm(total=len(sentences), desc="Processing Sentences", unit="sentence")
    
    # Start worker threads
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(
            target=worker,
            args=(sentence_queue, result_buffer, vocab_cache, pbar)
        )
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Final flush of any remaining results
    result_buffer.final_flush()
    pbar.close()
    
    print(f"\nMemory building complete. Buffer flushed {result_buffer.flush_count} times.")

def main():
    print(translation_dict)
    
    while True:
        print("\nOptions:")
        print("1. Build memory")
        print("2. Execute")
        print("3. Exit")

        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == "1":
            filename = input("Enter training file path: ")
            num_threads = input("Enter number of threads (press Enter for auto): ").strip()
            num_threads = int(num_threads) if num_threads else None
            build_memory_multithreaded(filename, num_threads)
            
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

                        # Process relationships
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
            print("Exiting program...")
            break

if __name__ == "__main__":
    main()