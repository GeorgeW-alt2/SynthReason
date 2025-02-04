from tqdm import tqdm
KB_limit = 99999
translation_dict = {"what": "descriptions.txt", "how": "actions.txt"}
print(translation_dict)

while True:
    print("\nOptions:")
    print("1. Build memory")
    print("2. Execute")

    choice = input("\nEnter your choice (1-2): ").strip()

    if choice == "1":
        filename = input("Enter training file path: ")
        with open(filename, 'r', encoding='utf-8') as f:
            text = ' '.join(f.read().split()[:KB_limit])

        print("\nBuilding memory...")
        sentences = text.split(".")  # Split into sentences

        with open("memory.txt", "a") as file:
            for sentence in tqdm(sentences, desc="Processing Sentences", unit="sentence"):
                temp = "["
                for category, item in translation_dict.items():
                    with open(item, "r", encoding="utf-8") as f:
                        vocab = {line.strip() for line in f.readlines()}  # Use a set for faster lookup

                    for word in sentence.split():
                        if word in vocab:
                            temp += f":{category}>{word}"
                temp += ":]\n"
                if len(temp) > 3:
                    file.write(temp)

    elif choice == "2":
        while True:
            query = input("Enter command: ").split()
            out = set()

            if len(query) < 2:
                print("Please enter a category (e.g., 'what') and a word (e.g., 'helped').")
                continue

            category_to_search = query[0] 
            search_words = query    

            if category_to_search not in translation_dict:
                print(f"Invalid category: {category_to_search}")
                continue

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

                    # syntactic arguments example
                    if search_word in words_in_entry.get("how", set()):
                        out.update(words_in_entry.get("what", set()))
                    if search_word in words_in_entry.get("what", set()):
                        out.update(words_in_entry.get("how", set()))

            print("[", ' '.join(out), "]")
            print()
