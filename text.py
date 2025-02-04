from tqdm import tqdm

translation_dict = {"what": "descriptions.txt", "how": "actions.txt"}
print(translation_dict)

while True:
    query = input("Enter command: ").split()
    # Check if the first word is in our dictionary
    if query[0] in translation_dict:
        # Load the appropriate vocabulary file
        with open(translation_dict[query[0]], "r", encoding="utf-8") as f:
            vocab = [line.strip() for line in f.readlines()]
        # Load the main data file
        with open("conscious.txt", "r", encoding="utf-8") as f:
            data = f.readlines()
        out = []
        
        # Count total number of groups for progress bar
        total_groups = sum(len(sub_text.split(":")) for sub_text in data)
        
        # Create progress bar
        with tqdm(total=total_groups, desc="Processing groups", unit="groups") as pbar:
            # Process each line in the data
            for sub_text in data:
                lingual = sub_text.split(":")
                for group in lingual:
                    pbar.update(1)  # Update progress bar for each group
                    
                    parts = group.split(">")
                    if len(parts) > 1:
                        element0 = parts[0]
                        element1 = parts[1]
                        
                        # Check each word in the query
                        for word in query:
                            # Only check if the element type matches our translation dictionary
                            if element0 in translation_dict:
                                for request in translation_dict.items():
                                    # Load the appropriate vocabulary file
                                    with open(translation_dict[query[0]], "r", encoding="utf-8") as f:
                                        vocabB = [line.strip() for line in f.readlines()]
                                    # We already loaded the vocabulary, just use it
                                    if word in vocab and element1 in vocabB and element1:
                                        out.append(element1)
                                        break
    print("[",' '.join(out),"]")
    print()