import requests
import zipfile
import numpy as np
import pickle
import os
import io

# Default URLs (Module Constants)
WORD_LIST_URL = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-usa-no-swears.txt"
GLOVE_ZIP_URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"

def generate_embeddings_file(output_path):
    """
    Downloads word lists and GloVe vectors, computes normalized embeddings,
    and pickles them to the specified output_path.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Download Word List
    print("1. Downloading target word list...")
    common_words = set(requests.get(WORD_LIST_URL).text.strip().split('\n')[:10000])
    print(f"   Targeting {len(common_words)} words.")

    # Download & Process GloVe Vectors
    print("2. Downloading GloVe vectors (approx 100MB)...")
    response = requests.get(GLOVE_ZIP_URL)
    
    print("3. Extracting and Parsing vectors...")
    embeddings = {}
    
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        with z.open('glove.6B.100d.txt') as f:
            for line in f:
                parts = line.decode('utf-8').split()
                word = parts[0]
                
                if word in common_words:
                    vector = np.array(parts[1:], dtype='float32')
                    norm = np.linalg.norm(vector)
                    if norm > 0:
                        embeddings[word] = vector / norm

    print(f"4. Mapped {len(embeddings)} words.")

    with open(output_path, "wb") as f:
        pickle.dump(embeddings, f)
        
    print(f"5. Saved to {output_path}")

if __name__ == "__main__":
    generate_embeddings_file("data/embeddings.pkl")