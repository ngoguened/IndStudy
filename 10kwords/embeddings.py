import requests
import zipfile
import numpy as np
import pickle
import os
import io
import time
from dotenv import load_dotenv

load_dotenv()

WORD_LIST_URL = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-usa-no-swears.txt"
GLOVE_ZIP_URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"

def get_common_words():
    """Downloads and returns the list of 10k common words."""
    print("Downloading target word list...")
    common_words_list = [w for w in requests.get(WORD_LIST_URL).text.strip().split('\n')[:10000] if w]
    return common_words_list, set(common_words_list)

def generate_embeddings(embedding_type="gemini", gemini_api_key=None):
    """
    Generates embeddings for 10k words.
    
    Args:
        embedding_type (str): 'gemini' or 'glove'
        gemini_api_key (str): Optional API key if using gemini. Falls back to env var.
        
    Returns:
        dict: Word to normalized numpy embedding vector.
    """
    common_words_list, common_words = get_common_words()
    print(f"Targeting {len(common_words)} words.")
    embeddings = {}

    if embedding_type == "gemini":
        print("Using Gemini API for embeddings...")
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("Please install the google-genai package to use Gemini. (pip install google-genai)")
        
        if not gemini_api_key:
            gemini_api_key = os.environ.get("GEMINI_API_KEY")
            if not gemini_api_key:
                raise ValueError("A Gemini API key must be provided or set in your .env file as GEMINI_API_KEY.")
        
        client = genai.Client(api_key=gemini_api_key)
        chunk_size = 100
        model = "models/gemini-embedding-001"
        
        print(f"Batch processing in chunks of {chunk_size}...")
        base_delay = 5

        for i in range(0, len(common_words_list), chunk_size):
            print(f"Chunk {i}...")
            chunk = common_words_list[i:i + chunk_size]
            success = False
            retries = 0

            while not success:
                try:
                    response = client.models.embed_content(
                        model=model,
                        contents=chunk,
                        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY") 
                    )
                    
                    for word, embed_obj in zip(chunk, response.embeddings):
                        vector = np.array(embed_obj.values, dtype='float32')
                        norm = np.linalg.norm(vector)
                        if norm > 0:
                            embeddings[word] = vector / norm
                    
                    success = True
                    time.sleep(2) 

                except Exception as e:
                    if "429" in str(e):
                        wait_time = base_delay * (2 ** retries)
                        print(f"[!] Quota hit. Waiting {wait_time}s before retry {retries+1}/...")
                        time.sleep(wait_time)
                        retries += 1
                    else:
                        print(f"[!] Permanent error at chunk {i}: {e}")
                        break
                        
    elif embedding_type == "glove":
        print("Downloading GloVe vectors (approx 100MB)...")
        response = requests.get(GLOVE_ZIP_URL)
        
        print("Extracting and Parsing GloVe vectors...")
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
    else:
        raise ValueError("Invalid embedding_type. Choose 'gemini' or 'glove'.")

    print(f"Mapped {len(embeddings)} words.")
    return embeddings

def save_embeddings(embeddings, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(output_path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Saved embeddings to {output_path}")

def load_embeddings(embedding_file):
    if not os.path.exists(embedding_file):
        raise FileNotFoundError(f"{embedding_file} not found.")
    
    with open(embedding_file, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings

def get_words_and_vectors(embeddings, filter_dict=True):
    """Converts the embedding dict into list of words and numpy array of vectors, optionally filtering non-dictionary words."""
    if filter_dict:
        try:
            import nltk
            from nltk.corpus import words as nltk_words
            try:
                valid_words = set(nltk_words.words())
            except LookupError:
                nltk.download('words', quiet=True)
                valid_words = set(nltk_words.words())
            
            # Words in the nltk corpus are typically lowercase
            valid_words_lower = {w.lower() for w in valid_words}
            
            filtered_words = []
            filtered_vectors = []
            for word, vector in embeddings.items():
                if word.lower() in valid_words_lower:
                    filtered_words.append(word)
                    filtered_vectors.append(vector)
            
            words = filtered_words
            vectors = np.array(filtered_vectors, dtype='float32')
            print(f"Filtered dictionary: kept {len(words)} out of {len(embeddings)} words.")
        except ImportError:
            print("Warning: nltk not installed. Skipping dictionary filtering.")
            words = list(embeddings.keys())
            vectors = np.array(list(embeddings.values()), dtype='float32')
    else:
        words = list(embeddings.keys())
        vectors = np.array(list(embeddings.values()), dtype='float32')
        
    return words, vectors
