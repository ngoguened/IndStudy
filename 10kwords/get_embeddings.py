import requests
import zipfile
import numpy as np
import os
import io
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm

# Default URLs (Module Constants)
WORD_LIST_URL = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt"
GLOVE_ZIP_URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

def generate_embeddings_file(output_path):
    """
    Downloads word lists and GloVe vectors, computes normalized embeddings,
    and pickles them to the specified output_path.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        use_cache=False,                          # Turns off KV Cache
        torch_dtype=torch.bfloat16,               # Required: FA2 generally needs fp16 or bf16
        device_map="auto"                         # Moves model to GPU (Required for FA2)
    )
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Download Word List
    print("1. Downloading target word list...")
    common_words = requests.get(WORD_LIST_URL).text.strip().split('\n')[:10000]
    print(f"   Targeting {len(common_words)} words.")

    inputs = tokenizer(common_words, add_special_tokens=False, return_tensors="pt", padding=True)

    # Download & Process GloVe Vectors
    # print("2. Downloading GloVe vectors (approx 100MB)...")
    # response = requests.get(GLOVE_ZIP_URL)
    
    print("3. Extracting and Parsing vectors...")

    with torch.no_grad():
        out = model(**inputs)
    embeddings = out.last_hidden_state
    print(embeddings.shape)

    # Before we were only directly taking the max pool of each embedding space. When you do that, each element of the word has equal importance
    mask = (embeddings.abs().sum(dim=-1) != 0)
    sum_embeddings = torch.sum(embeddings, dim=1)
    counts = mask.sum(dim=1, keepdim=True).float()
    mean_pooled = sum_embeddings / counts.clamp(min=1e-9)
    print(mean_pooled.shape)

   
    # embeddings = {}
    # with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    #     with z.open('glove.6B.100d.txt') as f:
    #         for line in f:
    #             parts = line.decode('utf-8').split()
    #             word = parts[0]
                
    #             if word in common_words:
    #                 vector = np.array(parts[1:], dtype='float32')
    #                 norm = np.linalg.norm(vector)
    #                 if norm > 0:
    #                     embeddings[word] = vector / norm

    print(f"4. Mapped {len(embeddings)} words.")

    with open(output_path, "wb") as f:
        torch.save(mean_pooled, f)
        
    print(f"5. Saved to {output_path}")

if __name__ == "__main__":
    generate_embeddings_file("data/embeddings.pth")