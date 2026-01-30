import numpy as np
import pickle
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import os
import random

# Config defaults
K = 35
K_RANDOM = 5

def load_vectors(embedding_file):
    if not os.path.exists(embedding_file):
        raise FileNotFoundError(f"{embedding_file} not found.")
    
    with open(embedding_file, "rb") as f:
        data = pickle.load(f)
    words = list(data.keys())
    vectors = np.array(list(data.values()), dtype='float32')
    return words, vectors

def build_small_world(words, vectors):
    num_nodes = len(words)
    
    # 0. Normalize vectors first.
    # This ensures that Cosine Similarity = Dot Product, speeding up the loop significantly.
    print("0. Normalizing vectors for fast distance calculation...")
    vectors = normalize(vectors, axis=1, norm='l2')

    print(f"1. Finding {K} nearest neighbors for {num_nodes} words...")
    nbrs = NearestNeighbors(n_neighbors=K+1, metric='euclidean', n_jobs=-1) 
    # Note: With normalized vectors, Euclidean ranking is identical to Cosine ranking
    # but often faster in sklearn.
    nbrs.fit(vectors)
    distances, indices = nbrs.kneighbors(vectors)

    print(f"2. Building Directed Graph ({K} NN + {K_RANDOM} Random edges per node)...")
    G = nx.DiGraph()
    for w in words: G.add_node(w)

    count = 0
    for i in range(num_nodes):
        word_a = words[i]
        vec_a = vectors[i]
        
        # --- A. Add K Nearest Neighbors ---
        # Skip indices[i][0] because it is the node itself
        candidates_idx = indices[i][1:]
        
        # Keep track of neighbors we've added to avoid duplicates in the random step
        existing_neighbors = set(candidates_idx)
        existing_neighbors.add(i) # Don't connect to self

        for target_id in candidates_idx:
            word_b = words[target_id]
            vec_b = vectors[target_id]
            
            # Since vectors are normalized: Sim = Dot Product
            sim = np.dot(vec_a, vec_b)
            G.add_edge(word_a, word_b, weight=sim)

        # --- B. Add K_RANDOM Long-Range Edges ---
        for _ in range(K_RANDOM):
            # 1. Pick a random target that isn't already a neighbor
            while True:
                rand_idx = random.randint(0, num_nodes - 1)
                if rand_idx not in existing_neighbors:
                    break
            
            # 2. Add to existing set to ensure unique random choices
            existing_neighbors.add(rand_idx)
            
            # 3. Calculate distance/weight on the fly
            word_rand = words[rand_idx]
            vec_rand = vectors[rand_idx]
            
            # Calculate Cosine Similarity on the fly
            sim_rand = np.dot(vec_a, vec_rand)
            
            # 4. Add the edge
            G.add_edge(word_a, word_rand, weight=sim_rand)

        count += 1
        if count % 1000 == 0:
            print(f"   Processed {count}/{num_nodes} nodes...")
            
    return G

def analyze_graph(G):
    print("\n--- Graph Statistics ---")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    
    if nx.is_strongly_connected(G):
        print("Graph is strongly connected (every node can reach every other node).")
    elif nx.is_weakly_connected(G):
        print("Graph is weakly connected (connected if we ignore direction).")
        # Calculate strongly connected components to see fragmentation
        scc_count = nx.number_strongly_connected_components(G)
        print(f"Graph has {scc_count} strongly connected components.")
    else:
        print("Graph is disconnected.")


def create_graph_from_embeddings(embedding_path, output_graph_path):
    """
    Loads embeddings from embedding_path, builds the small world graph,
    and saves the GEXF to output_graph_path.
    """
    try:
        words, vectors = load_vectors(embedding_path)
        G = build_small_world(words, vectors)
        analyze_graph(G)
        
        print(f"\nSaving to {output_graph_path}...")
        nx.write_gexf(G, output_graph_path)
        print("Done.")
    except Exception as e:
        print(f"Error generating graph: {e}")
        # sys.exit(1) # Commented out to prevent notebook kernel death if running interactively

if __name__ == "__main__":
    create_graph_from_embeddings("data/embeddings.pkl", "data/small_world.gexf")