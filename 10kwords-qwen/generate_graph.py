import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import sys
import os
import requests
import torch

# Config defaults
WORD_LIST_URL = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt"

K_CANDIDATES = 20
M_EDGES = 6

def load_vectors(embedding_file):
    if not os.path.exists(embedding_file):
        raise FileNotFoundError(f"{embedding_file} not found.")
    
    with open(embedding_file, "rb") as f:
        data = torch.load(f)
    words = requests.get(WORD_LIST_URL).text.strip().split('\n')[:10000]
    vectors = np.array(data, dtype='float32')
    return words, vectors

def build_small_world(words, vectors):
    num_nodes = len(words)
    print(f"1. Finding {K_CANDIDATES} nearest neighbors for {num_nodes} words...")
    
    nbrs = NearestNeighbors(n_neighbors=K_CANDIDATES+1, metric='cosine', n_jobs=-1)
    nbrs.fit(vectors)
    distances, indices = nbrs.kneighbors(vectors)

    print("2. Pruning edges using Relative Neighborhood Heuristic...")
    G = nx.Graph()
    for w in words: G.add_node(w)

    count = 0
    for i in range(num_nodes):
        candidates_idx = indices[i][1:]
        candidates_dist = distances[i][1:]
        selected_neighbors = []
        
        for j, candidate_id in enumerate(candidates_idx):
            if len(selected_neighbors) >= M_EDGES: break
            dist_to_source = candidates_dist[j]
            is_redundant = False
            vec_candidate = vectors[candidate_id]
            
            for selected_id in selected_neighbors:
                vec_selected = vectors[selected_id]
                # Cosine Dist = 1 - Dot
                dist_between_neighbors = 1.0 - np.dot(vec_candidate, vec_selected)
                
                # if dist_between_neighbors < dist_to_source:
                #     is_redundant = True
                #     break
            
            if not is_redundant:
                selected_neighbors.append(candidate_id)
                word_a = words[i]
                word_b = words[candidate_id]
                G.add_edge(word_a, word_b, weight=(1.0 - dist_to_source))

        count += 1
        if count % 1000 == 0:
            print(f"   Processed {count}/{num_nodes} nodes...")
    return G

def analyze_graph(G):
    print("\n--- Graph Statistics ---")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    if nx.is_connected(G):
        print("Graph is fully connected (Navigable!).")
    else:
        components = list(nx.connected_components(G))
        print(f"Graph has {len(components)} disconnected components.")

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
        sys.exit(1)

if __name__ == "__main__":
    create_graph_from_embeddings("data/embeddings.pth", "data/small_world.gexf")