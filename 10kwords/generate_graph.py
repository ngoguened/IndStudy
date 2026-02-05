import numpy as np
import pickle
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import os
import random

def load_vectors(embedding_file):
    if not os.path.exists(embedding_file):
        raise FileNotFoundError(f"{embedding_file} not found.")
    
    with open(embedding_file, "rb") as f:
        data = pickle.load(f)
    words = list(data.keys())
    vectors = np.array(list(data.values()), dtype='float32')
    return words, vectors

def build_small_world(words, vectors, k, n, algorithm='probabilistic', chunk_size=1000):
    """
    Builds a small-world graph using one of two specific strategies:
    
    1. 'probabilistic': Inverted KNN (Neighbors -> Node) + Weighted Probabilistic Long-Range
    2. 'random':        Standard KNN (Node -> Neighbors) + Uniform Random Long-Range
    """
    print(f"--- Building Graph: {len(words)} nodes | Mode={algorithm} | k: {k} | n: {n} ---")
    
    vectors = normalize(vectors, axis=1, norm='l2')
    
    G = nx.DiGraph()
    G.add_nodes_from(words)

    if algorithm == 'probabilistic':
        _add_knn_edges(G, words, vectors, k, inverted=True)
        _add_probabilistic_edges(G, words, vectors, n, chunk_size)
        
    elif algorithm == 'random':
        knn_neighbors = _add_knn_edges(G, words, vectors, k, inverted=False)
        _add_random_edges(G, words, vectors, n, knn_neighbors)
        
    else:
        raise ValueError("Algorithm must be one of the options")

    print(f"--- Complete: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges ---")
    return G

def _add_knn_edges(G, words, vectors, k, inverted=False):
    """
    Adds K-Nearest Neighbor edges with configurable direction.
    
    Parameters:
    - mode='standard': Node points TO its neighbors (Node -> Neighbor)
    - mode='inverted': Neighbors point TO the node (Neighbor -> Node)
    
    Returns:
    - A list of sets, where list[i] contains the indices of node i's neighbors.
    """
    
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='euclidean', n_jobs=-1).fit(vectors)
    _, indices = nbrs.kneighbors(vectors)
    
    neighbor_sets = []
    
    for i in range(len(words)):
        current_word = words[i]
        current_vec = vectors[i]
        
        my_neighbors_idx = indices[i][1:]
        neighbor_sets.append(set(my_neighbors_idx))
        
        for n_idx in my_neighbors_idx:
            neighbor_word = words[n_idx]
            neighbor_vec = vectors[n_idx]
            
            weight = np.dot(current_vec, neighbor_vec)
            
            if not inverted:
                G.add_edge(current_word, neighbor_word, weight=weight)
            else:
                G.add_edge(neighbor_word, current_word, weight=weight)
                
    return neighbor_sets

def _add_probabilistic_edges(G, words, vectors, k, chunk_size):
    """
    Adds long-range edges based on similarity probability.
    High similarity = Higher chance of a long-range connection.
    """
    
    for i in range(0, len(words), chunk_size):
        batch = vectors[i : i + chunk_size]
        
        sims = np.maximum(batch @ vectors.T, 0)
        
        for r in range(len(batch)):
            global_idx = i + r
            sims[r, global_idx] = 0
            
        row_sums = sims.sum(axis=1, keepdims=True) + 1e-9
        probs = sims / row_sums
        
        for r in range(len(batch)):
            if probs[r].sum() < 0.01: continue
                
            targets = np.random.choice(len(words), size=k, replace=False, p=probs[r])
            
            source_word = words[i + r]
            for t_idx in targets:
                weight = sims[r, t_idx]
                G.add_edge(source_word, words[t_idx], weight=weight)

def _add_random_edges(G, words, vectors, k, existing_neighbors):
    """
    Adds Uniform Random edges (Watts-Strogatz style).
    Ensures no self-loops and no duplicate connections.
    """
    num_nodes = len(words)
    
    for i in range(num_nodes):
        source_word = words[i]
        source_vec = vectors[i]
        
        exclude = existing_neighbors[i]
        
        added_count = 0
        while added_count < k:
            candidate = random.randint(0, num_nodes - 1)
            
            if candidate != i and candidate not in exclude:
                target_word = words[candidate]
                weight = np.dot(source_vec, vectors[candidate])
                
                G.add_edge(source_word, target_word, weight=weight)
                
                exclude.add(candidate)
                added_count += 1

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


def create_graph_from_embeddings(embedding_path, output_graph_path, k, n, algorithm='probabilistic'):
    """
    Loads embeddings from embedding_path, builds the small world graph,
    and saves the GEXF to output_graph_path.
    """
    try:
        words, vectors = load_vectors(embedding_path)
        G = build_small_world(words, vectors, k, n, algorithm)
        analyze_graph(G)
        
        print(f"\nSaving to {output_graph_path}...")
        nx.write_gexf(G, output_graph_path)
        print("Done.")
    except Exception as e:
        print(f"Error generating graph: {e}")
