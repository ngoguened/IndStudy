import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import random

def generate_graph(words, vectors, k, n=0, algorithm="relative_neighborhood", alpha=1.0):
    """
    Builds a small-world graph using one of three specific strategies:
    
    1. 'relative_neighborhood': Pure k-NN graph (Node -> Neighbors)
    2. 'k_nn+n_random': Directed k-NN edges + Uniform Random Long-Range 
    3. 'inv_knn+n_probabilistic': Inverted KNN (Neighbors -> Node) + Weighted Probabilistic Long-Range
    Alpha is used in 'inv_knn+n_probabilistic' to adjust the probability distribution. default=1.0.
    """
    print(f"--- Building Graph: {len(words)} nodes | Mode={algorithm} | k: {k} | n: {n} ---")
    
    # Ensure vectors are normalized
    vectors = normalize(vectors, axis=1, norm='l2')
    
    G = nx.DiGraph()
    G.add_nodes_from(words)

    if algorithm == 'relative_neighborhood':
        _add_knn_edges(G, words, vectors, k, inverted=False)
        
    elif algorithm == 'k_nn+n_random':
        knn_neighbors = _add_knn_edges(G, words, vectors, k, inverted=False)
        _add_random_edges(G, words, vectors, n, knn_neighbors)
        
    elif algorithm == 'inv_knn+n_probabilistic':
        _add_knn_edges(G, words, vectors, k, inverted=True)
        # Using a fixed chunk size for probabilistic memory efficiency
        _add_probabilistic_edges(G, words, vectors, n, alpha=alpha, chunk_size=1000)
        
    else:
        raise ValueError(f"Unknown graph algorithm: {algorithm}")

    print(f"--- Complete: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges ---")
    return G

def _add_knn_edges(G, words, vectors, k, inverted=False):
    """
    Adds K-Nearest Neighbor edges.
    Returns standard neighborhood sets (useful for exclusion in random algorithm).
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
                G.add_edge(current_word, neighbor_word, weight=float(weight))
            else:
                G.add_edge(neighbor_word, current_word, weight=float(weight))
                
    return neighbor_sets

def _add_random_edges(G, words, vectors, n, existing_neighbors):
    """
    Adds n Uniform Random edges per node.
    """
    num_nodes = len(words)
    
    for i in range(num_nodes):
        source_word = words[i]
        source_vec = vectors[i]
        exclude = existing_neighbors[i]
        
        added_count = 0
        while added_count < n:
            candidate = random.randint(0, num_nodes - 1)
            
            if candidate != i and candidate not in exclude:
                target_word = words[candidate]
                weight = np.dot(source_vec, vectors[candidate])
                
                G.add_edge(source_word, target_word, weight=float(weight))
                exclude.add(candidate)
                added_count += 1

def _add_probabilistic_edges(G, words, vectors, n, alpha=1.0, chunk_size=1000):
    """
    Adds n long-range edges per node based on similarity probability adjusted by an alpha parameter.
    If alpha > 1, the probability distribution becomes sharper (favoring highly similar nodes).
    If alpha < 1, the distribution becomes more uniform.
    """
    for i in range(0, len(words), chunk_size):
        batch = vectors[i : i + chunk_size]
        sims = np.maximum(batch @ vectors.T, 0)
        
        for r in range(len(batch)):
            global_idx = i + r
            sims[r, global_idx] = 0
            
        scaled_sims = sims ** alpha if alpha != 1.0 else sims
            
        row_sums = scaled_sims.sum(axis=1, keepdims=True) + 1e-9
        probs = scaled_sims / row_sums
        
        for r in range(len(batch)):
            if probs[r].sum() < 0.01:
                continue
                
            targets = np.random.choice(len(words), size=n, replace=False, p=probs[r])
            source_word = words[i + r]
            
            for t_idx in targets:
                weight = sims[r, t_idx]
                G.add_edge(source_word, words[t_idx], weight=float(weight))

def save_graph(G, output_path):
    nx.write_gexf(G, output_path)
    print(f"Saved graph to {output_path}")

def load_graph(graph_path):
    return nx.read_gexf(graph_path)

def analyze_graph(G):
    print("\n--- Graph Statistics ---")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    
    if nx.is_strongly_connected(G):
        print("Graph is strongly connected.")
    elif nx.is_weakly_connected(G):
        print("Graph is weakly connected.")
        scc_count = nx.number_strongly_connected_components(G)
        print(f"Graph has {scc_count} strongly connected components.")
    else:
        print("Graph is disconnected.")
