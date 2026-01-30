from sklearn.metrics.pairwise import cosine_similarity

MAX_STEPS = 50

def get_cosine_sim(vec_a, vec_b):
    """Calculates cosine similarity between two vectors (returns float -1 to 1)."""
    va = vec_a.reshape(1, -1)
    vb = vec_b.reshape(1, -1)
    return cosine_similarity(va, vb)[0][0]

def play_game(G, embeddings, start_word, target_word):
    """
    Plays one game using Greedy Heuristic:
    Choose the neighbor semantically closest to the target_word.
    """
    current_word = start_word
    path = [current_word]
    visited = {current_word} # Basic loop prevention
    
    target_vec = embeddings[target_word]

    for _ in range(MAX_STEPS):
        if current_word == target_word:
            return True, path

        # Get neighbors
        neighbors = list(G.neighbors(current_word))
        
        # Filter neighbors we have already visited in this specific session
        # (Prevents moving Back -> Forth -> Back)
        valid_neighbors = [n for n in neighbors if n not in visited]
        
        if not valid_neighbors:
            # Dead end
            return False, path

        # --- HEURISTIC: Greedy Choice ---
        best_neighbor = None
        best_score = -2.0
        
        for nbr in valid_neighbors:
            if nbr in embeddings:
                nbr_vec = embeddings[nbr]
                score = get_cosine_sim(nbr_vec, target_vec)
                
                if score > best_score:
                    best_score = score
                    best_neighbor = nbr
        
        if best_neighbor is None:
            # Should not happen unless embeddings are missing
            return False, path

        # Make the move
        current_word = best_neighbor
        path.append(current_word)
        visited.add(current_word)
    
    # Exceeded max steps
    return False, path
