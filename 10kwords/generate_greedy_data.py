import os
import csv
from embeddings import generate_embeddings, load_embeddings, get_words_and_vectors
from graph import generate_graph, analyze_graph
from game import play_greedy_game

def log_game_result(log_file, player_type, embedding_type, graph_alg, k, n, success, path, opt_path):
    path_len = len(path) - 1 if success else -1
    opt_len = len(opt_path) - 1
    diff = path_len - opt_len if success else -1
    path_str = " -> ".join(path)
    start_w = path[0] if path else ""
    target_w = opt_path[-1] if opt_path else ""
    
    file_exists = os.path.exists(log_file)
    with open(log_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["player", "embedding", "algorithm", "k_nn", "n_long", "start", "target", "success", "pth_len", "opt_len", "diff", "path_str"])
        writer.writerow([player_type, embedding_type, graph_alg, k, n, start_w, target_w, success, path_len, opt_len, diff, path_str])

def main():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    embed_file = os.path.join(data_dir, "embeddings_glove.pkl")
    log_file = os.path.join(data_dir, "results.csv")
        
    if not os.path.exists(embed_file):
        print(f"Generating gemini embeddings...")
        embeddings = generate_embeddings(embedding_type='gemini')
    else:
        print(f"Loading existing embeddings from {embed_file}...")
        embeddings = load_embeddings(embed_file)

    words, vectors = get_words_and_vectors(embeddings)
    
    # We want to test a matrix of k and n values
    k_vals = [0, 5, 10, 15, 20, 25, 50, 100, 200]
    n_vals = [0]
    
    algorithms = [
        'relative_neighborhood',
        'k_nn+n_random',
        'inv_knn+n_probabilistic'
    ]
    
    games_per_config = 100
    
    for alg in algorithms:
        for k in k_vals:
            for n in n_vals:
                if k == 0 and n == 0:
                    continue
                if alg == 'relative_neighborhood' and n != 0:
                    continue
                    
                print(f"\n--- Testing {alg} | k={k} | n={n} ---")
                G = generate_graph(words, vectors, k, n, algorithm=alg)
                
                print(f"Playing {games_per_config} games...")
                successes = 0
                for _ in range(games_per_config):
                    success, path, opt_path = play_greedy_game(G, embeddings)
                    if success:
                        successes += 1
                    log_game_result(log_file, "greedy", "glove", alg, k, n, success, path, opt_path)
                    
                print(f"Success rate: {successes / games_per_config:.2%}")

if __name__ == "__main__":
    main()
