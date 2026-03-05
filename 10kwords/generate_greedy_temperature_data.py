import os
import csv
from embeddings import generate_embeddings, load_embeddings, get_words_and_vectors
from graph import generate_graph, analyze_graph
from game import play_greedy_game

def log_game_result(log_file, player_type, embedding_type, graph_alg, k, n, temp, success, path, opt_path):
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
            writer.writerow(["player", "embedding", "algorithm", "k_nn", "n_long", "temp", "start", "target", "success", "pth_len", "opt_len", "diff", "path_str"])
        writer.writerow([player_type, embedding_type, graph_alg, k, n, temp, start_w, target_w, success, path_len, opt_len, diff, path_str])

def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    embed_file = os.path.join(data_dir, "embeddings_gemini.pkl")
    log_file = os.path.join(data_dir, "results.csv")
        
    if not os.path.exists(embed_file):
        print(f"Generating gemini embeddings...")
        embeddings = generate_embeddings(embedding_type='gemini')
    else:
        print(f"Loading existing embeddings from {embed_file}...")
        embeddings = load_embeddings(embed_file)

    words, vectors = get_words_and_vectors(embeddings)
    
    # We want to test a matrix of k and n values, specifically k=15,n=10 and k=10,n=15
    configs = [
        (15, 10),
        (10, 15)
    ]
    
    temp_vals = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    alg = 'inv_knn+n_probabilistic'
    games_per_config = 100
    
    for k, n in configs:
        for temp in temp_vals:
            print(f"\n--- Testing {alg} | k={k} | n={n} | temp={temp} ---")
            # Passing temp to alpha parameter in generate_graph
            G = generate_graph(words, vectors, k, n, algorithm=alg, alpha=temp)
            
            print(f"Playing {games_per_config} games...")
            successes = 0
            for _ in range(games_per_config):
                success, path, opt_path = play_greedy_game(G, embeddings)
                if success:
                    successes += 1
                log_game_result(log_file, "greedy", "gemini", alg, k, n, temp, success, path, opt_path)
                
            print(f"Success rate: {successes / games_per_config:.2%}")

if __name__ == "__main__":
    main()
