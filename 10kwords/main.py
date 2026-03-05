import argparse
import os
import sys
import csv

# Import modules
try:
    from embeddings import generate_embeddings, save_embeddings, load_embeddings, get_words_and_vectors
    from graph import generate_graph, save_graph, load_graph, analyze_graph
    from game import launch_cli_explorer, play_greedy_game
except ImportError as e:
    print(f"Error importing local modules: {e}")
    sys.exit(1)

def log_game_result(args, player_type, success, path, opt_path):
    path_len = len(path) - 1 if success else -1
    opt_len = len(opt_path) - 1
    diff = path_len - opt_len if success else -1
    path_str = " -> ".join(path)
    start_w = path[0]
    target_w = opt_path[-1]
    
    log_file = os.path.join(args.data_dir, "results.csv")
    file_exists = os.path.exists(log_file)
    with open(log_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["player", "embedding", "algorithm", "k_nn", "n_long", "temp", "start", "target", "success", "pth_len", "opt_len", "diff", "path_str"])
        writer.writerow([player_type, args.embedding, args.graph, args.k, args.n, args.alpha, start_w, target_w, success, path_len, opt_len, diff, path_str])
    print(f"\\n[Logged result to {log_file}]")

def main():
    parser = argparse.ArgumentParser(description="10kwords Semantic Graph Generator & Explorer")
    
    # Embedding specs
    parser.add_argument("--embedding", type=str, choices=['gemini', 'glove'], default='gemini',
                        help="Type of embeddings to use.")
    
    # Graph specs
    parser.add_argument("--graph", type=str, 
                        choices=['relative_neighborhood', 'k_nn+n_random', 'inv_knn+n_probabilistic'], 
                        default='k_nn+n_random',
                        help="Graph generation algorithm.")
    parser.add_argument("-k", type=int, default=15, help="Number of nearest neighbors to connect.")
    parser.add_argument("-n", type=int, default=10, help="Number of long range edges (random or probabilistic).")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha parameter for probabilistic edges. 1.0 is default, >1 is sharper, <1 is more uniform.")
    
    # Functional specs
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to save/load data.")
    
    args = parser.parse_args()
    # Paths mapped cleanly
    os.makedirs(args.data_dir, exist_ok=True)
    embed_file = os.path.join(args.data_dir, f"embeddings_{args.embedding}.pkl")
    alpha_str = f"_alpha{args.alpha}" if args.alpha != 1.0 else ""
    graph_file = os.path.join(args.data_dir, f"graph_{args.embedding}_{args.graph}_k{args.k}_n{args.n}{alpha_str}.gexf")

    print(f"\\n=== SMALL WORLD PIPELINE: {args.embedding.upper()} -> {args.graph.upper()} ===")

    # 1. Embeddings
    if not os.path.exists(embed_file):
        print(f"\\n[1/3] Generating {args.embedding} embeddings...")
        embeddings = generate_embeddings(embedding_type=args.embedding)
        save_embeddings(embeddings, embed_file)
    else:
        print(f"\\n[1/3] Loading existing embeddings from {embed_file}...")
        embeddings = load_embeddings(embed_file)

    # 2. Graph
    if not os.path.exists(graph_file):
        print(f"\\n[2/3] Generating graph using {args.graph} algorithm...")
        words, vectors = get_words_and_vectors(embeddings)
        G = generate_graph(words, vectors, args.k, args.n, algorithm=args.graph, alpha=args.alpha)
        analyze_graph(G)
        save_graph(G, graph_file)
    else:
        print(f"\\n[2/3] Loading existing graph from {graph_file}...")
        G = load_graph(graph_file)

    # 3. Explore
    print(f"\\n[3/3] Ready for Exploration!")
    while True:
        print("\\n=== MENU ===")
        print("1. Human Challenge Mode (CLI)")
        print("2. Greedy Search Demo")
        print("3. Quit")
        choice = input("> ").strip()
        
        if choice == '1':
            res = launch_cli_explorer(G)
            if res and res[0] is not None:
                success, path, opt_path = res
                log_game_result(args, "human", success, path, opt_path)
        elif choice == '2':
            success, path, opt_path = play_greedy_game(G, embeddings)
            print(f"\\nGreedy Algorithm Success: {success}")
            print(f"Greedy Path: {' -> '.join(path)}")
            print(f"Optimal Path: {' -> '.join(opt_path)}")
            log_game_result(args, "greedy", success, path, opt_path)
        elif choice == '3':
            break

if __name__ == "__main__":
    main()
