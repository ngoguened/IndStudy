import os
import sys

# Import the functional modules
try:
    import get_embeddings
    import generate_graph
    import small_world_graph
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# --- Configuration Center ---
DATA_DIR = "data"
EMBEDDING_FILE = os.path.join(DATA_DIR, "embeddings.pkl")
GRAPH_FILE = os.path.join(DATA_DIR, "small_world.gexf")

def main():
    print("=== SMALL WORLD PIPELINE ===\n")

    # 1. Check/Run Embeddings
    if not os.path.exists(EMBEDDING_FILE):
        print(f"[1/3] Missing {EMBEDDING_FILE}. Generating...")
        # Call the specific function from the module
        get_embeddings.generate_embeddings_file(EMBEDDING_FILE)
    else:
        print(f"[1/3] Found existing embeddings.")

    # 2. Check/Run Graph Generation
    if not os.path.exists(GRAPH_FILE):
        print(f"[2/3] Missing {GRAPH_FILE}.")
        
        if not os.path.exists(EMBEDDING_FILE):
            print("Embeddings missing. Cannot generate graph.")
            sys.exit(1)
            
        # Pass input and output paths explicitly
        print("what do you want K to be")
        k = int(input())
        print("what do you want N to be")
        n = int(input())
        print("what algorithm do you want? (random or probabilistic)")
        algorithm = input()
        generate_graph.create_graph_from_embeddings(EMBEDDING_FILE, GRAPH_FILE, k, n, algorithm)
    else:
        print(f"[2/3] Found existing graph.")

    # 3. Launch Explorer
    print(f"[3/3] Launching Game...")
    try:
        small_world_graph.launch_explorer(GRAPH_FILE)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()