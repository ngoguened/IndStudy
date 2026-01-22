import os
import sys

try:
    import get_embeddings
    import generate_graph
    import small_world_graph
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

DATA_DIR = "data"
EMBEDDING_FILE = os.path.join(DATA_DIR, "embeddings.pth")
GRAPH_FILE = os.path.join(DATA_DIR, "small_world.gexf")

def main():
    print("=== SMALL WORLD PIPELINE ===\n")

    # Run Embeddings
    if not os.path.exists(EMBEDDING_FILE):
        print(f"[1/3] Missing {EMBEDDING_FILE}. Generating...")
        # Call the specific function from the module
        get_embeddings.generate_embeddings_file(EMBEDDING_FILE)
    else:
        print(f"[1/3] Found existing embeddings.")

    # Run Graph Generation
    if not os.path.exists(GRAPH_FILE):
        print(f"[2/3] Missing {GRAPH_FILE}. Building graph...")
        
        if not os.path.exists(EMBEDDING_FILE):
            print("Embeddings missing. Cannot generate graph.")
            sys.exit(1)

        generate_graph.create_graph_from_embeddings(EMBEDDING_FILE, GRAPH_FILE)
    else:
        print(f"[2/3] Found existing graph.")

    print(f"[3/3] Launching Game...")
    try:
        small_world_graph.launch_explorer(GRAPH_FILE)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()