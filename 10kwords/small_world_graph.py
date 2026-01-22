import networkx as nx
import os
import sys
import random

def load_graph(graph_file):
    if not os.path.exists(graph_file):
        print(f"Error: {graph_file} not found.")
        sys.exit(1)
        
    print(f"Loading {graph_file}... ", end="", flush=True)
    try:
        G = nx.read_gexf(graph_file)
        print("Done.")
        return G
    except Exception as e:
        print(f"\nError reading GEXF: {e}")
        sys.exit(1)

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_neighbors_from_graph(G, current_word):
    if current_word not in G: return []
    neighbors = []
    for nbr in G.neighbors(current_word):
        edge_data = G.get_edge_data(current_word, nbr)
        weight = float(edge_data.get('weight', 0.0))
        neighbors.append((nbr, weight))
    neighbors.sort(key=lambda x: x[1], reverse=True)
    return neighbors

def explore_mode(G):
    nodes = list(G.nodes())
    current_word = random.choice(nodes)
    while True:
        clear_screen()
        print("=== STATIC GRAPH EXPLORER ===")
        print(f"\nLOCATION:   [{current_word.upper()}]")
        
        neighbors = get_neighbors_from_graph(G, current_word)
        options = []
        for i, (nbr, weight) in enumerate(neighbors):
            percent = weight * 100
            print(f"  {i+1}. {nbr:<15} (Sim: {percent:.1f}%)")
            options.append(nbr)
            
        if not options: print("  (Dead End!)")

        print("\n[number] move, [word] jump, 'back' menu")
        cmd = input("> ").strip().lower()
        if cmd == 'back': break
        
        if cmd.isdigit():
            idx = int(cmd) - 1
            if 0 <= idx < len(options): current_word = options[idx]
        elif cmd in G: current_word = cmd
        elif cmd: input(f"Word '{cmd}' not found...")

def challenge_mode(G):
    nodes = list(G.nodes())
    while True:
        start_word, target_word = random.choice(nodes), random.choice(nodes)
        if start_word != target_word:
            try:
                optimal_path = nx.shortest_path(G, start_word, target_word)
                optimal_dist = len(optimal_path) - 1
                if optimal_dist > 2: break
            except nx.NetworkXNoPath: continue
    
    current_word = start_word
    path = [start_word]
    
    while True:
        if current_word == target_word:
            print(f"\n\n*** VICTORY! ***\nPath: {' -> '.join(path)}")
            print(f"Optimal ({optimal_dist}): {' -> '.join(optimal_path)}")
            input("Press Enter...")
            break

        clear_screen()
        print(f"MISSION: Navigate to -> '{target_word.upper()}'")
        print(f"STEPS:   {len(path)-1}  (Optimal: {optimal_dist})")
        print(f"LOCATION:   [{current_word.upper()}]")
        
        neighbors = get_neighbors_from_graph(G, current_word)
        options = []
        for i, (nbr, weight) in enumerate(neighbors):
            percent = weight * 100
            print(f"  {i+1}. {nbr:<15} (Sim: {percent:.1f}%)")
            options.append(nbr)
            
        cmd = input("\n[number] move, 'back' give up > ").strip().lower()
        if cmd == 'back': 
            print(f"\nGAME OVER. Optimal path:\n{' -> '.join(optimal_path)}")
            input("Press Enter...")
            break
        
        if cmd.isdigit():
            idx = int(cmd) - 1
            if 0 <= idx < len(options):
                current_word = options[idx]
                path.append(current_word)

def launch_explorer(graph_file_path):
    """
    Loads the graph from graph_file_path and starts the interactive menu.
    """
    G = load_graph(graph_file_path)
    while True:
        print("\n=== MENU ===")
        print("1. Free Roam")
        print("2. Challenge Mode")
        print("3. Quit")
        choice = input("> ").strip()
        if choice == '3': break
        elif choice == '1': explore_mode(G)
        elif choice == '2': challenge_mode(G)

if __name__ == "__main__":
    launch_explorer("data/small_world.gexf")