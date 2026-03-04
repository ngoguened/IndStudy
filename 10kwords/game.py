import networkx as nx
import random
from sklearn.metrics.pairwise import cosine_similarity

class GameInstance:
    def __init__(self, G, max_steps=20):
        self.G = G
        self.max_steps = max_steps
        self.nodes = list(G.nodes())
        
        # State variables
        self.start_word = None
        self.target_word = None
        self.current_word = None
        self.optimal_path = []
        self.optimal_dist = 0
        self.path = []
        self.is_over = False
        self.is_success = False
        
    def start_new_game(self):
        """Finds a start and target word pair with an optimal path of at least 3."""
        while True:
            self.start_word, self.target_word = random.choice(self.nodes), random.choice(self.nodes)
            if self.start_word != self.target_word:
                try:
                    self.optimal_path = nx.shortest_path(self.G, self.start_word, self.target_word)
                    self.optimal_dist = len(self.optimal_path) - 1
                    if self.optimal_dist > 2:
                        break
                except nx.NetworkXNoPath:
                    continue
                    
        self.current_word = self.start_word
        self.path = [self.start_word]
        self.is_over = False
        self.is_success = False

    def get_moves(self):
        """Returns a list of sorted neighbors and their weights (similarity)."""
        if self.current_word not in self.G:
            return []
            
        neighbors = []
        for nbr in self.G.neighbors(self.current_word):
            edge_data = self.G.get_edge_data(self.current_word, nbr)
            weight = float(edge_data.get('weight', 0.0))
            neighbors.append((nbr, weight))
            
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors

    def make_move(self, neighbor_word):
        """Attempts to move to a neighbor. Updates game state."""
        if self.is_over:
            return False
            
        if self.G.has_edge(self.current_word, neighbor_word):
            self.current_word = neighbor_word
            self.path.append(self.current_word)
            self._check_game_over()
            return True
        return False
        
    def _check_game_over(self):
        steps_taken = len(self.path) - 1
        
        if self.current_word == self.target_word:
            self.is_over = True
            self.is_success = True
        elif steps_taken >= self.max_steps:
            self.is_over = True
            self.is_success = False

def get_cosine_sim(vec_a, vec_b):
    va = vec_a.reshape(1, -1)
    vb = vec_b.reshape(1, -1)
    return cosine_similarity(va, vb)[0][0]

def play_greedy_game(G, embeddings, max_steps=20):
    """Plays a single game using the greedy search algorithm."""
    game = GameInstance(G, max_steps)
    game.start_new_game()
    visited = {game.start_word}
    
    target_vec = embeddings[game.target_word]

    while not game.is_over:
        neighbors = [n[0] for n in game.get_moves()]
        valid_neighbors = [n for n in neighbors if n not in visited]
        
        if not valid_neighbors:
            break

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
            break
            
        game.make_move(best_neighbor)
        visited.add(best_neighbor)

    return game.is_success, game.path, game.optimal_path

def launch_cli_explorer(G):
    """A wrapper for the interactive terminal human exploration mode."""
    import os
    from prompt_toolkit import prompt
    from prompt_toolkit.completion import WordCompleter
    
    def clear_screen():
        os.system('cls' if os.name == 'nt' else 'clear')
        
    game = GameInstance(G)
    game.start_new_game()
    
    while True:
        if game.is_over:
            if game.is_success:
                print(f"\\n*** VICTORY! ***\\nPath: {' -> '.join(game.path)}")
            else:
                print(f"\\nGAME OVER. Too many steps.")
            print(f"Optimal ({game.optimal_dist}): {' -> '.join(game.optimal_path)}")
            input("Press Enter to continue...")
            return game.is_success, game.path, game.optimal_path

        clear_screen()
        steps_taken = len(game.path) - 1
        print(f"MISSION: Navigate to -> '{game.target_word.upper()}'")
        print(f"STEPS:   {steps_taken}  (Optimal: {game.optimal_dist})")
        print(f"LOCATION:   [{game.current_word.upper()}]")
        
        moves = game.get_moves()
        options = []
        neighbor_names = []
        
        for i, (nbr, weight) in enumerate(moves):
            percent = weight * 100
            print(f"  {i+1}. {nbr:<15} (Sim: {percent:.1f}%)")
            options.append(nbr)
            neighbor_names.append(nbr)
            
        search_completer = WordCompleter(neighbor_names, ignore_case=True, match_middle=True)
        
        try:
            cmd = prompt("> ", completer=search_completer).strip().lower()
        except KeyboardInterrupt:
            cmd = "back"
            
        if cmd == 'back':
            print(f"\\nGAME ABORTED. Optimal path:\\n{' -> '.join(game.optimal_path)}")
            input("Press Enter...")
            return None, None, None
            
        word_match = next((word for word in options if word.lower() == cmd), None)
        if word_match:
            game.make_move(word_match)
        elif cmd.isdigit():
            idx = int(cmd) - 1
            if 0 <= idx < len(options):
                game.make_move(options[idx])

