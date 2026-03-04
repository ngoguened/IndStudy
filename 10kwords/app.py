import streamlit as st
import networkx as nx
import os
import random

import json
from sqlalchemy import text

# App configuration
st.set_page_config(page_title="10kwords Challenge Mode", page_icon="🕸️", layout="centered")

# Paths & Graphs
DATA_DIR = "/home/nicholas/IdeaProjects/IndStudy/10kwords/data"
GRAPH_CONFIGS = [
    {"k": 20, "n": 5},
    {"k": 15, "n": 10},
    {"k": 10, "n": 15},
    {"k": 5, "n": 20},
]

# Database Setup
def init_db():
    try:
        conn = st.connection("postgresql", type="sql")
        with conn.session as s:
            s.execute(text("""
                CREATE TABLE IF NOT EXISTS game_results (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    k_val INT,
                    n_val INT,
                    start_word VARCHAR(255),
                    target_word VARCHAR(255),
                    steps_taken INT,
                    optimal_dist INT,
                    path_taken TEXT,
                    success BOOLEAN
                );
            """))
            s.commit()
        return conn
    except Exception as e:
        st.warning(f"Database connection not configured or failed: {e}")
        return None

def insert_game_result(conn, k_val, n_val, start_word, target_word, steps_taken, optimal_dist, path_taken, success):
    if conn is None:
        return
    try:
        with conn.session as s:
            s.execute(text("""
                INSERT INTO game_results (k_val, n_val, start_word, target_word, steps_taken, optimal_dist, path_taken, success)
                VALUES (:k, :n, :start, :target, :steps, :opt, :path, :success)
            """), {
                "k": k_val, "n": n_val, "start": start_word, "target": target_word, 
                "steps": steps_taken, "opt": optimal_dist, "path": json.dumps(path_taken), "success": success
            })
            s.commit()
    except Exception as e:
        st.error(f"Failed to log game result to database: {e}")

@st.cache_resource
def load_graph(k, n):
    """Loads and caches the NetworkX graph from file based on k and n."""
    graph_path = os.path.join(DATA_DIR, f"graph_gemini_inv_knn+n_probabilistic_k{k}_n{n}.gexf")
    if not os.path.exists(graph_path):
        st.error(f"Error: {graph_path} not found.")
        return None
    try:
        G = nx.read_gexf(graph_path)
        return G
    except Exception as e:
        st.error(f"Error reading GEXF: {e}")
        return None

def get_neighbors(G, current_word):
    """Gets sorted neighbors for the current word."""
    if current_word not in G: 
        return []
    neighbors = []
    for nbr in G.neighbors(current_word):
        edge_data = G.get_edge_data(current_word, nbr)
        weight = float(edge_data.get('weight', 0.0))
        neighbors.append((nbr, weight))
    # Sort by weight descending
    neighbors.sort(key=lambda x: x[1], reverse=True)
    return neighbors

def generate_challenge(G):
    """Finds a start and target word pair with an optimal path of at least 3."""
    nodes = list(G.nodes())
    while True:
        start_word, target_word = random.choice(nodes), random.choice(nodes)
        if start_word != target_word:
            try:
                optimal_path = nx.shortest_path(G, start_word, target_word)
                optimal_dist = len(optimal_path) - 1
                if optimal_dist > 2:
                    return start_word, target_word, optimal_path, optimal_dist
            except nx.NetworkXNoPath:
                continue

def initialize_game(G):
    """Initializes a new game in session state."""
    start_word, target_word, optimal_path, optimal_dist = generate_challenge(G)
    st.session_state.start_word = start_word
    st.session_state.target_word = target_word
    st.session_state.current_word = start_word
    st.session_state.path = [start_word]
    st.session_state.optimal_path = optimal_path
    st.session_state.optimal_dist = optimal_dist
    st.session_state.game_over = False
    st.session_state.success = False
    st.session_state.db_logged = False

def restart_game():
    """Callback to reset the game state."""
    for key in ['start_word', 'target_word', 'current_word', 'path', 'optimal_path', 'optimal_dist', 'game_over', 'success', 'graph_k', 'graph_n', 'db_logged']:
        if key in st.session_state:
            del st.session_state[key]

# Initialize graph configuration if first run or reset
if 'graph_k' not in st.session_state:
    config = random.choice(GRAPH_CONFIGS)
    st.session_state.graph_k = config['k']
    st.session_state.graph_n = config['n']
    st.session_state.db_logged = False

# Load graph
G = load_graph(st.session_state.graph_k, st.session_state.graph_n)

if G is None:
    st.stop()

# Initialize session state if first run or reset
if 'start_word' not in st.session_state:
    initialize_game(G)

conn = init_db()

st.title("10kwords Challenge Mode")

# Determine game status
steps_taken = len(st.session_state.path) - 1
max_steps = 20

if not st.session_state.game_over:
    if st.session_state.current_word == st.session_state.target_word:
        st.session_state.game_over = True
        st.session_state.success = True
    elif steps_taken >= max_steps:
        st.session_state.game_over = True
        st.session_state.success = False

# Database Logging
if st.session_state.game_over and not st.session_state.db_logged:
    insert_game_result(
        conn, 
        st.session_state.graph_k, 
        st.session_state.graph_n, 
        st.session_state.start_word, 
        st.session_state.target_word, 
        steps_taken, 
        st.session_state.optimal_dist, 
        st.session_state.path, 
        st.session_state.success
    )
    st.session_state.db_logged = True

# UI: Status Information
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Target Word", value=st.session_state.target_word.upper())
with col2:
    st.metric(label="Steps", value=f"{steps_taken} / {max_steps}", delta=f"Optimal: {st.session_state.optimal_dist}", delta_color="off")

st.markdown(f"**Current Location:**  `{st.session_state.current_word.upper()}`")

# Display current path
with st.expander("Show current path"):
    st.write(" -> ".join(st.session_state.path))

# Game over screens
if st.session_state.game_over:
    if st.session_state.success:
        st.success(f"🎉 **VICTORY!** You reached **{st.session_state.target_word.upper()}** in {steps_taken} steps!")
    else:
        st.error(f"💀 **GAME OVER.** You exceeded the maximum of {max_steps} steps.")
    
    st.info(f"**Optimal Path ({st.session_state.optimal_dist} steps):**\n" + " -> ".join(st.session_state.optimal_path))
    st.button("Play Again", on_click=restart_game, type="primary")

# Ongoing game mechanics
else:
    neighbors = get_neighbors(G, st.session_state.current_word)
    
    if not neighbors:
        st.warning("Dead End! You can't move anywhere from here.")
    else:
        options = [nbr for nbr, weight in neighbors]
        
        # Prepare display labels
        display_options = {}
        for nbr, weight in neighbors:
            display_options[nbr] = f"{nbr} (Sim: {weight*100:.1f}%)"
            
        def on_move(selected):
            st.session_state.current_word = selected
            st.session_state.path.append(selected)
                
        st.markdown("**Choose your next word:**")
        
        # Display as a grid of buttons (3 columns)
        cols = st.columns(3)
        for i, option in enumerate(options):
            with cols[i % 3]:
                st.button(
                    display_options[option], 
                    key=f"btn_{option}_{steps_taken}_{i}", 
                    on_click=on_move, 
                    args=(option,),
                    use_container_width=True
                )
