import streamlit as st
import networkx as nx
import os
import random

import json
from sqlalchemy import text
from st_keyup import st_keyup
import streamlit.components.v1 as components
import pickle
import numpy as np

# App configuration
st.set_page_config(page_title="10kwords Challenge Mode", page_icon="🕸️", layout="centered")

# Custom CSS for bigger buttons
st.markdown("""
<style>
div[data-testid="stButton"] button {
    min-height: 60px;
    font-size: 18px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Paths & Graphs
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GRAPH_CONFIGS = [
    {"k": 20, "n": 5},
    {"k": 15, "n": 10},
    {"k": 10, "n": 15},
    {"k": 5, "n": 20},
]

# Database Setup
def init_db():
    try:
        conn = st.connection("postgresql", type="sql", autocommit=True)
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
        return conn
    except Exception as e:
        st.warning(f"Database connection not configured or failed: {e}")
        return None

def upsert_game_result(conn, db_id, k_val, n_val, start_word, target_word, steps_taken, optimal_dist, path_taken, success):
    if conn is None:
        return db_id
    try:
        with conn.session as s:
            if db_id is None:
                result = s.execute(text("""
                    INSERT INTO game_results (k_val, n_val, start_word, target_word, steps_taken, optimal_dist, path_taken, success)
                    VALUES (:k, :n, :start, :target, :steps, :opt, :path, :success)
                    RETURNING id
                """), {
                    "k": k_val, "n": n_val, "start": start_word, "target": target_word, 
                    "steps": steps_taken, "opt": optimal_dist, "path": json.dumps(path_taken), "success": success
                })
                return result.scalar()
            else:
                s.execute(text("""
                    UPDATE game_results 
                    SET steps_taken = :steps, path_taken = :path, success = :success
                    WHERE id = :id
                """), {
                    "steps": steps_taken, "path": json.dumps(path_taken), "success": success, "id": db_id
                })
                return db_id
    except Exception as e:
        st.error(f"Failed to log game result to database: {e}")
        return db_id

from supabase import create_client, Client

@st.cache_resource
def fetch_graph_data(k, n):
    """Loads and caches the NetworkX graph from Supabase Storage or local file."""
    file_name = f"graph_gemini_inv_knn+n_probabilistic_k{k}_n{n}.gexf"
    graph_path = os.path.join(DATA_DIR, file_name)
    
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # If the file isn't locally cached, download it from Supabase Storage
    if not os.path.exists(graph_path):
        try:
            # Initialize Supabase client
            url: str = st.secrets["SUPABASE_URL"]
            key: str = st.secrets["SUPABASE_KEY"]
            supabase: Client = create_client(url, key)
            
            with st.spinner(f"Downloading graph {k}-{n} from cloud..."):
                # Download file from the 'graphs' bucket
                res = supabase.storage.from_("graphs").download(file_name)
                
                # Save to local file
                with open(graph_path, 'wb') as f:
                    f.write(res)
        except Exception as e:
            st.error(f"Error downloading graph from Supabase: {e}\n\nMake sure your .streamlit/secrets.toml has SUPABASE_URL and SUPABASE_KEY configured, and that the 'graphs' bucket exists.")
            return None
            
    try:
        G = nx.read_gexf(graph_path)
        return G
    except Exception as e:
        st.error(f"Error reading GEXF: {e}")
        return None

@st.cache_resource
def fetch_embeddings_data():
    """Loads and caches word embeddings from Supabase Storage or local file."""
    file_name = "embeddings_gemini.pkl"
    embeddings_path = os.path.join(DATA_DIR, file_name)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Fast path: If the combined file exists locally, just load it.
    if os.path.exists(embeddings_path):
        try:
            with open(embeddings_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Error reading local embeddings: {e}")
            return None

    # If not, we download the 3 chunks from Supabase, merge them, and save.
    try:
        url: str = st.secrets["SUPABASE_URL"]
        key: str = st.secrets["SUPABASE_KEY"]
        supabase: Client = create_client(url, key)
    except Exception as e:
        st.error(f"Failed to initialize Supabase client: {e}")
        return None
        
    combined_embeddings = {}
    for i in range(1, 4):
        part_name = f"embeddings_gemini_part{i}.pkl"
        part_path = os.path.join(DATA_DIR, part_name)
        
        if not os.path.exists(part_path):
            try:
                with st.spinner(f"Downloading word embeddings part {i}/3 from cloud (approx 40MB)..."):
                    res = supabase.storage.from_("graphs").download(part_name)
                    with open(part_path, 'wb') as f:
                        f.write(res)
            except Exception as e:
                st.error(f"Error downloading {part_name} from Supabase: {e}")
                return None
                
        try:
            with open(part_path, 'rb') as f:
                part_data = pickle.load(f)
                combined_embeddings.update(part_data)
        except Exception as e:
            st.error(f"Error reading {part_name}: {e}")
            return None
            
    # Save the combined embeddings locally so future loads are fast
    try:
        with open(embeddings_path, 'wb') as f:
            pickle.dump(combined_embeddings, f)
    except Exception as e:
        pass # Not critical if we can't save the combined file, we can still return it
        
    return combined_embeddings

def get_cosine_sim(vec_a, vec_b):
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))

def compute_greedy_path(G, embeddings, start_word, target_word, max_steps=20):
    if not embeddings:
        return []
    
    path = [start_word]
    current_word = start_word
    visited = {start_word}
    
    target_vec = embeddings.get(target_word)
    if target_vec is None:
        return path
        
    for _ in range(max_steps):
        if current_word == target_word:
            break
            
        neighbors = []
        if current_word in G:
            neighbors = [n for n in G.neighbors(current_word) if n not in visited]
            
        if not neighbors:
            break
            
        best_neighbor = None
        best_score = -2.0
        
        for nbr in neighbors:
            nbr_vec = embeddings.get(nbr)
            if nbr_vec is not None:
                score = get_cosine_sim(nbr_vec, target_vec)
                if score > best_score:
                    best_score = score
                    best_neighbor = nbr
                    
        if best_neighbor is None:
            break
            
        current_word = best_neighbor
        path.append(current_word)
        visited.add(current_word)
        
    return path

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
    st.session_state.db_id = None
    st.session_state.last_logged_step = 0
    st.session_state.greedy_path = None

def restart_game():
    """Callback to reset the game state."""
    for key in ['start_word', 'target_word', 'current_word', 'path', 'optimal_path', 'optimal_dist', 'greedy_path', 'game_over', 'success', 'graph_k', 'graph_n', 'db_logged', 'db_id', 'last_logged_step']:
        if key in st.session_state:
            del st.session_state[key]

# Initialize graph configuration if first run or reset
if 'graph_k' not in st.session_state:
    config = random.choice(GRAPH_CONFIGS)
    st.session_state.graph_k = config['k']
    st.session_state.graph_n = config['n']
    st.session_state.db_logged = False

# Load graph
G = fetch_graph_data(st.session_state.graph_k, st.session_state.graph_n)
embeddings = fetch_embeddings_data()

if G is None or embeddings is None:
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
should_log = False
success_val = None

if st.session_state.game_over and not st.session_state.db_logged:
    should_log = True
    success_val = st.session_state.success
    st.session_state.db_logged = True
elif not st.session_state.game_over and steps_taken > 2 and st.session_state.get('last_logged_step') != steps_taken:
    should_log = True
    success_val = None
    st.session_state.last_logged_step = steps_taken

if should_log:
    st.session_state.db_id = upsert_game_result(
        conn,
        st.session_state.get('db_id'),
        st.session_state.graph_k, 
        st.session_state.graph_n, 
        st.session_state.start_word, 
        st.session_state.target_word, 
        steps_taken, 
        st.session_state.optimal_dist, 
        st.session_state.path, 
        success_val
    )

# UI: Status Information
col1, col2, col3 = st.columns(3)
with col1:
    if steps_taken == 0:
        st.metric(label="Start Word", value=st.session_state.start_word.upper())
    else:
        st.metric(label="Current Word", value=st.session_state.current_word.upper())
with col2:
    st.metric(label="Target Word", value=st.session_state.target_word.upper())
with col3:
    st.metric(label="Steps", value=f"{steps_taken} / {max_steps}")

# Display current path
with st.expander("Show current path"):
    st.write(" -> ".join(st.session_state.path))

# Game over screens
if st.session_state.game_over:
    if st.session_state.greedy_path is None:
        st.session_state.greedy_path = compute_greedy_path(G, embeddings, st.session_state.start_word, st.session_state.target_word, max_steps=max_steps)

    if st.session_state.success:
        st.success(f"**VICTORY!** You reached **{st.session_state.target_word.upper()}** in {steps_taken} steps!")
    else:
        st.error(f"**GAME OVER.** You exceeded the maximum of {max_steps} steps.")
    
    greedy_len = len(st.session_state.greedy_path) - 1
    st.info(f"**Greedy Path ({greedy_len} steps):**\n" + " -> ".join(st.session_state.greedy_path))
    st.button("Play Again", on_click=restart_game, type="primary")

# Ongoing game mechanics
else:
    neighbors = get_neighbors(G, st.session_state.current_word)
    
    if not neighbors:
        st.warning("Dead End! You can't move anywhere from here.")
    else:
        options = [nbr for nbr, weight in neighbors]
        
        if "filter_key_counter" not in st.session_state:
            st.session_state.filter_key_counter = 0
            
        def on_move(selected):
            st.session_state.current_word = selected
            st.session_state.path.append(selected)
            st.session_state.filter_key_counter += 1
                
        st.markdown("**Choose your next word:**")
        filter_text = st_keyup("Filter available words...", key=f"word_filter_{st.session_state.filter_key_counter}")
        
        # Autofocus hack
        components.html(
            f"""
            <script>
                // We need a slight delay to ensure the component is fully rendered
                setTimeout(function() {{
                    var iframe = window.parent.document.querySelector('iframe[title="st_keyup.st_keyup"]');
                    if (iframe && iframe.contentDocument) {{
                        var input = iframe.contentDocument.querySelector('input');
                        if (input) {{
                            input.focus();
                        }}
                    }}
                }}, 100);
            </script>
            """,
            height=0,
            width=0,
        )
        
        # In case it's None initially to avoid errors
        if filter_text is None:
            filter_text = ""
        else:
            filter_text = filter_text.strip().lower()
        
        filtered_options = [opt for opt in options if filter_text in opt.lower()]
        
        if not filtered_options:
            st.write("No words matching filter.")
        else:
            # Display as a grid of buttons (3 columns)
            cols = st.columns(3)
            for i, option in enumerate(filtered_options):
                with cols[i % 3]:
                    st.button(
                        option, 
                        key=f"btn_{option}_{steps_taken}_{i}", 
                        on_click=on_move, 
                        args=(option,),
                        use_container_width=True
                    )
