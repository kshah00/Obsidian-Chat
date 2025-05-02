import streamlit as st
import logging
from pathlib import Path
import os
import numpy as np
import urllib.parse

# Assuming these modules exist relative to the app's location or are installed
# Adjust imports based on your project structure if needed
from obsidian_ai_search.indexer import Indexer
from obsidian_ai_search.embedding import get_embedding_model, generate_embeddings
from obsidian_ai_search.vector_store import FaissVectorStore
from obsidian_ai_search.utils import get_query_suggestions, get_note_title
from obsidian_ai_search.rag import generate_answer_with_gemini
from obsidian_ai_search.config import APP_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Streamlit App State ---
# Use Streamlit's session state to store resources
if "model" not in st.session_state:
    st.session_state.model = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "index_dir" not in st.session_state:
    st.session_state.index_dir = None
if "resources_loaded" not in st.session_state:
    st.session_state.resources_loaded = False
if "last_loaded_index_dir" not in st.session_state:
    st.session_state.last_loaded_index_dir = None

# --- Helper Functions (Adapted from CLI) ---

def load_resources(index_dir: str):
    """Loads or reloads model and vector store, updating Streamlit state."""
    # Check if already loaded for this directory
    if st.session_state.resources_loaded and st.session_state.last_loaded_index_dir == index_dir:
        return True

    st.session_state.resources_loaded = False # Mark as loading
    st.session_state.model = None
    st.session_state.vector_store = None
    st.session_state.last_loaded_index_dir = index_dir # Store attempted load dir

    with st.spinner("Loading embedding model..."):
        try:
            model = get_embedding_model()
            if not model:
                st.error("Failed to load embedding model.")
                return False
            st.session_state.model = model
        except Exception as e:
            st.error(f"Error loading embedding model: {e}")
            logger.error(f"Failed to load model: {e}", exc_info=True)
            return False

    with st.spinner(f"Loading index from {index_dir}..."):
        vector_store = FaissVectorStore(index_dir=index_dir)
        try:
            vector_store.load()
            st.session_state.vector_store = vector_store
            # Dimension check moved to search to avoid blocking app load if mismatch
            st.success(f"Index loaded ({vector_store.count} vectors).")
            st.session_state.resources_loaded = True # Mark as loaded successfully
            return True
        except FileNotFoundError:
             st.warning(f"Index file not found in {index_dir}. Please run the indexing command.")
             st.session_state.vector_store = vector_store # Keep the object even if not loaded
             st.session_state.resources_loaded = False # Not fully loaded
             return False # Indicate index is not ready
        except Exception as e:
            st.error(f"Error loading index: {e}")
            logger.error(f"Failed to load index from {index_dir}: {e}", exc_info=True)
            st.session_state.vector_store = None # Ensure it's None on error
            st.session_state.resources_loaded = False
            return False

# --- Streamlit UI ---

st.set_page_config(layout="wide")
st.title("🧠 Obsidian AI Search")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")
    
    # Index Directory Selection
    default_index_dir = APP_CONFIG.get('index_dir', './.obsidian_ai_search_index')
    index_dir_input = st.text_input(
        "Index Directory",
        value=st.session_state.get("index_dir", default_index_dir),
        help="Path to the directory containing the FAISS index and metadata."
    )
    st.session_state.index_dir = index_dir_input # Update state immediately

    # Attempt to load resources if directory changed or not loaded
    if st.session_state.index_dir and (st.session_state.index_dir != st.session_state.last_loaded_index_dir or not st.session_state.resources_loaded):
         if Path(st.session_state.index_dir).exists():
             load_resources(st.session_state.index_dir)
         else:
             st.warning(f"Index directory not found: {st.session_state.index_dir}")
             st.session_state.resources_loaded = False
             st.session_state.model = None
             st.session_state.vector_store = None
    elif not st.session_state.index_dir:
        st.warning("Please specify an index directory.")
        st.session_state.resources_loaded = False
        st.session_state.model = None
        st.session_state.vector_store = None

    # Display Index Status in Sidebar
    if st.session_state.get("resources_loaded") and st.session_state.vector_store:
        vs = st.session_state.vector_store
        if vs.is_initialized and vs.count > 0:
            file_count = len(vs.file_path_to_chunk_ids)
            st.success(f"Index Ready: {vs.count} chunks, {file_count} notes.")
        elif vs.is_initialized:
            st.warning("Index is empty.")
        else:
            st.warning("Index exists but not loaded/initialized correctly.")
    elif st.session_state.get("last_loaded_index_dir"): # Show warning if load was attempted but failed
         if Path(st.session_state.last_loaded_index_dir).exists():
             st.warning("Index not loaded. Check errors above.")
         # else: warning about dir not found is shown above


    st.divider()
    
    # Search Options
    st.header("Search Options")
    top_k = st.slider(
        "Number of Results (K)",
        min_value=1,
        max_value=50,
        value=st.session_state.get("top_k", APP_CONFIG.get('default_top_k', 5)),
        key="top_k",
        help="Number of initial results to retrieve."
    )
    
    use_gemini = st.toggle(
        "Enable Gemini RAG",
        value=st.session_state.get("use_gemini", False),
        key="use_gemini",
        help="Use Google Gemini to generate an answer based on search results (requires GEMINI_API_KEY)."
    )
    
    if use_gemini and not os.environ.get("GEMINI_API_KEY"):
        st.warning("GEMINI_API_KEY environment variable not set. RAG will be disabled.")
        st.session_state.use_gemini = False # Force disable if key missing
        st.rerun() # Rerun to update the toggle state visually

# --- Main Search Area ---
st.header("Search Your Notes")

query = st.text_input("Enter your search query:", key="query_input")

search_button = st.button("Search")

# --- Search Execution and Display ---
if search_button and query:
    if not st.session_state.get("resources_loaded") or not st.session_state.model or not st.session_state.vector_store:
        st.error("Resources (model/index) not loaded. Cannot search. Check configuration in the sidebar.")
    elif not st.session_state.vector_store.is_initialized:
         st.error(f"Index not initialized in {st.session_state.index_dir}. Please run the indexing command.")
    elif st.session_state.vector_store.count == 0:
         st.warning("Index is empty. Cannot search.")
    elif st.session_state.model.get_sentence_embedding_dimension() != st.session_state.vector_store.dimension:
         st.error(f"Index/model dimension mismatch ({st.session_state.vector_store.dimension} vs {st.session_state.model.get_sentence_embedding_dimension()}). Please re-index.")
    else:
        # All checks passed, proceed with search
        model = st.session_state.model
        vector_store = st.session_state.vector_store
        current_top_k = st.session_state.top_k
        use_rag = st.session_state.use_gemini

        try:
            # --- Embed Query ---
            with st.spinner("Embedding query..."):
                query_chunk = [{'content': query}]
                query_with_embedding = generate_embeddings(query_chunk, model)
                if not query_with_embedding or 'embedding' not in query_with_embedding[0]:
                    st.error("Failed to generate query embedding.")
                    st.stop() # Stop execution for this run
                query_vector = query_with_embedding[0]['embedding']

            # --- Initial Search (Semantic + Keyword Fallback) ---
            gemini_answer = None
            initial_semantic_results = []
            initial_keyword_results = []
            
            with st.spinner(f"Performing semantic search (k={current_top_k})..."):
                 initial_semantic_results = vector_store.search(query_vector, k=current_top_k)

            search_results = initial_semantic_results
            search_performed = "Semantic"

            if len(initial_semantic_results) < 3:
                 st.info("Semantic search returned < 3 results. Checking keyword fallback...")
                 keyword_limit = max(current_top_k, 5) # Ensure keyword check is decent
                 with st.spinner(f"Performing keyword search (limit={keyword_limit})..."):
                     initial_keyword_results = vector_store.keyword_search(query, limit=keyword_limit)
                 if initial_keyword_results:
                     # Simple merge for now: prioritize semantic, then add distinct keyword results
                     # A more sophisticated merge could be implemented
                     semantic_ids = {res.get('chunk_id') for res in initial_semantic_results}
                     combined = list(initial_semantic_results) # Start with semantic
                     for kw_res in initial_keyword_results:
                         if kw_res.get('chunk_id') not in semantic_ids:
                             combined.append(kw_res)
                     
                     # If keyword found *anything* and semantic found *nothing*, use keyword fully
                     if not initial_semantic_results and initial_keyword_results:
                         search_results = initial_keyword_results
                         search_performed = "Keyword"
                         st.info("Using keyword results as primary.")
                     # If both found something, but semantic was < 3, use combined
                     elif initial_semantic_results and initial_keyword_results:
                          search_results = combined[:current_top_k] # Limit combined results
                          search_performed = "Semantic + Keyword"
                          st.info("Combining semantic and keyword results.")
                     # Else (semantic < 3, keyword 0), stick with semantic results & type
                     
            # --- Optional Gemini RAG ---
            if use_rag and search_results:
                with st.spinner("Preparing context and calling Gemini API..."):
                    context_chunks = [res.get('content', '') for res in search_results if res.get('content')]
                    if not context_chunks:
                        st.warning("Skipping RAG: Initial results found, but none contained text content.")
                    else:
                        api_key = os.environ.get("GEMINI_API_KEY")
                        # We already checked for API key existence in the sidebar if toggle is on
                        if api_key:
                             try:
                                 gemini_answer = generate_answer_with_gemini(query, context_chunks, api_key)
                                 if gemini_answer and not gemini_answer.startswith("Error:"):
                                     st.success("Received Gemini answer.")
                                 elif gemini_answer:
                                     st.warning(f"Gemini RAG issue: {gemini_answer}")
                                 else:
                                      st.warning("Gemini RAG returned no answer (potentially empty context passed).")
                             except Exception as rag_err:
                                 st.error(f"Error during RAG: {rag_err}")
                                 gemini_answer = f"Error during RAG: {rag_err}" # Display error in RAG box
                        # No need for else here, covered by sidebar check / toggle state

            elif use_rag and not search_results:
                st.info("Skipping RAG (no results).")

            # --- Display Results ---
            st.divider()

            if gemini_answer and not gemini_answer.startswith("Error:"):
                st.subheader("🤖 Gemini Answer")
                st.markdown(gemini_answer)
                st.divider()
                st.subheader(f"Source Results (using {search_performed} search)")
            elif gemini_answer: # Display RAG error
                 st.subheader("🤖 Gemini RAG Error")
                 st.error(gemini_answer)
                 st.divider()
                 st.subheader(f"Search Results (using {search_performed} search)")
            else:
                 st.subheader(f"Search Results (using {search_performed} search)")


            if not search_results:
                st.warning(f"No results found for your query: '{query}'")
                # Add suggestions if possible
                try:
                    note_titles = vector_store.get_all_note_titles()
                    if note_titles:
                        suggestions = get_query_suggestions(query, note_titles)
                        if suggestions:
                            suggestions_str = ", ".join([f"'{s}'" for s in suggestions])
                            st.info(f"Did you mean: {suggestions_str}?")
                except Exception as suggestion_err:
                    logger.error(f"Error generating suggestions: {suggestion_err}", exc_info=True)

            else:
                # Display results using expanders
                for i, result in enumerate(search_results[:current_top_k], start=1): # Ensure we respect top_k display limit
                    file_path = result.get('file_path', 'N/A')
                    relative_path = result.get('relative_path') # Get relative path if available

                    try:
                        note_title = get_note_title(file_path)
                    except Exception:
                         note_title = Path(file_path).stem if file_path != 'N/A' else "Unknown Title"

                    heading = result.get('heading', None)
                    score = result.get('similarity_score')
                    
                    header = f"Result {i}: {note_title}"
                    if score is not None:
                         header += f" (Score: {score:.4f})"
                    elif search_performed == "Keyword" or (search_performed == "Semantic + Keyword" and score is None):
                         header += f" (Keyword Match)" # Indicate if it was likely a keyword fallback result

                    with st.expander(header):
                         # Create Obsidian link if possible
                         if relative_path and relative_path != file_path: # Check if relative path exists and is different from absolute
                             try:
                                 # Ensure the path is relative to the *vault*, not the index dir
                                 # This might require adjusting how relative_path is stored/retrieved
                                 encoded_path = urllib.parse.quote(relative_path, safe='')
                                 obsidian_link = f"obsidian://open?path={encoded_path}"
                                 st.markdown(f"**File:** [{file_path}]({obsidian_link})")
                             except Exception as link_err:
                                 logger.warning(f"Failed to create obsidian link for {relative_path}: {link_err}")
                                 st.markdown(f"**File:** {file_path}")
                         else:
                             st.markdown(f"**File:** {file_path}") # Fallback

                         if heading:
                             st.markdown(f"**Heading:** *{heading}*")
                         
                         st.markdown("**Content:**")
                         st.markdown(f"""```markdown
{result.get('content', '...')}
```""") # Use markdown code block for better display


        except Exception as e:
            st.error(f"An unexpected error occurred during search: {e}")
            logger.error(f"Search failed for query '{query}': {e}", exc_info=True)

elif search_button and not query:
    st.warning("Please enter a query.")

# Add instructions on how to run
st.sidebar.divider()
st.sidebar.markdown("### How to Run:")
st.sidebar.code("streamlit run app.py")
st.sidebar.markdown("Ensure you have indexed your vault first using the CLI:")
st.sidebar.code("python -m obsidian_ai_search index -v /path/to/your/vault") 