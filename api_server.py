import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from pathlib import Path
import numpy as np

# Import necessary components from our project
from obsidian_ai_search.vector_store import FaissVectorStore
from obsidian_ai_search.embedding import get_embedding_model
from obsidian_ai_search.utils import get_query_suggestions, get_note_title

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# TODO: Make these configurable (e.g., via env vars or config file)
DEFAULT_INDEX_DIR = ".obsidian_ai_search_index"
DEFAULT_INDEX_NAME = "index" # Base name used by FaissVectorStore

# --- Global State (Load model and index on startup) ---
MODEL = None
VECTOR_STORE = None
INDEX_LOAD_ERROR = None # Track if loading failed

def load_resources():
    global MODEL, VECTOR_STORE, INDEX_LOAD_ERROR
    logger.info("Loading resources (embedding model and vector index)...")
    try:
        MODEL = get_embedding_model()
        logger.info("Embedding model loaded successfully.")
        
        index_dir_path = Path(DEFAULT_INDEX_DIR)
        VECTOR_STORE = FaissVectorStore(index_dir=str(index_dir_path), index_name=DEFAULT_INDEX_NAME)
        VECTOR_STORE.load()
        
        if not VECTOR_STORE.is_initialized:
            logger.warning(f"Vector index not found or empty in {index_dir_path}. API search will return no results until index is built.")
            # Not raising error, API can still run but search won't work
        elif VECTOR_STORE.dimension != MODEL.get_sentence_embedding_dimension():
            mismatch_msg = f"Index dimension ({VECTOR_STORE.dimension}) does not match model dimension ({MODEL.get_sentence_embedding_dimension()})!"
            logger.error(mismatch_msg)
            INDEX_LOAD_ERROR = ValueError(mismatch_msg) # Store error
            VECTOR_STORE = None # Prevent usage
        else:
            logger.info(f"Vector index loaded successfully from {index_dir_path} ({VECTOR_STORE.count} vectors).")

    except Exception as e:
        logger.error(f"Failed to load resources: {e}", exc_info=True)
        INDEX_LOAD_ERROR = e # Store the exception
        MODEL = None
        VECTOR_STORE = None

# Call load_resources() when the module is loaded
load_resources()

# --- API Definition ---
app = FastAPI(
    title="Obsidian AI Search API",
    description="API for searching indexed Obsidian notes.",
    version="0.1.0",
)

class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

class SearchResultItem(BaseModel):
    note_title: str
    excerpt: str
    score: float
    file_path: str | None = None # Optional: useful for obsidian:// links

class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    message: str | None = None

@app.post("/search", response_model=SearchResponse)
async def search_notes(search_query: SearchQuery):
    """
    Performs a semantic search on the indexed Obsidian notes.
    """
    logger.info(f"Received search request: query='{search_query.query}', top_k={search_query.top_k}")

    # Check if resources failed to load
    if INDEX_LOAD_ERROR:
        logger.error(f"Search unavailable due to resource load error: {INDEX_LOAD_ERROR}")
        raise HTTPException(status_code=503, detail=f"Search service unavailable: {INDEX_LOAD_ERROR}")
    if not MODEL or not VECTOR_STORE or not VECTOR_STORE.is_initialized:
        logger.warning("Search attempted but model or index not ready.")
        raise HTTPException(status_code=503, detail="Search index is not available or not initialized. Please build the index first.")

    try:
        # 1. Embed the query
        logger.debug("Embedding search query...")
        # Model expects a list of sentences, generate_embeddings handles this
        # Need a temporary structure like the chunker output for generate_embeddings
        # OR call the model directly if possible
        query_embedding = MODEL.encode([search_query.query], convert_to_numpy=True)
        if query_embedding is None or query_embedding.size == 0:
             logger.error("Failed to embed query.")
             raise HTTPException(status_code=500, detail="Failed to generate query embedding.")

        # 2. Perform semantic search
        logger.info(f"Performing semantic search with top_k={search_query.top_k}...")
        semantic_results_raw = VECTOR_STORE.search(query_embedding, k=search_query.top_k)
        search_performed = "Semantic"
        response_message = None

        # 3. Fallback to keyword search if semantic results are insufficient
        if len(semantic_results_raw) < 3:
            logger.info(f"Semantic search returned only {len(semantic_results_raw)} results. Falling back to keyword search.")
            keyword_limit = max(search_query.top_k, 5)
            keyword_results_raw = VECTOR_STORE.keyword_search(search_query.query, limit=keyword_limit)
            search_results_raw = keyword_results_raw # Replace results
            search_performed = "Keyword"
            if len(semantic_results_raw) > 0:
                 response_message = f"Showing keyword results as semantic search found < 3 matches."
            else:
                 response_message = f"Showing keyword results as semantic search found no matches."
        else:
             search_results_raw = semantic_results_raw

        # 4. Format results
        formatted_results = []
        for res in search_results_raw:
            file_path = res.get('file_path', 'Unknown File')
            try:
                note_title = get_note_title(file_path)
            except Exception:
                note_title = Path(file_path).stem
            
            # Keyword results have score=None, API expects float. Use 0.0 for keyword?
            # Or change API model? Let's use 0.0 for now and maybe adjust later.
            score_val = res.get('similarity_score', 0.0) 
            if score_val is None: # Handle explicit None from keyword search
                score_val = 0.0 # Represent keyword match score as 0?
                
            formatted_results.append(
                SearchResultItem(
                    note_title=note_title,
                    excerpt=res.get('text', '...'),
                    score=score_val, # Send float score
                    file_path=file_path
                )
            )

        if not formatted_results:
            logger.info(f"No results found via {search_performed.lower()} search for query: '{search_query.query}'")
            
            # Attempt to provide suggestions
            suggestions_list = []
            final_message = response_message if response_message else "No results found."
            try:
                note_titles = VECTOR_STORE.get_all_note_titles()
                if note_titles:
                    suggestions_list = get_query_suggestions(search_query.query, note_titles)
                    if suggestions_list:
                        suggestions_str = ", ".join([f"'{s}'" for s in suggestions_list])
                        final_message += f" Did you mean: {suggestions_str}?"
            except Exception as suggestion_err:
                 logger.error(f"Error generating suggestions for API: {suggestion_err}", exc_info=True)
                 # Append simple no results message if suggestions fail

            # Clarify if keyword search was also attempted
            if search_performed == "Keyword" and not response_message and not suggestions_list:
                final_message = "No results found via semantic or keyword search."
            elif search_performed == "Keyword" and not response_message and suggestions_list:
                 final_message = f"No results found via semantic or keyword search. Did you mean: {', '.join([f'{s}' for s in suggestions_list])}?" # Reconstruct msg if needed
                
            return SearchResponse(results=[], message=final_message)

        logger.info(f"Returning {len(formatted_results)} results (via {search_performed}) for query: '{search_query.query}'")
        return SearchResponse(results=formatted_results, message=response_message)

    except Exception as e:
        logger.error(f"An error occurred during search for query '{search_query.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred during search: {e}")


@app.get("/")
async def root():
    # Include status of resources in the root response
    status = {
        "message": "Obsidian AI Search API is running.",
        "model_loaded": MODEL is not None,
        "index_loaded": VECTOR_STORE is not None and VECTOR_STORE.is_initialized,
        "index_vector_count": VECTOR_STORE.count if VECTOR_STORE else 0,
        "load_error": str(INDEX_LOAD_ERROR) if INDEX_LOAD_ERROR else None
    }
    return status

# Run with: uvicorn api_server:app --reload --port 8000
if __name__ == "__main__":
    logger.info("Starting FastAPI server with Uvicorn (programmatic run - recommend command line for reload)...")
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False) # Use string for app import with reload 