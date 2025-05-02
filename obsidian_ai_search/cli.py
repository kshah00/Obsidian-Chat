import click
import logging
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.console import Group
import numpy as np
import os
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
import urllib.parse # Import urlencode

from .indexer import Indexer
from .embedding import get_embedding_model, generate_embeddings
from .vector_store import FaissVectorStore
from .utils import get_query_suggestions, get_note_title
from .rag import generate_answer_with_gemini
from .config import APP_CONFIG # Import loaded config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup console for rich output
console = Console()

# --- Global cache for CLI (Load model/index once per session) ---
cli_resources = {
    "model": None,
    "vector_store": None,
    "index_dir": None
}

def load_cli_resources(index_dir: str):
    """Loads or reloads model and vector store if index_dir changes."""
    global cli_resources
    if cli_resources["index_dir"] == index_dir and cli_resources["model"] and cli_resources["vector_store"]:
        # Already loaded for this directory
        return True

    console.print("Loading embedding model...")
    try:
        model = get_embedding_model()
        if not model:
             console.print(":x: [bold red]Error:[/bold red] Failed to load embedding model.")
             return False
        cli_resources["model"] = model
    except Exception as e:
         console.print(f":x: [bold red]Error loading embedding model:[/bold red] {e}")
         logger.error(f"Failed to load model: {e}", exc_info=True)
         cli_resources["model"] = None
         return False

    console.print(f"Loading index from [cyan]{index_dir}[/cyan]...")
    vector_store = FaissVectorStore(index_dir=index_dir)
    try:
        vector_store.load()
        # We handle initialization/empty checks within the search command itself
        cli_resources["vector_store"] = vector_store
        cli_resources["index_dir"] = index_dir
        console.print(f"Index loaded ({vector_store.count} vectors). Accessing resources...")
        return True
    except Exception as e:
        console.print(f":x: [bold red]Error loading index:[/bold red] {e}")
        logger.error(f"Failed to load index from {index_dir}: {e}", exc_info=True)
        cli_resources["vector_store"] = None
        cli_resources["index_dir"] = None
        return False

# --- Core Search Logic (extracted for reuse) ---
def perform_search_and_display(query: str, top_k: int, gemini: bool):
    """Performs the search, RAG, and display logic."""
    model = cli_resources["model"]
    vector_store = cli_resources["vector_store"]

    if not model or not vector_store:
        console.print(":x: [bold red]Error:[/bold red] Resources (model/index) not loaded. Cannot search.")
        return

    # Check index status
    if not vector_store.is_initialized:
         console.print(f":x: [bold red]Error:[/bold red] Index not initialized in {cli_resources['index_dir']}.")
         return
    if vector_store.count == 0:
        console.print(":warning: [yellow]Index is empty.[/yellow]")
        return
    if model.get_sentence_embedding_dimension() != vector_store.dimension:
         console.print(f":x: [bold red]Error:[/bold red] Index/model dimension mismatch. Please re-index.")
         return

    try:
        # Embed query
        console.print("Embedding query...")
        query_chunk = [{'content': query}]
        query_with_embedding = generate_embeddings(query_chunk, model)
        if not query_with_embedding or 'embedding' not in query_with_embedding[0]:
            console.print(":x: [bold red]Error:[/bold red] Failed to generate query embedding.")
            return
        query_vector = query_with_embedding[0]['embedding']

        # --- Initial Search (Semantic + Keyword Fallback) ---
        gemini_answer = None
        console.print(f"\nPerforming initial semantic search for top {top_k} results...")
        initial_semantic_results = vector_store.search(query_vector, k=top_k)
        initial_search_results = initial_semantic_results
        initial_search_performed = "Semantic"

        if len(initial_semantic_results) < 3:
            console.print(f"[yellow]Semantic search returned < 3 results. Checking keyword...[/yellow]")
            keyword_limit = max(top_k, 5)
            initial_keyword_results = vector_store.keyword_search(query, limit=keyword_limit)
            if initial_keyword_results:
                initial_search_results = initial_keyword_results
                initial_search_performed = "Keyword"

        # --- Optional Gemini RAG ---
        if gemini and initial_search_results:
            console.print("\n:robot: Preparing context and calling Gemini API...")
            # console.print(f"Initial search results: {initial_search_results}")
            # Prepare context only from results that have text
            context_chunks = [res.get('content', '') for res in initial_search_results if res.get('content')]
            
            # Check if context_chunks is empty *after* filtering
            if not context_chunks:
                 console.print(":warning: [yellow]Skipping RAG: Initial results found, but none contained text content.[/yellow]")
                 gemini_answer = None # Ensure gemini_answer is None if we skip
            else:
                api_key = os.environ.get("GEMINI_API_KEY") 
                if not api_key:
                     console.print(":warning: [bold yellow]GEMINI_API_KEY not set. Skipping RAG.[/bold yellow]")
                     gemini_answer = None # Ensure gemini_answer is None
                else:
                    console.print("[dim]Sending query and context...[/dim]")
                    try:
                        gemini_answer = generate_answer_with_gemini(query, context_chunks, api_key)
                        if gemini_answer and not gemini_answer.startswith("Error:"):
                             console.print(":sparkles: [bold green]Received Gemini answer.[/bold green]")
                        elif gemini_answer: # Handle None case from RAG function explicitly
                             console.print(f":warning: [yellow]Gemini RAG issue: {gemini_answer}[/yellow]")
                        else: # Handles the case where RAG returns None (e.g. no context was actually useful)
                             console.print(":warning: [yellow]Gemini RAG returned no answer (potentially empty context passed).[/yellow]")
                    except Exception as rag_err:
                        console.print(f":x: [bold red]Error during RAG: {rag_err}[/bold red]")
                        gemini_answer = f"Error during RAG: {rag_err}"
        elif gemini and not initial_search_results:
             console.print("\n:robot: [yellow]Skipping RAG (no results).[/yellow]")
             gemini_answer = None # Explicitly set to None

        # --- Display Loop with Pagination ---
        current_k = top_k
        displayed_count = 0
        all_results_so_far = initial_search_results
        search_performed = initial_search_performed
        
        if gemini_answer and not gemini_answer.startswith("Error:"):
             console.print(Panel(Markdown(gemini_answer), title=":robot: Gemini Answer", border_style="magenta"))
             console.print("--- Source Results ---")
        elif gemini_answer:
             console.print(Panel(f"[red]{gemini_answer}[/red]", title=":robot: Gemini RAG Error", border_style="red"))

        while True:
            results_this_iteration = []
            needs_fetch = False
            if displayed_count < len(all_results_so_far):
                results_this_iteration = all_results_so_far[displayed_count:]
            elif displayed_count >= current_k:
                # This condition means we've shown K results and user must have asked for more
                needs_fetch = True
            # If displayed_count < current_k but >= len(all_results), means initial search found less than k, don't fetch yet

            if needs_fetch:
                console.print(f"\nPerforming {search_performed} search for top {current_k} results...")
                if search_performed == "Semantic":
                    fetched_results = vector_store.search(query_vector, k=current_k)
                else:
                    keyword_limit = max(current_k, 5)
                    fetched_results = vector_store.keyword_search(query, limit=keyword_limit)
                
                if len(fetched_results) > len(all_results_so_far):
                    all_results_so_far = fetched_results
                    results_this_iteration = all_results_so_far[displayed_count:]
                else:
                    # No new results found in the fetch
                    results_this_iteration = [] 

            # --- Display Logic ---
            if not results_this_iteration and displayed_count == 0:
                console.print(f":mag_right: [yellow]No results found via {search_performed.lower()} search for your query: '{query}'[/yellow]")
                try:
                    note_titles = vector_store.get_all_note_titles()
                    if note_titles:
                        suggestions = get_query_suggestions(query, note_titles)
                        if suggestions:
                            suggestions_str = ", ".join([f"'{s}'" for s in suggestions])
                            console.print(f"Did you mean: {suggestions_str}?")
                except Exception as suggestion_err:
                    logger.error(f"Error generating suggestions: {suggestion_err}", exc_info=True)
                break # Exit loop if no initial results
            
            elif not results_this_iteration and displayed_count > 0:
                 console.print("[i]No more results found.[/i]")
                 break
            
            else: # We have results for this iteration
                 if displayed_count == 0 and not gemini_answer: 
                      console.print(f":page_facing_up: [bold green]Found {len(all_results_so_far)} results (via {search_performed} search):[/bold green]")
                 elif displayed_count > 0: 
                      console.print(f":page_facing_up: [bold green]Showing more results (via {search_performed} search):[/bold green]")
                 
                 for i, result in enumerate(results_this_iteration, start=displayed_count + 1):
                     file_path = result.get('file_path', 'N/A') # Absolute path
                     relative_path = result.get('relative_path') # Relative path from chunker
                     
                     try:
                         note_title = get_note_title(file_path) # Still use absolute for title extraction logic
                     except (ImportError, NameError, Exception):
                         note_title = Path(file_path).stem if file_path != 'N/A' else "Unknown Title"
                         
                     heading = result.get('heading', None)
                     score_val = result.get('similarity_score')
                     if score_val is not None:
                         panel_title = f"Result {i}: {note_title} (Score: {score_val:.4f})"
                     else:
                         panel_title = f"Result {i}: {note_title} (Keyword Match)"
                         
                     # Create Obsidian link if relative path is available
                     if relative_path and relative_path != file_path: # Check if relative path is valid
                         try:
                             encoded_path = urllib.parse.quote(relative_path, safe='')
                             obsidian_link = f"obsidian://open?path={encoded_path}"
                             file_display = f"[link={obsidian_link}]{file_path}[/link]"
                         except Exception as link_err:
                             logger.warning(f"Failed to create obsidian link for {relative_path}: {link_err}")
                             file_display = f"[cyan]{file_path}[/cyan]"
                     else:
                          file_display = f"[cyan]{file_path}[/cyan]" # Fallback if no relative path
                          
                     panel_content = f"[bold]File:[/bold] {file_display}\n"
                     if heading:
                         panel_content += f"[bold]Heading:[/bold] [italic]{heading}[/italic]\n"
                     
                     # Use the correct key 'content' instead of 'text'
                     content_markdown = Markdown(result.get('content', '...'), style="", code_theme="default") 
                     panel_renderable = f"{panel_content}\n[bold]Content:[/bold]\n"
                     console.print(Panel(Group(panel_renderable, content_markdown), title=panel_title, border_style="blue"))
                 
                 displayed_count += len(results_this_iteration)

                 # --- Pagination Prompt Logic ---
                 potential_more_results = False
                 if search_performed == "Semantic" and len(all_results_so_far) == current_k:
                     potential_more_results = True
                 elif search_performed == "Keyword":
                      last_keyword_limit = current_k # Keyword limit grows with current_k in fetch logic
                      if len(all_results_so_far) == last_keyword_limit:
                         potential_more_results = True
                 
                 if potential_more_results and displayed_count == current_k: # Only ask if we've shown up to K
                     show_more = click.confirm("\nShow more results?", default=False)
                     if show_more:
                         current_k += top_k # Increase k for the next potential fetch
                         # Continue loop to fetch more
                     else:
                         console.print("\n--- End of Results ---")
                         break
                 elif displayed_count >= len(all_results_so_far) and not needs_fetch:
                     # This means we displayed all results found initially and there weren't enough to trigger pagination check
                      console.print("\n--- End of Results (All found) ---")
                      break
                 # If potential_more_results is False, we implicitly know we've shown all
                 elif not potential_more_results and displayed_count >= len(all_results_so_far):
                     console.print("\n--- End of Results (All found) ---")
                     break
                     
    except Exception as e:
        console.print(f":x: [bold red]An unexpected error occurred during search:[/bold red] {e}")
        logger.error(f"Search function failed for query '{query}': {e}", exc_info=True)

# --- Click CLI Commands ---
@click.group()
def cli():
    """Obsidian AI Search: Index and search your notes locally."""
    pass

@cli.command()
@click.option('--vault-path', '-v', required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True),
              help='Absolute path to the Obsidian vault directory.')
@click.option('--index-dir', '-i', type=click.Path(file_okay=False, dir_okay=True, writable=True, resolve_path=True),
              default=APP_CONFIG['index_dir'], show_default=True,
              help='Directory to store index files.')
@click.option('--force-reindex', is_flag=True, default=False, 
              help='Force a complete re-indexing of the vault, ignoring existing state.')
def index(vault_path: str, index_dir: str, force_reindex: bool):
    """Build or update the search index for the specified vault."""
    console.print(f":floppy_disk: Starting indexing process for vault: [cyan]{vault_path}[/cyan]")
    console.print(f":file_folder: Index will be stored in: [cyan]{index_dir}[/cyan]")
    if force_reindex:
        console.print(":warning: [bold yellow]Forcing re-index.[/bold yellow]")

    try:
        # Ensure index directory exists
        Path(index_dir).mkdir(parents=True, exist_ok=True)
        
        indexer = Indexer(vault_path=vault_path, index_dir=index_dir)
        indexer.build_index_from_scratch(force_reindex=force_reindex)
        console.print(f":white_check_mark: [bold green]Indexing complete.[/bold green] Index contains {indexer.vector_store.count} vectors.")
    except ValueError as e:
        console.print(f":x: [bold red]Error:[/bold red] {e}")
        logger.error(f"Indexing failed: {e}", exc_info=True)
    except RuntimeError as e:
        console.print(f":x: [bold red]Runtime Error:[/bold red] {e}")
        logger.error(f"Indexing failed: {e}", exc_info=True)
    except Exception as e:
        console.print(f":x: [bold red]An unexpected error occurred during indexing:[/bold red] {e}")
        logger.error(f"Indexing failed with unexpected error: {e}", exc_info=True)

@cli.command()
@click.argument('query', required=False)
@click.option('--index-dir', '-i', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True),
              default=APP_CONFIG['index_dir'], show_default=True,
              help='Directory where index files are stored.')
@click.option('--top-k', '-k', type=int, default=APP_CONFIG['default_top_k'], show_default=True, help='Default number of results to return per query.')
@click.option('--gemini/--no-gemini', default=False, help='Enable/disable Gemini RAG for answers.')
def search(query: str, index_dir: str, top_k: int, gemini: bool):
    """Search the index for notes matching the query (single shot)."""
    if not query:
        console.print(":x: [bold red]Error:[/bold red] Please provide a search query.")
        return
        
    console.print(f":mag: Searching index in [cyan]{index_dir}[/cyan] for: '[yellow]{query}[/yellow]' (Top {top_k}) - Single Shot")
    if gemini:
        # Check for API key early if Gemini is requested
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            console.print(":warning: [bold red]Error:[/bold red] Gemini RAG requested, but the 'GEMINI_API_KEY' environment variable is not set.")
            console.print("Please set the environment variable and try again.")
            return

    # Load resources once for this command
    if not load_cli_resources(index_dir):
         return

    # Perform the search using the extracted function
    perform_search_and_display(query, top_k, gemini)

@cli.command()
@click.option('--index-dir', '-i', type=click.Path(file_okay=False, dir_okay=True, writable=True, resolve_path=True),
              default=APP_CONFIG['index_dir'], show_default=True,
              help='Directory where index files are stored.')
@click.option('--top-k', '-k', type=int, default=APP_CONFIG['default_top_k'], show_default=True, help='Default number of results to return per query.')
@click.option('--gemini/--no-gemini', default=False, help='Enable/disable Gemini RAG for answers.')
def repl(index_dir: str, top_k: int, gemini: bool):
    """Start an interactive search session (REPL)."""
    console.print(":rocket: Starting Interactive Search Session...")
    console.print(f":file_folder: Using index directory: [cyan]{index_dir}[/cyan]")
    
    # Initial resource load
    if not load_cli_resources(index_dir):
        console.print(":x: [bold red]Failed to load initial resources. Exiting REPL.[/bold red]")
        return
        
    # Display startup stats
    vector_store = cli_resources["vector_store"]
    if vector_store and vector_store.is_initialized:
        chunk_count = vector_store.count
        # Get unique file count from the reverse map keys
        file_count = len(vector_store.file_path_to_chunk_ids)
        console.print(f":bar_chart: Index loaded: [bold green]{chunk_count}[/bold green] chunks from [bold green]{file_count}[/bold green] notes.")
    elif vector_store:
         console.print(":warning: Index directory exists but index seems empty or invalid.")
    # If vector_store is None, load_cli_resources already printed an error
        
    if gemini:
        # Check API key once at the start of REPL
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            console.print(":warning: [bold yellow]GEMINI_API_KEY not set. Gemini RAG will be disabled for this session.[/bold yellow]")
            gemini = False # Disable if key missing
        else:
             console.print(":robot: [bold magenta]Gemini RAG enabled for this session.[/bold magenta]")

    # Set up prompt session with history
    history_path = Path(index_dir) / '.repl_history'
    session = PromptSession(history=FileHistory(str(history_path)))

    console.print("\nEnter your search query. Type '/help' for commands, '/quit' to exit.")

    while True:
        try:
            user_input = session.prompt(
                'search> ',
                auto_suggest=AutoSuggestFromHistory(),
                refresh_interval=0.5 # Check for history updates periodically
            )
            user_input = user_input.strip()

            if not user_input:
                continue

            if user_input.lower() == '/quit':
                console.print("Exiting search session.")
                break
            elif user_input.lower() == '/help':
                console.print("Available commands:")
                console.print("  /quit          - Exit the interactive session.")
                console.print("  /gemini [on|off] - Toggle Gemini RAG (requires restart if turning on without API key).")
                console.print("  /topk <number> - Set the number of results to show (e.g., /topk 10).")
                console.print("  /help          - Show this help message.")
                console.print("Any other input is treated as a search query.")
                continue
            elif user_input.lower().startswith('/gemini'):
                 parts = user_input.split()
                 if len(parts) == 2:
                     if parts[1].lower() == 'on':
                         api_key = os.environ.get("GEMINI_API_KEY")
                         if not api_key:
                              console.print(":warning: Cannot enable Gemini: GEMINI_API_KEY not set.")
                         else:
                             gemini = True
                             console.print(":robot: Gemini RAG enabled.")
                     elif parts[1].lower() == 'off':
                         gemini = False
                         console.print(":robot: Gemini RAG disabled.")
                     else:
                          console.print("Usage: /gemini [on|off]")
                 else:
                      console.print(f"Gemini RAG is currently {'ON' if gemini else 'OFF'}.")
                 continue
            elif user_input.lower().startswith('/topk'):
                 parts = user_input.split()
                 if len(parts) == 2 and parts[1].isdigit():
                     new_k = int(parts[1])
                     if new_k > 0:
                         top_k = new_k
                         console.print(f"Number of results set to: {top_k}")
                     else:
                         console.print("Number of results must be positive.")
                 else:
                      console.print("Usage: /topk <number>")
                 continue
            
            # Treat as search query
            perform_search_and_display(user_input, top_k, gemini)

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            console.print("\nUse /quit to exit.")
            continue
        except EOFError:
            # Handle Ctrl+D
            console.print("\nExiting search session.")
            break
        except Exception as loop_err:
            console.print(f":x: [bold red]An error occurred in the REPL loop:[/bold red] {loop_err}")
            logger.error(f"REPL loop error: {loop_err}", exc_info=True)
            # Decide whether to continue or break? Continue for now.

if __name__ == '__main__':
    cli() 