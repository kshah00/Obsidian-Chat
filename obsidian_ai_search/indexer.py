import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from .utils import find_markdown_files, calculate_sha256
from .chunker import chunk_markdown
from .embedding import get_embedding_model, generate_embeddings
from .vector_store import FaissVectorStore
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define path for storing the hash tracking file
STATE_FILE_NAME = "indexing_state.json"

class Indexer:
    """Orchestrates the process of indexing Obsidian notes."""

    def __init__(self, vault_path: str, index_dir: str = ".", index_name: str = "index"):
        self.vault_path = Path(vault_path)
        if not self.vault_path.is_dir():
            raise ValueError(f"Vault path is not a valid directory: {vault_path}")
        
        self.index_dir = Path(index_dir)
        self.state_file_path = self.index_dir / STATE_FILE_NAME
        self.vector_store = FaissVectorStore(index_dir=str(index_dir), index_name=index_name)
        self.embedding_model: Optional[SentenceTransformer] = None
        self.processed_files: Dict[str, str] = {} # Maps absolute file_path -> content_hash

    def _load_state(self):
        """Loads the processed file hashes from the state file."""
        if self.state_file_path.exists():
            try:
                with open(self.state_file_path, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
                    self.processed_files = state_data.get('processed_files', {})
                    logger.info(f"Loaded indexing state for {len(self.processed_files)} files from {self.state_file_path}")
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from state file {self.state_file_path}. Starting with empty state.")
                self.processed_files = {}
            except Exception as e:
                logger.error(f"Error loading state file {self.state_file_path}: {e}. Starting with empty state.", exc_info=True)
                self.processed_files = {}
        else:
            logger.info("Indexing state file not found. Starting fresh.")
            self.processed_files = {}

    def _save_state(self):
        """Saves the current processed file hashes to the state file."""
        try:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            with open(self.state_file_path, 'w', encoding='utf-8') as f:
                json.dump({'processed_files': self.processed_files}, f, indent=4)
            logger.info(f"Saved indexing state for {len(self.processed_files)} files to {self.state_file_path}")
        except Exception as e:
            logger.error(f"Error saving state file {self.state_file_path}: {e}", exc_info=True)

    def _ensure_model_loaded(self):
        """Loads the embedding model if it hasn't been loaded yet."""
        if self.embedding_model is None:
            logger.info("Embedding model not loaded. Loading now...")
            self.embedding_model = get_embedding_model() # Uses default MiniLM
            if self.embedding_model is None:
                 raise RuntimeError("Failed to load the embedding model.")

    def build_index_from_scratch(self, force_reindex: bool = False):
        """Scans the vault, chunks, embeds, and indexes all markdown files.

        Args:
            force_reindex: If True, ignores existing state and re-indexes all files.
        """
        start_time = time.time()
        logger.info("Starting build index from scratch...")
        self._ensure_model_loaded()
        
        if force_reindex:
            logger.info("Force reindex requested. Clearing existing index and state.")
            self.vector_store = FaissVectorStore(index_dir=str(self.index_dir), index_name=self.vector_store.index_path.stem)
            self.processed_files = {}
        else:
             # Load existing index and state if not forcing reindex
            logger.info("Loading existing vector store and state...")
            self.vector_store.load() # Load index data
            self._load_state() # Load processed file hashes
        
        logger.info(f"Scanning vault: {self.vault_path}")
        all_md_files = find_markdown_files(str(self.vault_path))
        logger.info(f"Found {len(all_md_files)} markdown files.")

        files_to_process = []
        if force_reindex:
            files_to_process = all_md_files
            logger.info("Processing all found files due to force_reindex=True.")
        else:
            # Determine which files are new or modified
            for file_path in all_md_files:
                current_hash = calculate_sha256(file_path)
                if not current_hash: # Skip files that couldn't be hashed
                    continue 
                
                stored_hash = self.processed_files.get(file_path)
                if current_hash != stored_hash:
                    files_to_process.append(file_path)
                    if stored_hash:
                        logger.info(f"File modified: {Path(file_path).name}")
                    else:
                        logger.info(f"New file found: {Path(file_path).name}")
            logger.info(f"Identified {len(files_to_process)} new or modified files to process.")
            
            # Identify deleted files (present in state but not in current scan)
            deleted_files = [fp for fp in self.processed_files if fp not in all_md_files]
            if deleted_files:
                logger.info(f"Found {len(deleted_files)} files to remove from index: {[Path(f).name for f in deleted_files]}")
                chunks_to_remove = []
                for fp in deleted_files:
                    # Get all chunk_ids associated with this file and remove them
                    # This requires the vector store to have a way to map file_path -> chunk_ids
                    # Adding get_chunk_ids_by_file_path to FaissVectorStore
                    ids = self.vector_store.get_chunk_ids_by_file_path(fp)
                    if ids:
                        chunks_to_remove.extend(ids)
                    # Remove from processed_files state
                    del self.processed_files[fp]
                
                if chunks_to_remove:
                    logger.info(f"Removing {len(chunks_to_remove)} chunks for deleted files...")
                    self.vector_store.remove_by_chunk_ids(chunks_to_remove)
                else:
                     logger.info("No corresponding chunks found in index for deleted files.")

        if not files_to_process:
            logger.info("No new or modified files to process.")
        else:
            # Process new/modified files
            all_chunks_to_add = []
            processed_file_hashes = {} # Track hashes of files successfully processed in this run
            for file_path in files_to_process:
                logger.debug(f"Processing: {file_path}")
                file_hash = calculate_sha256(file_path)
                if not file_hash: continue # Skip if hashing failed

                # If file was modified (not new), remove its old chunks first
                if not force_reindex and file_path in self.processed_files: 
                    logger.debug(f"Removing old chunks for modified file: {Path(file_path).name}")
                    old_chunk_ids = self.vector_store.get_chunk_ids_by_file_path(file_path)
                    if old_chunk_ids:
                        self.vector_store.remove_by_chunk_ids(old_chunk_ids)
                
                # Pass vault path to chunker to get relative paths
                chunks = chunk_markdown(file_path, str(self.vault_path))
                if not chunks:
                    logger.warning(f"No chunks generated for {file_path}")
                    # If file becomes empty, ensure it's tracked as processed with current hash
                    processed_file_hashes[file_path] = file_hash 
                    continue
                
                logger.debug(f"Generated {len(chunks)} chunks for {Path(file_path).name}")
                chunks_with_embeddings = generate_embeddings(chunks, self.embedding_model)
                
                if chunks_with_embeddings:
                    all_chunks_to_add.extend(chunks_with_embeddings)
                    processed_file_hashes[file_path] = file_hash # Mark as processed with this hash
                else:
                    logger.error(f"Failed to generate embeddings for {file_path}. Skipping file.")

            if all_chunks_to_add:
                logger.info(f"Adding {len(all_chunks_to_add)} new/updated chunks to the vector store...")
                self.vector_store.add(all_chunks_to_add)
            
            # Update the main processed_files state with hashes from this run
            self.processed_files.update(processed_file_hashes)

        # Save final state and index
        logger.info("Saving vector store and indexing state...")
        self.vector_store.save()
        self._save_state()

        end_time = time.time()
        logger.info(f"Index build/update finished in {end_time - start_time:.2f} seconds.")
        logger.info(f"Index now contains {self.vector_store.count} vectors.")

# Example Usage:
if __name__ == '__main__':
    # Create dummy vault structure for testing
    VAULT_DIR = Path("./test_vault")
    INDEX_DIR = Path("./test_index_output")
    VAULT_DIR.mkdir(exist_ok=True)
    INDEX_DIR.mkdir(exist_ok=True)
    
    # Create some dummy files
    (VAULT_DIR / "note1.md").write_text("# Note 1\nContent for the first note.")
    (VAULT_DIR / "note2.md").write_text("---\ntitle: Note Two Title\n---\n# Note 2 Header\nMore content here.")
    sub_dir = VAULT_DIR / "subdir"
    sub_dir.mkdir(exist_ok=True)
    (sub_dir / "note3.md").write_text("## Note 3 in Subdir\nText.")

    print(f"Test vault created at: {VAULT_DIR.resolve()}")
    print(f"Index will be stored in: {INDEX_DIR.resolve()}")

    try:
        indexer = Indexer(vault_path=str(VAULT_DIR), index_dir=str(INDEX_DIR))
        
        print("\n--- Building index for the first time ---")
        indexer.build_index_from_scratch(force_reindex=True) # Force clean build first time

        print("\n--- Running index build again (should be incremental) ---")
        # Modify a file
        time.sleep(1) # Ensure modification time changes if filesystem relies on it
        (VAULT_DIR / "note1.md").write_text("# Note 1\nUPDATED Content for the first note.")
        # Add a new file
        (VAULT_DIR / "note4.md").write_text("Brand new note 4.")
        # Delete a file
        (sub_dir / "note3.md").unlink()

        indexer.build_index_from_scratch() # Run again, should detect changes

    except Exception as e:
        print(f"An error occurred during indexing: {e}")
    finally:
        # Clean up dummy files and dirs (optional)
        # import shutil
        # print("\nCleaning up test files...")
        # shutil.rmtree(VAULT_DIR, ignore_errors=True)
        # shutil.rmtree(INDEX_DIR, ignore_errors=True)
        pass 