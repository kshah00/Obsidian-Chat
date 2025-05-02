import faiss
import numpy as np
import os
import json
import logging
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaissVectorStore:
    """Manages a FAISS index for storing and searching text embeddings."""

    def __init__(self, index_dir: str = ".", index_name: str = "index"):
        """Initializes the vector store.

        Args:
            index_dir: Directory to save/load index files.
            index_name: Base name for index and metadata files (e.g., 'index').
        """
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / f"{index_name}.faiss"
        self.metadata_path = self.index_dir / f"{index_name}_metadata.json"
        self.index: Optional[faiss.IndexIDMap] = None # Specifically using IndexIDMap
        self.metadata: Dict[int, Dict[str, Any]] = {} # Maps FAISS index ID -> chunk metadata
        self.chunk_id_to_faiss_id: Dict[str, int] = {} # Maps chunk_id -> FAISS index ID
        self.file_path_to_chunk_ids: Dict[str, List[str]] = defaultdict(list) # Reverse map for fast lookup
        self._dimension: Optional[int] = None # Deduced from first added vector

    def _initialize_index(self, dimension: int):
        """Initializes a new FAISS index (IndexFlatL2 wrapped in IndexIDMap)."""
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
        self._dimension = dimension
        logger.info(f"Initialized new FAISS IndexIDMap(IndexFlatL2) with dimension {dimension}")

    def add(self, chunks: List[Dict[str, Any]]):
        """Adds chunk embeddings and metadata to the index.
        
        Assumes chunks contain 'embedding' (numpy array) and 'chunk_id' keys.
        Also stores other keys from the chunk dict as metadata.
        Handles initialization if the index doesn't exist.
        """
        if not chunks:
            return

        embeddings = [chunk['embedding'] for chunk in chunks if 'embedding' in chunk and isinstance(chunk['embedding'], np.ndarray)]
        valid_chunks = [chunk for chunk in chunks if 'embedding' in chunk and isinstance(chunk['embedding'], np.ndarray) and 'chunk_id' in chunk]

        if not embeddings:
            logger.warning("No valid embeddings found in the provided chunks. Nothing to add.")
            return
            
        embeddings_np = np.array(embeddings, dtype='float32')
        dimension = embeddings_np.shape[1]

        if self.index is None:
            self._initialize_index(dimension)
        elif dimension != self._dimension:
            error_msg = f"Dimension mismatch: Index is {self._dimension}D, new embeddings are {dimension}D."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check for duplicate chunk IDs before adding
        new_chunk_ids = {chunk['chunk_id'] for chunk in valid_chunks}
        existing_ids = new_chunk_ids.intersection(self.chunk_id_to_faiss_id.keys())
        if existing_ids:
            logger.warning(f"Attempting to add duplicate chunk IDs: {existing_ids}. Skipping duplicates.")
            # Filter out chunks with existing IDs
            valid_chunks = [chunk for chunk in valid_chunks if chunk['chunk_id'] not in existing_ids]
            embeddings = [chunk['embedding'] for chunk in valid_chunks]
            if not embeddings:
                logger.warning("All provided chunks had duplicate IDs. Nothing added.")
                return
            embeddings_np = np.array(embeddings, dtype='float32')

        # Generate unique FAISS IDs (simple approach: use current max ID + 1)
        # A more robust approach might involve UUIDs or managing an ID pool if deletions are frequent
        max_existing_faiss_id = max(self.metadata.keys()) if self.metadata else -1
        faiss_ids = np.arange(max_existing_faiss_id + 1, max_existing_faiss_id + 1 + len(embeddings_np), dtype=np.int64)

        try:
            self.index.add_with_ids(embeddings_np, faiss_ids)
            logger.info(f"Added {len(embeddings_np)} vectors to FAISS index.")

            # Update metadata mappings
            for i, chunk in enumerate(valid_chunks):
                faiss_id = int(faiss_ids[i]) # Ensure Python int for JSON keys later
                chunk_id = chunk['chunk_id']
                file_path = chunk.get('file_path') # Get file path from chunk data
                chunk_metadata = {k: v for k, v in chunk.items() if k != 'embedding'}
                self.metadata[faiss_id] = chunk_metadata
                self.chunk_id_to_faiss_id[chunk_id] = faiss_id
                # Update reverse map
                if file_path:
                    self.file_path_to_chunk_ids[file_path].append(chunk_id)
                
        except Exception as e:
            logger.error(f"Error adding vectors to FAISS index: {e}", exc_info=True)
            # TODO: Consider cleanup or marking index as potentially inconsistent

    def remove_by_chunk_ids(self, chunk_ids: List[str]) -> int:
        """Removes chunks from the index based on their chunk_ids.
        Also updates the reverse file path lookup map.
        Returns the number of items successfully removed from the index.
        """
        if not self.index or self.index.ntotal == 0:
            # logger.warning("Attempted to remove from an empty or non-existent index.")
            return 0

        faiss_ids_to_remove = []
        chunk_details_to_remove = {} # Store chunk_id: {faiss_id, file_path}
        
        for chunk_id in chunk_ids:
            if chunk_id in self.chunk_id_to_faiss_id:
                faiss_id = self.chunk_id_to_faiss_id[chunk_id]
                # Find metadata to get file_path for reverse map update
                meta = self.metadata.get(faiss_id)
                file_path = meta.get('file_path') if meta else None
                faiss_ids_to_remove.append(faiss_id)
                chunk_details_to_remove[chunk_id] = {'faiss_id': faiss_id, 'file_path': file_path}
            else:
                logger.warning(f"Chunk ID '{chunk_id}' not found in index mapping for removal.")

        if not faiss_ids_to_remove:
            return 0

        try:
            # FAISS expects an array of int64 for removal via IDSelectorArray
            ids_to_remove_np = np.array(faiss_ids_to_remove, dtype=np.int64)
            num_removed = self.index.remove_ids(faiss.IDSelectorArray(ids_to_remove_np))
            
            logger.info(f"Removed {num_removed} vectors from FAISS index for {len(faiss_ids_to_remove)} requested IDs.")

            if num_removed > 0:
                # Update metadata and reverse maps for removed items
                # We assume FAISS successfully removed the IDs we asked for
                removed_count_check = 0
                for chunk_id, details in chunk_details_to_remove.items():
                    faiss_id = details['faiss_id']
                    file_path = details['file_path']
                    
                    # Check if ID likely existed before attempting map removals
                    # (FAISS num_removed is the ultimate source of truth, but this adds safety)
                    id_existed = faiss_id in self.metadata and chunk_id in self.chunk_id_to_faiss_id

                    if id_existed:
                        # Remove from primary mappings
                        del self.metadata[faiss_id]
                        del self.chunk_id_to_faiss_id[chunk_id]
                        # Remove from reverse mapping
                        if file_path and file_path in self.file_path_to_chunk_ids:
                            try:
                                self.file_path_to_chunk_ids[file_path].remove(chunk_id)
                                # If list becomes empty, remove the key
                                if not self.file_path_to_chunk_ids[file_path]:
                                    del self.file_path_to_chunk_ids[file_path]
                            except ValueError:
                                logger.warning(f"Chunk ID {chunk_id} not found in reverse map list for file {file_path} during removal.")
                        removed_count_check += 1
                    else:
                        logger.warning(f"Metadata/Mapping inconsistency detected for chunk {chunk_id} / faiss_id {faiss_id} during removal cleanup.")
                
                if removed_count_check != num_removed:
                    logger.warning(f"FAISS removed {num_removed} items, but only {removed_count_check} were cleaned from mappings. Potential inconsistency.")
            return num_removed
        except Exception as e:
            logger.error(f"Error removing vectors from FAISS index: {e}", exc_info=True)
            return 0

    def get_chunk_ids_by_file_path(self, file_path: str) -> List[str]:
        """Retrieves all chunk_ids associated with a given file_path using the reverse map."""
        # Return a copy to prevent modification of the internal list
        return self.file_path_to_chunk_ids.get(file_path, []).copy()

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Searches the index for the k nearest neighbors to the query embedding.
        
        Returns:
            List of dictionaries, each containing metadata of a matching chunk 
            and an additional 'similarity_score' (lower is better for L2 distance).
        """
        if self.index is None or self.index.ntotal == 0:
            # logger.info("Search attempted on an empty or non-existent index.")
            return []

        # Ensure query embedding is 2D, float32, and C-contiguous for FAISS
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        contiguous_query_embedding = np.ascontiguousarray(query_embedding, dtype='float32')

        # Check dimension match *after* ensuring contiguity and dtype
        if contiguous_query_embedding.shape[1] != self._dimension:
            logger.error(f"Query dimension ({contiguous_query_embedding.shape[1]}) does not match index dimension ({self._dimension}).")
            return []

        # Adjust k if the index has fewer items than requested
        actual_k = min(k, self.index.ntotal)
        if actual_k <= 0: # Check for <= 0 as index might exist but be empty after removals
            return []
        
        try:
            # Perform the search
            distances, faiss_ids = self.index.search(contiguous_query_embedding, actual_k)
            
            results = []
            if len(faiss_ids) > 0: # Ensure faiss_ids is not empty
                for i in range(len(faiss_ids[0])): # Iterate through results for the first query vector
                    faiss_id = int(faiss_ids[0][i]) # Ensure integer ID
                    if faiss_id == -1: 
                        # FAISS returns -1 if fewer than k results are found
                        continue 
                    if faiss_id in self.metadata:
                        result_metadata = self.metadata[faiss_id].copy()
                        # Store L2 distance as similarity score (lower is more similar)
                        result_metadata['similarity_score'] = float(distances[0][i]) 
                        results.append(result_metadata)
                    else:
                        logger.warning(f"FAISS ID {faiss_id} found during search but missing in metadata! Index might be inconsistent.")
            
            return results
        except Exception as e:
            logger.error(f"Error during FAISS search: {e}", exc_info=True)
            return []

    def keyword_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Performs a simple case-insensitive keyword search over chunk text.
        
        Args:
            query: The keyword/phrase to search for.
            limit: The maximum number of results to return.

        Returns:
            List of dictionaries, each containing metadata of a matching chunk.
            Includes a 'similarity_score' of None for consistency.
        """
        if not self.metadata:
            logger.info("Keyword search attempted on empty metadata.")
            return []

        query_lower = query.lower()
        matches = []
        
        for faiss_id, meta in self.metadata.items():
            chunk_text = meta.get('text', '')
            if query_lower in chunk_text.lower():
                result_meta = meta.copy()
                result_meta['similarity_score'] = None # Indicate keyword match
                matches.append(result_meta)
                if len(matches) >= limit:
                    break # Stop once limit is reached
        
        logger.info(f"Keyword search for '{query}' found {len(matches)} matches (limit {limit}).")
        return matches

    def get_all_note_titles(self) -> List[str]:
        """Extracts unique note titles from the stored metadata.
        
        Uses 'file_path' in metadata and extracts the stem (filename without extension)
        as the basic title representation.
        TODO: Could potentially use a dedicated 'title' field if stored during indexing.
        """
        if not self.metadata:
            return []
        
        titles = set()
        for meta in self.metadata.values():
            file_path_str = meta.get('file_path')
            if file_path_str:
                try:
                    # Basic title extraction from filename
                    title = Path(file_path_str).stem 
                    titles.add(title)
                except Exception as e:
                    logger.warning(f"Could not parse title from file path {file_path_str}: {e}")
                    
        return sorted(list(titles))

    def save(self):
        """Saves the FAISS index, metadata, and mappings to disk."""
        logger.info(f"Attempting to save index to {self.index_path} and metadata to {self.metadata_path}")
        # Ensure directory exists
        self.index_dir.mkdir(parents=True, exist_ok=True)

        if self.index:
            try:
                faiss.write_index(self.index, str(self.index_path))
                logger.info(f"FAISS index saved successfully ({self.index.ntotal} vectors)." )
            except Exception as e:
                 logger.error(f"Error saving FAISS index to {self.index_path}: {e}", exc_info=True)
        else:
            logger.warning("Index object is None. Nothing to save for FAISS index.")
            # If index file exists but index obj is None, should we delete the file?
            if self.index_path.exists():
                logger.warning(f"Deleting existing index file {self.index_path} as current index object is None.")
                try:
                    self.index_path.unlink()
                except OSError as e:
                    logger.error(f"Failed to delete existing index file: {e}")

        try:
            # Save metadata and mappings
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                # Convert defaultdict back to regular dict for JSON serialization
                file_path_map_serializable = dict(self.file_path_to_chunk_ids)
                json.dump({
                    "metadata": self.metadata,
                    "chunk_id_to_faiss_id": self.chunk_id_to_faiss_id,
                    "file_path_to_chunk_ids": file_path_map_serializable, # Save the new map
                    "dimension": self._dimension
                }, f, indent=4)
            logger.info(f"Metadata and mappings saved successfully ({len(self.metadata)} metadata entries, {len(file_path_map_serializable)} file paths).")
        except TypeError as e:
             logger.error(f"Metadata contains non-serializable data: {e}", exc_info=True)
             logger.error(f"Failed to save metadata to {self.metadata_path}. State may be inconsistent.")
        except Exception as e:
            logger.error(f"Error saving metadata to {self.metadata_path}: {e}", exc_info=True)
            
    def load(self):
        """Loads the FAISS index, metadata, and mappings from disk.
           Returns True if loading was successful (or if files don't exist), False otherwise.
        """
        logger.info(f"Attempting to load index from {self.index_path} and metadata from {self.metadata_path}")
        index_loaded_successfully = False
        metadata_loaded_successfully = False

        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                # Important: Ensure it's an IndexIDMap after loading
                if not isinstance(self.index, faiss.IndexIDMap):
                     logger.warning(f"Loaded index from {self.index_path} is not an IndexIDMap. Re-wrapping.")
                     # This assumes the loaded index is compatible (e.g., IndexFlatL2)
                     # If the loaded index was fundamentally different, this might fail.
                     base_index = self.index # Keep ref to original loaded index
                     if hasattr(base_index, 'd'): # Check if dimension attribute exists
                         self._dimension = base_index.d
                         self.index = faiss.IndexIDMap(base_index) # Wrap it
                         # Note: This loses any existing IDs if the original wasn't IDMap!
                         # A proper migration path would be needed for index type changes.
                         logger.warning("Index re-wrapped. Original IDs might be lost if not IndexIDMap previously.")
                     else:
                          raise TypeError(f"Loaded index type {type(base_index)} has no dimension attribute 'd'. Cannot re-wrap.")
                else:
                    self._dimension = self.index.index.d # Get dimension from the underlying index
                
                logger.info(f"FAISS index loaded from {self.index_path} ({self.index.ntotal} vectors, {self._dimension} dimensions)." )
                index_loaded_successfully = True
            except Exception as e:
                 logger.error(f"Error loading FAISS index from {self.index_path}: {e}", exc_info=True)
                 self.index = None 
                 self._dimension = None
        else:
            logger.info(f"FAISS index file '{self.index_path}' not found. Will initialize a new index on first add.")
            index_loaded_successfully = True # No file is not an error state for loading

        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)
                    # JSON keys are strings, need to convert FAISS IDs back to int
                    self.metadata = {int(k): v for k, v in saved_data.get("metadata", {}).items()}
                    self.chunk_id_to_faiss_id = saved_data.get("chunk_id_to_faiss_id", {})
                    # Load the reverse map, converting back to defaultdict
                    loaded_file_path_map = saved_data.get("file_path_to_chunk_ids", {})
                    self.file_path_to_chunk_ids = defaultdict(list, loaded_file_path_map)
                    loaded_dimension = saved_data.get("dimension")
                    
                    # Consistency check
                    if index_loaded_successfully and self.index is not None:
                        if loaded_dimension != self._dimension:
                             logger.warning(f"Dimension mismatch! Index ({self._dimension}d) vs Metadata ({loaded_dimension}d). Using index dimension.")
                        if self.index.ntotal != len(self.metadata):
                             logger.warning(f"Index size ({self.index.ntotal}) != metadata size ({len(self.metadata)}). Potential inconsistency.")
                    elif loaded_dimension is not None:
                        self._dimension = loaded_dimension # Trust metadata if index wasn't loaded/doesn't exist
                    
                    logger.info(f"Metadata and mappings loaded from {self.metadata_path} ({len(self.metadata)} metadata entries, {len(self.file_path_to_chunk_ids)} file paths).")
                    metadata_loaded_successfully = True

            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from metadata file {self.metadata_path}: {e}")
                self.metadata = {} # Reset to empty
                self.chunk_id_to_faiss_id = {}
                self.file_path_to_chunk_ids = defaultdict(list)
            except Exception as e:
                 logger.error(f"Error loading metadata from {self.metadata_path}: {e}", exc_info=True)
                 self.metadata = {}
                 self.chunk_id_to_faiss_id = {}
                 self.file_path_to_chunk_ids = defaultdict(list)
        else:
             logger.info(f"Metadata file '{self.metadata_path}' not found. Mappings will be built as data is added.")
             metadata_loaded_successfully = True # No file is not an error state
        
        return index_loaded_successfully and metadata_loaded_successfully

    @property
    def dimension(self) -> Optional[int]:
        return self._dimension

    @property
    def is_initialized(self) -> bool:
        return self.index is not None

    @property
    def count(self) -> int:
        return self.index.ntotal if self.index else 0

# --- Example Usage --- 
if __name__ == '__main__':
    INDEX_DIR = Path("./test_faiss_index")
    INDEX_NAME = "my_test_index"
    INDEX_FILE = INDEX_DIR / f"{INDEX_NAME}.faiss"
    META_FILE = INDEX_DIR / f"{INDEX_NAME}_metadata.json"

    # Clean up previous test files if they exist
    print(f"Cleaning up previous test files in {INDEX_DIR}...")
    if INDEX_FILE.exists(): INDEX_FILE.unlink()
    if META_FILE.exists(): META_FILE.unlink()
    # Only remove dir if it exists and is empty (or handle non-empty case)
    # For simplicity, just try to remove; ignore errors if it doesn't exist
    try:
        INDEX_DIR.rmdir()
    except OSError:
        pass # Ignore if it doesn't exist or isn't empty (manual cleanup might be needed)

    # --- Create and Add --- 
    print("\n--- Initializing and Adding ---")
    vector_store = FaissVectorStore(index_dir=str(INDEX_DIR), index_name=INDEX_NAME)
    vector_store.load() # Try loading first (should find nothing)
    
    # Simulate some chunks with embeddings
    DIM = 384 # Dimension for MiniLM
    chunks_to_add = [
        {'chunk_id': 'doc1.md#0', 'file_path': 'doc1.md', 'title': 'Doc1', 'content': 'apple banana fruit', 'embedding': np.random.rand(DIM).astype('float32')},
        {'chunk_id': 'doc1.md#1', 'file_path': 'doc1.md', 'title': 'Doc1', 'content': 'orange grape citrus', 'embedding': np.random.rand(DIM).astype('float32')},
        {'chunk_id': 'doc2.md#0', 'file_path': 'doc2.md', 'title': 'Doc2', 'content': 'kiwi strawberry berry', 'embedding': np.random.rand(DIM).astype('float32')},
    ]
    vector_store.add(chunks_to_add)
    print(f"Index count after add: {vector_store.count}")
    print(f"Metadata FAISS IDs: {list(vector_store.metadata.keys())}")
    print(f"Chunk ID map keys: {list(vector_store.chunk_id_to_faiss_id.keys())}")
    
    # Add duplicate test
    print("\n--- Adding Duplicate Chunk ID ---")
    duplicate_chunk = [{'chunk_id': 'doc1.md#0', 'file_path': 'doc1_new.md', 'title': 'Doc1 New', 'content': 'updated apple content', 'embedding': np.random.rand(DIM).astype('float32')}]
    vector_store.add(duplicate_chunk)
    print(f"Index count after adding duplicate: {vector_store.count}") # Should be unchanged

    # --- Save --- 
    print("\n--- Saving ---")
    vector_store.save()
    print(f"Index file exists: {INDEX_FILE.exists()}")
    print(f"Metadata file exists: {META_FILE.exists()}")

    # --- Load --- 
    print("\n--- Loading ---")
    vector_store_loaded = FaissVectorStore(index_dir=str(INDEX_DIR), index_name=INDEX_NAME)
    load_success = vector_store_loaded.load()
    print(f"Load successful: {load_success}")
    print(f"Loaded index count: {vector_store_loaded.count}")
    print(f"Loaded dimension: {vector_store_loaded.dimension}")
    print(f"Loaded metadata FAISS IDs: {list(vector_store_loaded.metadata.keys())}")

    # --- Search --- 
    print("\n--- Searching ---")
    query_vec = np.random.rand(DIM).astype('float32')
    results = vector_store_loaded.search(query_vec, k=2)
    print(f"Search results for k=2: {len(results)}")
    for res in results:
        print(f"  ID: {res['chunk_id']}, Score: {res['similarity_score']:.4f}, File: {res['file_path']}")

    # --- Get chunks by file --- 
    print("\n--- Getting chunks by file path ('doc1.md') ---")
    doc1_chunks = vector_store_loaded.get_chunk_ids_by_file_path('doc1.md')
    print(f"Chunks for doc1.md: {doc1_chunks}")

    # --- Remove --- 
    print("\n--- Removing ('doc1.md#0', 'non_existent_id') --- ")
    ids_to_remove = ['doc1.md#0', 'non_existent_id']
    num_removed = vector_store_loaded.remove_by_chunk_ids(ids_to_remove)
    print(f"Attempted removal of {len(ids_to_remove)} IDs, FAISS removed: {num_removed}")
    print(f"Index count after remove: {vector_store_loaded.count}")
    print(f"Metadata FAISS IDs after remove: {list(vector_store_loaded.metadata.keys())}")
    print(f"Chunk ID map keys after remove: {list(vector_store_loaded.chunk_id_to_faiss_id.keys())}")
    
    # --- Search After Remove ---
    print("\n--- Searching After Remove ---")
    results_after_remove = vector_store_loaded.search(query_vec, k=2)
    print(f"Search results after remove for k=2: {len(results_after_remove)}")
    for res in results_after_remove:
         print(f"  ID: {res['chunk_id']}, Score: {res['similarity_score']:.4f}, File: {res['file_path']}")

    # --- Add More --- 
    print("\n--- Adding More ---")
    more_chunks = [
         {'chunk_id': 'doc3.md#0', 'file_path': 'doc3.md', 'title': 'Doc3', 'content': 'pineapple coconut tropical', 'embedding': np.random.rand(DIM).astype('float32')}
    ]
    vector_store_loaded.add(more_chunks)
    print(f"Index count after adding more: {vector_store_loaded.count}")
    print(f"Metadata FAISS IDs: {list(vector_store_loaded.metadata.keys())}")

    # --- Save Final --- 
    print("\n--- Saving Final ---")
    vector_store_loaded.save()

    print(f"\nTest files are in {INDEX_DIR}")
