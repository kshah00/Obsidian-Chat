import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import torch # sentence-transformers uses PyTorch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Loads and returns the specified Sentence Transformer model.
    
    Args:
        model_name: The name of the model to load (e.g., 'all-MiniLM-L6-v2').
    
    Returns:
        The loaded SentenceTransformer model.
    """
    logger.info(f"Loading embedding model: {model_name}")
    # Check if CUDA is available, otherwise use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    try:
        model = SentenceTransformer(model_name, device=device)
        logger.info(f"Model {model_name} loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}", exc_info=True)
        raise

def generate_embeddings(chunks: List[Dict[str, Any]], model: SentenceTransformer, batch_size: int = 32) -> List[Dict[str, Any]]:
    """Generates embeddings for a list of text chunks and adds them to the chunk dictionaries.
    
    Args:
        chunks: A list of dictionaries, where each dict represents a chunk 
                and must have a 'content' key with the text.
        model: The loaded SentenceTransformer model.
        batch_size: The number of chunks to process in one batch for efficiency.

    Returns:
        The same list of chunks, with an 'embedding' key added to each dictionary,
        containing the generated numpy array embedding.
        Returns an empty list if input chunks is empty.
    """
    if not chunks:
        logger.warning("generate_embeddings called with an empty list of chunks.")
        return []

    # Extract the text content from each chunk
    contents = [chunk['content'] for chunk in chunks]
    
    logger.info(f"Generating embeddings for {len(contents)} chunks...")
    try:
        # Encode the contents in batches
        embeddings = model.encode(contents, batch_size=batch_size, show_progress_bar=True)
        logger.info("Embeddings generated successfully.")
        
        # Add the generated embeddings back to the chunk dictionaries
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i]
            
        return chunks
        
    except Exception as e:
        logger.error(f"Error during embedding generation: {e}", exc_info=True)
        # Depending on desired robustness, could return partially processed chunks
        # or raise the exception. Returning empty for now to signal failure.
        return []

# Example Usage (for testing)
if __name__ == '__main__':
    # Sample chunks (replace with actual chunks from chunker.py if needed)
    sample_chunks = [
        {'chunk_id': 'file1#0', 'file_path': 'file1.md', 'title': 'File One', 'heading': 'Intro', 'content': 'This is the first document content.'},
        {'chunk_id': 'file1#1', 'file_path': 'file1.md', 'title': 'File One', 'heading': 'Details', 'content': 'More details about the first document.'},
        {'chunk_id': 'file2#0', 'file_path': 'file2.md', 'title': 'File Two', 'heading': None, 'content': 'Content for the second file.'}
    ]

    try:
        print("Loading model...")
        embedding_model = get_embedding_model()
        
        print("\nGenerating embeddings...")
        chunks_with_embeddings = generate_embeddings(sample_chunks, embedding_model)
        
        if chunks_with_embeddings:
            print("\nEmbeddings generated:")
            for chunk in chunks_with_embeddings:
                print(f"--- Chunk ID: {chunk['chunk_id']} ---")
                print(f"Content: {chunk['content']}")
                # print(f"Embedding: {chunk['embedding']}") # Usually too long to print
                print(f"Embedding shape: {chunk['embedding'].shape}") 
                print(f"Embedding dtype: {chunk['embedding'].dtype}")
        else:
            print("\nFailed to generate embeddings.")
            
    except Exception as e:
        print(f"An error occurred: {e}") 