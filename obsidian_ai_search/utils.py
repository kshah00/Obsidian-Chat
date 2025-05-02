import os
from pathlib import Path
from typing import List
import hashlib
from rapidfuzz import process, fuzz

def find_markdown_files(vault_path: str) -> List[str]:
    """Recursively finds all markdown (.md) files in the given directory."""
    markdown_files = []
    base_path = Path(vault_path)
    if not base_path.is_dir():
        raise ValueError(f"Vault path is not a valid directory: {vault_path}")

    for item in base_path.rglob('*.md'):
        if item.is_file():
            # Check if item is directly within the vault or in a subfolder
            # Exclude files starting with '.' (like .DS_Store)
            if not item.name.startswith('.'):
                try:
                    # Attempt to get relative path. If it fails (e.g., permission issues, weird symlinks),
                    # skip this file.
                    relative_path = item.relative_to(base_path)
                    markdown_files.append(str(base_path / relative_path)) # Store absolute path
                except ValueError as e:
                    print(f"Skipping file {item} due to error determining relative path: {e}")
                except Exception as e:
                     print(f"Skipping file {item} due to unexpected error: {e}")

    return markdown_files

def calculate_sha256(file_path: str) -> str:
    """Calculates the SHA-256 hash of a file's content."""
    hasher = hashlib.sha256()
    try:
        with open(file_path, 'rb') as file:
            while True:
                # Read in chunks to handle large files efficiently
                chunk = file.read(8192)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        # This might happen if the file is deleted between discovery and hashing
        print(f"Warning: File not found for hashing: {file_path}")
        return ""
    except PermissionError:
        print(f"Warning: Permission denied when reading file for hashing: {file_path}")
        return ""
    except IOError as e:
        print(f"Warning: Error reading file for hashing {file_path}: {e}")
        return ""
    except Exception as e:
        print(f"Warning: Unexpected error hashing file {file_path}: {e}")
        return ""

def get_note_title(file_path: str) -> str:
    """Extracts a display title from a Markdown file path.
    
    Currently uses the filename stem. Can be enhanced later to read H1.
    """
    if not file_path:
        return "Unknown Title"
    try:
        return Path(file_path).stem
    except Exception:
        # Fallback if path parsing fails somehow
        return os.path.basename(file_path) if file_path else "Unknown Title"

# Function for query suggestions
def get_query_suggestions(query: str, choices: List[str], score_cutoff: int = 75, limit: int = 3) -> List[str]:
    """Provides spelling suggestions for a query based on a list of choices.

    Uses fuzzy string matching (Levenshtein distance based) to find the best matches.

    Args:
        query: The user's search query.
        choices: A list of potential correct strings (e.g., note titles).
        score_cutoff: Minimum similarity score (0-100) to consider a match.
        limit: The maximum number of suggestions to return.

    Returns:
        A list of suggested strings, ordered by similarity.
    """
    if not query or not choices:
        return []

    # Use rapidfuzz process.extract to find best matches
    # It returns tuples of (choice, score, index)
    # We use WRatio which handles different lengths well.
    results = process.extract(query, choices, scorer=fuzz.WRatio, score_cutoff=score_cutoff, limit=limit)

    # Extract just the suggested strings (the first element of each tuple)
    suggestions = [result[0] for result in results]
    
    return suggestions 