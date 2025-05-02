import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import frontmatter

# Constants for chunking
TARGET_WORDS_MIN = 150  # Adjusted lower bound
TARGET_WORDS_MAX = 350  # Adjusted upper bound
# Regex to find markdown headings (e.g., # Heading, ## Heading)
HEADING_REGEX = re.compile(r"^(#{1,6})\s+(.*)")

def count_words(text: str) -> int:
    """Counts the approximate number of words in a text string."""
    return len(text.split())

def get_note_title(file_path: str, content: str, metadata: dict) -> str:
    """Determines the note title (filename, H1, or metadata title)."""
    # Check metadata first
    if metadata and isinstance(metadata.get("title"), str):
        title = metadata["title"].strip()
        if title:
            return title

    # Check for H1 heading in content
    lines = content.splitlines()
    for line in lines:
        line = line.strip()
        if line.startswith("# "):
            title = line[2:].strip()
            if title:
                return title

    # Fallback to filename (without extension)
    return Path(file_path).stem

def chunk_markdown(file_path: str, vault_base_path: str) -> List[Dict[str, Any]]:
    """Chunks a markdown file by paragraphs and headings, aiming for word count targets.

    Args:
        file_path: Absolute path to the markdown file.
        vault_base_path: Absolute path to the vault root directory.

    Returns:
        A list of dictionaries, each representing a chunk with:
        - 'content': The text content of the chunk.
        - 'file_path': The absolute original file path.
        - 'relative_path': The file path relative to the vault base.
        - 'chunk_id': A unique ID for the chunk within the file (e.g., file_path#0).
        - 'title': The determined title of the note.
        - 'heading': The nearest preceding heading text (if any).
        - 'heading_level': The level (1-6) of the nearest preceding heading.
    """
    try:
        file_path_obj = Path(file_path)
        vault_base_obj = Path(vault_base_path)
        relative_path = str(file_path_obj.relative_to(vault_base_obj))
    except ValueError:
        # If file is not relative to vault path (e.g., symlink outside?), log and use absolute.
        print(f"Warning: File {file_path} seems not relative to vault base {vault_base_path}. Using absolute path only.")
        relative_path = file_path # Fallback, though obsidian:// links might not work
    except Exception as path_e:
         print(f"Error calculating relative path for {file_path} against {vault_base_path}: {path_e}")
         relative_path = file_path # Fallback

    try:
        # Use autodetect_encoding=True for broader compatibility
        post = frontmatter.load(file_path, autodetect_encoding=True)
        content = post.content
        metadata = post.metadata
    except Exception as e:
        print(f"Error reading or parsing frontmatter for {file_path}: {e}")
        # Fallback to reading without frontmatter
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            metadata = {}
        except Exception as read_e:
            print(f"Error reading file {file_path}: {read_e}")
            return []  # Cannot process file

    note_title = get_note_title(file_path, content, metadata)

    # Split content into logical blocks based on double newlines.
    # This keeps paragraphs and headings together initially.
    blocks = [block.strip() for block in re.split(r"\n\s*\n", content) if block.strip()]

    chunks = []
    current_chunk_content = []
    current_chunk_word_count = 0
    current_heading: Optional[str] = None
    current_heading_level: Optional[int] = None
    chunk_index = 0

    for block in blocks:
        block_word_count = count_words(block)
        heading_match = HEADING_REGEX.match(block)

        # --- Handle Headings --- 
        if heading_match:
            level = len(heading_match.group(1)) # Count the '#' characters
            heading_text = heading_match.group(2).strip()

            # If current chunk has content, finalize it before starting anew with the heading
            if current_chunk_content:
                chunk_content_str = "\n\n".join(current_chunk_content).strip()
                chunks.append({
                    "content": chunk_content_str,
                    "file_path": file_path,
                    "relative_path": relative_path,
                    "chunk_id": f"{file_path}#{chunk_index}",
                    "title": note_title,
                    "heading": current_heading,
                    "heading_level": current_heading_level
                })
                chunk_index += 1
                current_chunk_content = []
                current_chunk_word_count = 0
            
            # Update the current heading context for subsequent blocks
            current_heading = heading_text
            current_heading_level = level
            
            # Add heading itself to the *start* of the next potential chunk
            current_chunk_content.append(block)
            current_chunk_word_count += block_word_count # Include heading words
            
            # If the heading *itself* is already large, make it its own chunk immediately
            if current_chunk_word_count >= TARGET_WORDS_MIN:
                chunk_content_str = "\n\n".join(current_chunk_content).strip()
                chunks.append({
                    "content": chunk_content_str,
                    "file_path": file_path,
                    "relative_path": relative_path,
                    "chunk_id": f"{file_path}#{chunk_index}",
                    "title": note_title,
                    "heading": current_heading, # Associate heading with itself
                    "heading_level": current_heading_level
                })
                chunk_index += 1
                current_chunk_content = []
                current_chunk_word_count = 0
                # Keep current_heading/level for the *next* block unless it's also a heading

            continue # Process next block

        # --- Handle Non-Heading Blocks (Paragraphs, Lists, Code blocks etc.) ---

        # Scenario 1: Chunk is empty, just add the block.
        if not current_chunk_content:
            current_chunk_content.append(block)
            current_chunk_word_count += block_word_count
        # Scenario 2: Adding block doesn't exceed MAX words.
        elif current_chunk_word_count + block_word_count <= TARGET_WORDS_MAX:
            current_chunk_content.append(block)
            current_chunk_word_count += block_word_count
        # Scenario 3: Chunk is too small (< MIN), force add block even if it exceeds MAX.
        # Prefer slightly larger chunks over tiny ones separated by small paragraphs.
        elif current_chunk_word_count < TARGET_WORDS_MIN:
            current_chunk_content.append(block)
            current_chunk_word_count += block_word_count
        # Scenario 4: Adding block *would* exceed MAX, and current chunk is big enough (>= MIN).
        # Finalize the current chunk and start a new one with the current block.
        else:
            chunk_content_str = "\n\n".join(current_chunk_content).strip()
            chunks.append({
                "content": chunk_content_str,
                "file_path": file_path,
                "relative_path": relative_path,
                "chunk_id": f"{file_path}#{chunk_index}",
                "title": note_title,
                "heading": current_heading,
                "heading_level": current_heading_level
            })
            chunk_index += 1
            # Start new chunk with the current block
            current_chunk_content = [block]
            current_chunk_word_count = block_word_count
            # Heading context carries over to this new chunk

    # Add the last remaining chunk if it has content
    if current_chunk_content:
        chunk_content_str = "\n\n".join(current_chunk_content).strip()
        chunks.append({
            "content": chunk_content_str,
            "file_path": file_path,
            "relative_path": relative_path,
            "chunk_id": f"{file_path}#{chunk_index}",
            "title": note_title,
            "heading": current_heading,
            "heading_level": current_heading_level
        })

    return chunks

# Example Usage:
if __name__ == '__main__':
    # Create a dummy markdown file for testing
    dummy_file = Path("dummy_note.md")
    dummy_content = """
---
title: My Test Note
tags: [example, chunking]
---

# Main Title (H1)

This is the first paragraph introducing the topic. It should be long enough to potentially meet the minimum word count, but maybe not the maximum.

This second paragraph continues the introduction. We want to see if it gets merged with the first one into a single chunk, as they are under the same H1 heading and likely won't exceed the maximum word count together.

## Section Alpha (H2)

Now we start section Alpha. This paragraph introduces the specifics of Alpha. It might be relatively short.

*   Here is a list item.
*   Another list item.
*   A third one. Lists should ideally stay within the same chunk as surrounding paragraphs if possible.

This paragraph follows the list in Section Alpha. Let's make this one a bit longer to test the upper word count limit. If the combination of the introductory paragraph, the list, and this paragraph exceeds TARGET_WORDS_MAX, a split should occur. We need enough words here to trigger that condition potentially. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

### Subsection Alpha.1 (H3)

A short note within a subsection. This might get chunked with the preceding long paragraph or start its own if the previous one was already finalized.

## Section Beta (H2)

This marks the beginning of Section Beta. It's a new H2 heading, so it should definitely force the end of the previous chunk (related to Alpha) if it hadn't ended already.

This is the only paragraph in Section Beta. It's quite short. Ideally, it might be its own chunk if the heading forced a split, or potentially merged later if a more sophisticated strategy was used (but for now, likely its own small chunk).

# Another Main Topic (H1)

This is a new top-level section. It should signal a clear break from Section Beta.

Final paragraph. This comes after the second H1. It provides some concluding thoughts.

"""
    dummy_file.write_text(dummy_content, encoding='utf-8')

    DUMMY_VAULT_PATH = str(Path(".").resolve()) # Use current dir as dummy vault

    print(f"Chunking file: {dummy_file.resolve()} relative to vault: {DUMMY_VAULT_PATH}")
    chunks = chunk_markdown(str(dummy_file.resolve()), DUMMY_VAULT_PATH)

    print(f"\nFound {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i} ---")
        print(f"ID: {chunk['chunk_id']}")
        print(f"File Path: {chunk['file_path']}")
        print(f"Relative Path: {chunk['relative_path']}") # Show relative path
        print(f"Title: {chunk['title']}")
        print(f"Heading: {chunk['heading']} (Level: {chunk['heading_level']})")
        print(f"Word Count: {count_words(chunk['content'])}")
        print(f"Content Preview:\n{chunk['content'][:200]}...\n") # Show preview
        # print(f"Content:\n{chunk['content']}\n") # Uncomment for full content

    # Clean up the dummy file
    # dummy_file.unlink() # Keep the file for inspection if needed
    print(f"\nDummy file '{dummy_file}' created for inspection.")
