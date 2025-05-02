# Testing Guide: Obsidian AI Search

This guide provides step-by-step instructions to test the core indexing and search functionality implemented so far (Phases 1 & 2).

## Prerequisites

1.  **Python:** Ensure you have Python 3.8+ installed.
2.  **Virtual Environment:** It's highly recommended to use a virtual environment. Activate yours if you have one.
3.  **Dependencies:** Install the required packages:
    ```bash
    pip install -r requirements.txt 
    # Or: pip install click rich watchdog sentence-transformers faiss-cpu numpy
    ```
4.  **Project Root:** Run all commands from the root directory of the `Obsidian-AI-Search` project.

## Part 1: Setting up a Test Vault

1.  **Create Test Vault Directory:**
    ```bash
    mkdir test_vault
    ```
2.  **Create Sample Notes:**
    *   `test_vault/note1.md`:
        ```markdown
        # Note One

        This is the first sample note. It talks about apples and oranges.
        Fruit is healthy.
        ```
    *   `test_vault/note2.md`:
        ```markdown
        # Note Two: Bananas

        Bananas are yellow and curved. They are a popular fruit.
        This note focuses on bananas specifically.
        ```
    *   `test_vault/subdir/`:
        ```bash
        mkdir test_vault/subdir
        ```
    *   `test_vault/subdir/note3.md`:
        ```markdown
        # Note Three - Inside Subdirectory

        This note is nested. It discusses vegetables like carrots and broccoli.
        ```

## Part 2: Testing Indexing (`index` command)

1.  **Initial Indexing:**
    *   **Command:**
        ```bash
        python -m obsidian_ai_search.cli index --vault-path ./test_vault
        ```
    *   **Expected Output:**
        *   Log messages indicating the start of indexing, vault path, and index directory (`.obsidian_ai_search_index` by default).
        *   Messages about loading the embedding model.
        *   Progress bar for embedding generation (may be quick for few notes).
        *   Messages about adding vectors to FAISS.
        *   Messages about saving the index and state.
        *   Final confirmation message like ":white_check_mark: [bold green]Indexing complete.[/bold green] Index contains X vectors." (X should be > 0, likely around 3-6 depending on chunking).
    *   **Verification:**
        *   Check if the `.obsidian_ai_search_index` directory exists.
        *   Inside `.obsidian_ai_search_index`, check for `index.faiss` and `index_metadata.json`, and `indexing_state.json`.

2.  **Indexing Non-existent Vault:**
    *   **Command:**
        ```bash
        python -m obsidian_ai_search.cli index --vault-path ./non_existent_vault
        ```
    *   **Expected Output:** An error message from Click indicating the path does not exist.

3.  **Incremental Indexing (Modification):**
    *   **Modify a file:** Change `test_vault/note1.md` to:
        ```markdown
        # Note One - Updated

        This is the first sample note. It talks about apples and oranges.
        Fruit is very healthy and delicious. This line was added.
        ```
    *   **Run index again:**
        ```bash
        python -m obsidian_ai_search.cli index --vault-path ./test_vault
        ```
    *   **Expected Output:**
        *   Messages indicating loading existing state and index.
        *   Message like "File modified: note1.md".
        *   Messages about removing old chunks and adding new ones for `note1.md`.
        *   Final indexing complete message. Vector count might change slightly.
    *   **Verification:** Check the timestamp of `index.faiss` and `indexing_state.json` - they should be updated.

4.  **Incremental Indexing (Addition):**
    *   **Add a file:** Create `test_vault/note4.md`:
        ```markdown
        # Note Four

        A completely new note about pears.
        ```
    *   **Run index again:**
        ```bash
        python -m obsidian_ai_search.cli index --vault-path ./test_vault
        ```
    *   **Expected Output:**
        *   Message like "New file found: note4.md".
        *   Messages about adding new chunks for `note4.md`.
        *   Final indexing complete message. Vector count should increase.
    *   **Verification:** Check timestamps and index file size (it might increase).

5.  **Incremental Indexing (Deletion):**
    *   **Delete a file:**
        ```bash
        rm test_vault/subdir/note3.md
        ```
    *   **Run index again:**
        ```bash
        python -m obsidian_ai_search.cli index --vault-path ./test_vault
        ```
    *   **Expected Output:**
        *   Message like "Found 1 files to remove from index: ['note3.md']".
        *   Message about removing chunks for the deleted file.
        *   Final indexing complete message. Vector count should decrease.
    *   **Verification:** Check timestamps.

6.  **Force Re-indexing:**
    *   **Command:**
        ```bash
        python -m obsidian_ai_search.cli index --vault-path ./test_vault --force-reindex
        ```
    *   **Expected Output:**
        *   Warning message ":warning: [bold yellow]Forcing re-index.[/bold yellow]".
        *   Messages indicating processing *all* found files, not just changed ones.
        *   Final indexing complete message.
    *   **Verification:** Check timestamps.

## Part 3: Testing Search (`search` command)

**Note:** Ensure you have run the `index` command successfully at least once before running search tests.

1.  **Basic Search:**
    *   **Command:**
        ```bash
        python -m obsidian_ai_search.cli search "healthy fruit"
        ```
    *   **Expected Output:**
        *   Messages about loading the index and embedding the query.
        *   A message like ":page_facing_up: [bold green]Found X results:[/bold green]" (where X is up to 5 by default).
        *   Formatted results displayed in Panels, each including:
            *   Title (e.g., "Result 1: Note One - Updated (Score: 0.XXX)")
            *   File Path (e.g., `test_vault/note1.md`)
            *   Heading (if applicable)
            *   Content Snippet
            *   Similarity Score (lower is better for L2 distance)
        *   Results related to "apples", "oranges", "bananas" are likely candidates.

2.  **Search with Different `top-k`:**
    *   **Command:**
        ```bash
        python -m obsidian_ai_search.cli search "fruit" --top-k 2
        ```
    *   **Expected Output:** Similar to above, but should show at most 2 results.

3.  **Search Before Indexing:**
    *   **Setup:** Delete the index directory: `rm -rf .obsidian_ai_search_index`
    *   **Command:**
        ```bash
        python -m obsidian_ai_search.cli search "anything"
        ```
    *   **Expected Output:** Click error about the `--index-dir` path not existing, or if the directory exists but is empty, an error message like ":x: [bold red]Error:[/bold red] Index file not found or invalid...".

4.  **Search Empty Index:**
    *   **Setup:** Create an empty index directory (`mkdir .obsidian_ai_search_index`) or run index on an empty vault.
    *   **Command:** (Assuming index was run on empty vault or index files are missing)
        ```bash
        # First ensure index exists but is empty/invalid
        # mkdir .obsidian_ai_search_index 
        # OR run index on empty vault
        python -m obsidian_ai_search.cli index --vault-path ./empty_vault --index-dir .obsidian_ai_search_index
        
        # Then search
        python -m obsidian_ai_search.cli search "anything"
        ```
    *   **Expected Output:** Depending on exact state, either the "Index file not found or invalid" error or ":warning: [yellow]Index is empty.[/yellow]".

5.  **Search with No Results:**
    *   **Command:** (Use a query unlikely to match anything)
        ```bash
        python -m obsidian_ai_search.cli search "quantum entanglement xylophone"
        ```
    *   **Expected Output:** Message like ":mag_right: [yellow]No results found for your query.[/yellow]".

---

Run through these steps to ensure the core functionality is working as expected. Report any discrepancies or errors. 