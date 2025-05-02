# Obsidian AI Search

A tool to perform local semantic search on your Obsidian notes.

## Features

*   **Local First:** Indexes and searches your notes entirely on your machine.
*   **Semantic Search:** Finds notes based on meaning, not just keywords.
*   **Keyword Fallback:** Automatically uses keyword search if semantic results are sparse.
*   **Interactive REPL:** Provides a command-line interface for iterative searching.
*   **Clickable Links:** Results include `obsidian://` links to open notes directly.
*   **(Optional) RAG:** Can use the Gemini API (requires API key) to generate answers based on search results.
*   **Configuration:** Supports a TOML configuration file for defaults.

## Installation

This project uses Conda for environment management to ensure compatibility with native libraries like FAISS.

1.  **Install Conda:** If you don't have Conda, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda.
2.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/obsidian-ai-search.git
    cd obsidian-ai-search 
    ```
3.  **Create and Activate Conda Environment:** Create the environment and install dependencies from the provided `environment.yml` file:
    ```bash
    # Make sure you are not in any other conda/pip environment first (run `conda deactivate`)
    conda env create -f environment.yml
    conda activate obsidian-search
    ```
    Your terminal prompt should now show `(obsidian-search)`. All subsequent commands should be run within this activated environment.

4.  **(Optional) Environment Variables:** If you want to use the optional Gemini RAG feature:
    ```bash
    cp .env.sample .env
    # Edit .env and add your GEMINI_API_KEY
    ```

## Usage

Run commands using `python -m obsidian_ai_search <command>`.

**1. Indexing**

Create or update the search index for your vault:

```bash
python -m obsidian_ai_search index --vault-path "/path/to/your/obsidian/vault"
```

*   Use `--index-dir` to specify a custom location for index files (defaults to `.obsidian_ai_search_index` in the current directory or as set in config).
*   Use `--force-reindex` to rebuild the index from scratch.

**2. Searching (Single Shot)**

Perform a one-off search:

```bash
python -m obsidian_ai_search search "Your search query"
```

*   Use `--top-k <number>` to specify the number of results (defaults to 5 or config value).
*   Use `--index-dir` if your index is not in the default location.
*   Use `--gemini` to enable RAG answer generation (requires `GEMINI_API_KEY` environment variable to be set).

**3. Searching (Interactive REPL)**

Start an interactive session:

```bash
python -m obsidian_ai_search repl
```

*   Loads the index and model once for faster subsequent searches.
*   Options like `--index-dir`, `--top-k`, `--gemini` set initial defaults for the session.
*   **REPL Commands:**
    *   `/quit`: Exit the session.
    *   `/help`: Show available commands.
    *   `/gemini [on|off]`: Toggle Gemini RAG for subsequent queries.
    *   `/topk <number>`: Change the number of results to display.
    *   Any other input is treated as a search query.

## Configuration (Optional)

You can create a configuration file at `~/.config/obsidian-ai-search/config.toml` (or `$XDG_CONFIG_HOME/obsidian-ai-search/config.toml`) to set defaults:

```toml
# Example config.toml
index_dir = "/path/to/preferred/index/location"
default_top_k = 10
# embedding_model = "all-MiniLM-L6-v2" # Currently only supports default
```

Command-line options will override values set in the configuration file.

## Development

To set up the project for development:

1. Clone the repository
2. Create and activate the conda environment as described above
3. Install additional development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. Run tests:
   ```bash
   pytest
   ```

## Security Note

This tool processes your Obsidian notes locally. The optional Gemini RAG feature is the only component that sends data externally (to Google's API). This feature:
1. Is disabled by default
2. Requires you to explicitly provide an API key 
3. Only sends your search query and the retrieved context chunks

## License

MIT 