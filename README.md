# Obsidian AI Search & Chat

Tired of losing notes in the depths of your Obsidian vault? Wish you could just *ask* your notes questions and find connections you missed?

Obsidian AI Search uses semantic search and optional Retrieval-Augmented Generation (RAG) with Google Gemini to help you rediscover and engage with your knowledge base. It's 100% local-first for privacy, open-source, and designed to make revisiting your notes intuitive.

**Link:** [https://github.com/kshah00/Obsidian-Chat](https://github.com/kshah00/Obsidian-Chat)

## Features

*   **🧠 Semantic Search:** Find notes based on meaning and context, not just exact keywords. Resurface old ideas and related concepts effortlessly.
*   **🔐 Local First & Private:** Your notes are indexed and searched *entirely on your machine*. No data leaves your system unless you explicitly enable the cloud features.
*   **🤖 Optional RAG with Gemini:** Ask questions in natural language! The tool can retrieve relevant notes and use Google's Gemini API (free tier available) to synthesize answers based *only* on your content. (Requires API key).
*   **✨ Web Interface (Streamlit):** An easy-to-use web UI for searching, viewing results, and interacting with the RAG feature.
*   **💻 CLI Interface:** Includes an interactive REPL and single-shot search commands for terminal users.
*   **🔗 Clickable Obsidian Links:** Results include `obsidian://` links to jump directly to the note in your vault.
*   **⚙️ Configurable:** Set defaults via a configuration file.
*   **📂 Open Source:** Built for the community. Contributions welcome!

## Installation

This project uses Conda for environment management to ensure compatibility with native libraries like FAISS.

1.  **Install Conda:** If you don't have Conda, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda.
2.  **Clone the repository:**
    ```bash
    git clone https://github.com/kshah00/Obsidian-Chat.git
    cd Obsidian-Chat
    ```
3.  **Create and Activate Conda Environment:** Create the environment and install dependencies from the provided `environment.yml` file:
    ```bash
    # Make sure you are not in any other conda/pip environment first (run `conda deactivate`)
    conda env create -f environment.yml
    conda activate obsidian-search # The environment name is defined in environment.yml
    ```
    Your terminal prompt should now show `(obsidian-search)`. All subsequent commands should be run within this activated environment.

4.  **(Optional) Gemini API Key:** If you want to use the RAG feature:
    *   Copy the sample environment file:
        ```bash
        cp .env.sample .env
        ```
    *   Edit the `.env` file and add your Google AI Studio API key:
        ```
        GEMINI_API_KEY=your_google_ai_api_key_here
        ```
        You can get a key from [Google AI Studio](https://aistudio.google.com/app/apikey). The Gemini Pro model used has a generous free tier.

## Usage

You can interact with the tool via the Web Interface or the Command-Line Interface. **You must index your vault first.**

**1. Indexing (Required First Step)**

Create or update the search index for your vault. Run this from the project directory (`Obsidian-Chat`):

```bash
python -m obsidian_ai_search index --vault-path "/path/to/your/obsidian/vault"
```

*   Use `--index-dir` to specify a custom location for index files (defaults to `.obsidian_ai_search_index` in the current directory or as set in config). The Web UI and CLI need access to this directory.
*   Use `--force-reindex` to rebuild the index from scratch.

**2. Web Interface (Streamlit)**

The easiest way to search and use RAG.

*   **Launch:** Make sure your `obsidian-search` conda environment is active, then run:
    ```bash
    streamlit run app.py
    ```
*   **Features:**
    *   Enter your query in the main search bar.
    *   Use the sidebar to configure the index directory path and the number of results (K).
    *   Toggle "Enable Gemini RAG" in the sidebar to turn the question-answering feature on/off (requires API key in `.env`).
    *   Results are displayed with content snippets and clickable Obsidian links.
    *   If RAG is enabled, a synthesized answer from Gemini (based *only* on your notes) appears above the source results.

**3. Command-Line Interface (CLI)**

For terminal users.

*   **Single Search:**
    ```bash
    python -m obsidian_ai_search search "Your search query" [--top-k 5] [--index-dir path/to/index] [--gemini]
    ```
*   **Interactive REPL:**
    ```bash
    python -m obsidian_ai_search repl [--top-k 5] [--index-dir path/to/index] [--gemini]
    ```
    *   Enter queries directly.
    *   Use commands like `/gemini on|off`, `/topk <number>`, `/quit`.

## Configuration (Optional)

Create a configuration file at `~/.config/obsidian-ai-search/config.toml` (or `$XDG_CONFIG_HOME/obsidian-ai-search/config.toml`) to set defaults:

```toml
# Example config.toml
index_dir = "/path/to/shared/index/location"
default_top_k = 10
# embedding_model = "all-MiniLM-L6-v2" # Currently only supports default
```
Command-line options and Web UI settings override values in the configuration file.

## Development

Let's build this together!

1.  Follow the Installation steps 1-3.
2.  Install development dependencies:
    ```bash
    pip install -e ".[dev]"
    ```
3.  Run tests:
    ```bash
    pytest
    ```
4.  See `CONTRIBUTING.md` for more guidelines.

## Security & Privacy

*   **Core Functionality:** Indexing and searching happen **entirely locally**. Your notes stay on your machine.
*   **Optional Gemini RAG:** This is the *only* feature that sends data externally (to Google's API).
    *   It's **disabled by default**.
    *   Requires you to explicitly provide your own API key via the `.env` file.
    *   Only sends your search query and the text snippets from the *locally retrieved* top search results to Google Gemini to generate an answer. Your full notes are **not** sent.

## License

MIT 