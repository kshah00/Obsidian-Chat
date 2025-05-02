# Obsidian AI Search - Project Plan & Tasks

This document outlines the development phases and key tasks for the Obsidian AI Search project, based on `prd.md`.

## UI/UX Principles

*   **Clarity:** Output should be easy to understand. Use clear labels, minimal jargon, and logical formatting.
*   **Efficiency:** Interactions should be quick and require minimal user effort. Prioritize speed and sensible defaults.
*   **Feedback:** Keep the user informed about what the tool is doing (e.g., indexing progress, search status, errors).
*   **Consistency:** Maintain consistent command structures, output formats, and terminology.
*   **Helpfulness:** Provide useful error messages and guidance (e.g., suggestions, help commands).
*   **Discoverability:** Make features easy to find and use, potentially through clear help text and intuitive flags/commands.
*   **Privacy-First:** Reinforce that processing happens locally unless explicitly stated (like the optional Gemini feature).

## Phase 1: Core Ingestion and Indexing

*   **Objective:** Set up the foundation for scanning, chunking, embedding, and indexing Obsidian notes locally.
*   **Key Tasks:**
    *   [x] **Project Setup:** Initialize Python project structure, virtual environment, and install base dependencies (`watchdog`, `sentence-transformers`, `faiss-cpu`).
    *   [x] **File Discovery:** Implement logic to scan the specified Obsidian vault for `.md` files on startup.
    *   [x] **File Watching:** Integrate `watchdog` to monitor the vault for file creation, modification, and deletion events.
    *   [x] **Chunking Logic:** Develop a parser to chunk `.md` files based on paragraphs and headings, aiming for ~200-300 word chunks.
    *   [x] **Change Detection:** Implement file hashing (e.g., SHA-256) to track content changes efficiently.
    *   [x] **Embedding Integration:** Load the `sentence-transformers` (MiniLM) model and implement a function to embed text chunks into vectors.
    *   [x] **FAISS Indexing:** Set up `faiss-cpu` to create, save (`index.faiss`), and load a vector index.
    *   [x] **Indexing Pipeline:** Combine discovery, chunking, embedding, and indexing into a pipeline that runs on startup.
    *   [x] **Incremental Updates:** Implement logic to efficiently update the FAISS index based on detected file changes (add, edit, delete) without full re-indexing.
    *   [x] **Basic CLI Stub:** Create a minimal CLI entry point using `Click` for future commands.

## Phase 2: Search Functionality (CLI/API)

*   **Objective:** Implement the core semantic search feature accessible via a CLI.
*   **Key Tasks:**
    *   [x] **CLI Search Command:** Add a `search` command to the Click CLI.
    *   [x] **Query Embedding:** Reuse the embedding function to convert the user's search query into a vector.
    *   [x] **FAISS Search:** Implement the logic to perform a similarity search on the loaded FAISS index using the query vector and retrieve top-K results (default K=5).
    *   [x] **Result Formatting:** Process the FAISS results to extract and format the required information: Note title (filename or H1), chunk excerpt (with context markers like "..."), and similarity score.
    *   [x] **CLI Output:** Display the formatted search results clearly in the terminal.
    *   [x] **(Optional) API Endpoint:** Create a basic FastAPI application with a `/search` endpoint that mirrors the CLI search functionality.

## Phase 3: UX Enhancements and Optional RAG

*   **Objective:** Improve the user experience of the search and add optional RAG capabilities.
*   **Key Tasks:**
    *   [x] **Fallback Keyword Search:** Implement a simple keyword search mechanism to be used if semantic search yields fewer than 3 results.
    *   [x] **Query Suggestions:** If no results are found, implement a basic "Did you mean...?" suggestion mechanism (e.g., based on edit distance or simpler semantic alternatives if feasible).
    *   [x] **Pagination/More Results:** Add logic to handle large numbers of results, either through pagination or a "show more" prompt.
    *   [x] **Result Presentation:** Enhance CLI output with syntax highlighting or markdown rendering for the retrieved chunks.
    *   [x] **RAG CLI Flag:** Add a `--gemini` flag (default: False) to the `search` command.
    *   [x] **Gemini API Integration:** If the flag is set, implement the logic to send the top-N search results (chunks) and the original query to the Gemini API.
    *   [x] **Display RAG Output:** Display the generated answer from Gemini below the standard search results.
    *   [x] **API Key Management:** Implement secure handling for the Gemini API key (e.g., environment variable).

## Phase 4: Refinement, Performance, and Open-Ended Features

*   **Objective:** Optimize performance, add requested open-ended features, and prepare for release.
*   **Key Tasks:**
    *   [x] **Performance Tuning:** Profile and optimize initial indexing time, incremental updates, and search latency based on PRD targets.
    *   [x] **Interactive REPL:** Implement an interactive loop (REPL) for the CLI, potentially with history (up/down arrows).
    *   [x] **Configurable Result Count:** Add a `--top N` option to the search command.
    *   [x] **Markdown Output Links:** Enhance markdown output to include clickable `obsidian://` links to the source notes/files.
    *   [x] **Startup Stats:** Display basic statistics (number of notes/chunks indexed) when the tool starts.
    *   [ ] **Configuration File:** Consider adding a simple configuration file (e.g., `