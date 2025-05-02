## Obsidian AI Search — PRD (v1.1)

**Owner:** Krish Shah  
**Date:** 2025-05-01

---

### 1. Objectives  
- **Semantic search** over Obsidian notes (local, privacy-first)  
- **Auto-sync** on note add/modify/delete  
- **Optional RAG** via Gemini API for answer generation  
- **Clear, helpful UX** with fallbacks

---

### 2. Core Features

1. **Ingestion & Sync**  
   - Scan vault for `.md` files on startup and watch for changes (via `watchdog`)  
   - Chunk by **paragraphs and headings** (whichever yields ~200–300-word chunks)  
   - Track file hashes to detect additions/edits/removals  

2. **Embedding & Indexing**  
   - Embed chunks with `sentence-transformers` (MiniLM)  
   - Store vectors in FAISS (`index.faiss`)  
   - Rebuild or update only affected vectors on change  

3. **Search API/CLI**  
   - Query → embed → FAISS top-K (default K=5)  
   - Return for each hit:  
     - **Note title** (filename or first H1)  
     - **Chunk excerpt** (with “…” context if truncated)  
     - **Similarity score**  

4. **UX Enhancements**  
   - **Semantic + fallback keyword search** if fewer than 3 semantic hits  
   - **Query suggestions** when no hits (“Did you mean…?”)  
   - **Pagination** or “more results” prompt for large vaults  
   - **Syntax-highlighting**/markdown preview of returned chunks  

5. **Optional RAG Completion**  
   - CLI flag `--gemini` (off by default)  
   - Send top-N chunks + user query to Gemini API  
   - Display Gemini’s answer beneath raw hits  

---

### 3. Technical Stack

| Layer           | Technology              | Notes                             |
|-----------------|-------------------------|-----------------------------------|
| File watcher    | `watchdog`              | real-time sync                    |
| Chunker         | built-in parser         | paragraph + heading logic         |
| Embeddings      | `sentence-transformers` | MiniLM for CPU-friendly speed     |
| Vector DB       | `faiss-cpu`             | quick similarity search           |
| CLI / API       | Python + Click / FastAPI| lightweight interface             |
| RAG             | Gemini 1.5 Flash API    | optional via CLI flag             |

---

### 4. Performance Targets

| Metric               | Target                           |
|----------------------|----------------------------------|
| Initial index time   | <2 min for 500 notes             |
| Incremental update   | <5 sec per changed file          |
| Search latency       | <200 ms for K=5 (semantic only)  |
| Gemini response time | <5 sec (network-dependent)       |

---

### 5. Open-Ended UX Ideas

- **Interactive REPL** with up/down arrow to revisit past queries  
- **Configurable result count** (`--top 5/10/…`)  
- **Markdown-rich output**: clickable links to vault files  
- **Basic stats**: “Indexed X notes, Y chunks” on startup  

---
