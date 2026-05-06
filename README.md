# 🎬 RAG Films TMDB

A complete Retrieval-Augmented Generation (RAG) system for film recommendations using the TMDB 5000 dataset.

**Technology Stack:**
- **LLM:** Groq (llama-3.3-70b-versatile)
- **Embeddings:** SentenceTransformers (paraphrase-multilingual-mpnet-base-v2)
- **Vector Database:** FAISS (IndexFlatIP)
- **Interface:** Streamlit
- **No LangChain/LlamaIndex** (pure implementation)

---

## 📋 Setup

### 1. Download Dataset

Download TMDB 5000 from [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata):
- File: `tmdb_5000_movies.csv`
- Place in this directory

### 2. Environment Setup

Create a `.env` file in this directory:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Step 1: Index the Dataset

```bash
python indexation.py
```

This will:
- Load 4504 English films from TMDB
- Chunk each film (title, genres, rating, overview)
- Generate embeddings using sentence-transformers
- Create a FAISS index
- Save `faiss_index.bin` and `chunks_meta.pkl`

**Output:** 5387 chunks ready for retrieval

### Step 2: Run the RAG Interface

```bash
streamlit run rag.py
```

Then open your browser to `http://localhost:8501`

---

## 💡 Features

**Interactive Streamlit App:**
- 🔍 Ask questions about films
- ⚙️ Adjust number of results (k: 1-10)
- 📽️ View relevant film sources with full context
- 💬 Get AI-powered recommendations

**Example Queries:**
- "Recommend me a sci-fi film with a high rating"
- "What action films are available?"
- "Show me films directed by Christopher Nolan"
- "Find romantic comedies with ratings above 8"

---

## 🔧 How It Works

### Indexation Pipeline

1. **Load Data:** Read TMDB CSV, filter by English language
2. **Format:** Create structured text (title + genres + rating + overview)
3. **Chunk:** Split into overlapping chunks (size: 500 chars, overlap: 50)
4. **Embed:** Generate vectors using SentenceTransformer
5. **Index:** Build FAISS index with cosine similarity (IP)
6. **Persist:** Save index and metadata for reuse

### Q&A Pipeline

1. **Query Embedding:** Convert question to vector
2. **Search:** Find k nearest neighbors in FAISS
3. **Build Prompt:** Combine question + relevant film chunks
4. **Generate:** Use Groq to produce natural response
5. **Display:** Show answer + source films

---

## 📊 Configuration

**Chunking Strategy:**
- Size: 500 characters per chunk
- Overlap: 50 characters
- Method: RecursiveCharacterTextSplitter (sentence-aware)

**Embedding Model:**
- `paraphrase-multilingual-mpnet-base-v2`
- Dimension: 768
- L2 normalized before FAISS indexing

**FAISS Index:**
- Type: IndexFlatIP (inner product = cosine after normalization)
- Fast exact search, suitable for ~500 films

**LLM Settings:**
- Model: `llama-3.3-70b-versatile` (Groq)
- Temperature: 0.1 (low, focused responses)
- Max tokens: 500

---

## 📁 Files

- `config.py` - Centralized configuration
- `requirements.txt` - Python dependencies
- `indexation.py` - Dataset indexing pipeline
- `rag.py` - Streamlit Q&A interface
- `faiss_index.bin` - FAISS index (generated)
- `chunks_meta.pkl` - Chunk metadata (generated)
- `embedding_config.json` - Embedding model config (generated)
- `.env` - API keys (not committed)

---

## ⚡ Performance

- **Indexation Time:** ~5-10 min (embedding + FAISS)
- **Query Time:** <1 sec (search + generation)
- **Memory:** ~500 MB (index + model)

---

## 🔄 Troubleshooting

**Issue:** "Module not found" error
- **Solution:** `pip install -r requirements.txt`

**Issue:** Groq API key error
- **Solution:** Check `.env` file has `GROQ_API_KEY=...`

**Issue:** Index files not found
- **Solution:** Run `python indexation.py` first

**Issue:** Dataset file not found
- **Solution:** Download `tmdb_5000_movies.csv` from Kaggle and place in this directory

---

## 📝 Notes

- Minimum 500 films required for good results
- Only English-language films are indexed
- Each film is split into multiple chunks for better retrieval
- Temperature is kept low (0.1) to prevent hallucinations

---

**Author:** Claude Code  
**TP Reference:** TP_RAG.pdf (Sujet A - Films)
