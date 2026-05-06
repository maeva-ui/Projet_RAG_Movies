"""
Configuration centralisée pour le projet RAG Films TMDB.
Toutes les constantes et configurations sont définies ici.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ==================== Chemins ====================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT

# Fichiers de données
DATASET_PATH = DATA_DIR / "tmdb_5000_movies.csv"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"
CHUNKS_META_PATH = DATA_DIR / "chunks_meta.pkl"
CONFIG_PATH = DATA_DIR / "embedding_config.json"
ENV_PATH = DATA_DIR / ".env"

# Charger les variables d'environnement
load_dotenv(ENV_PATH)

# ==================== Embedding ====================
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_DIMENSION = 768

# ==================== Chunking ====================
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ==================== FAISS ====================
FAISS_INDEX_TYPE = "IndexFlatIP"  # Inner Product = Cosine similarity après normalisation

# ==================== Groq / LLM ====================
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_MAX_TOKENS = 500
LLM_TEMPERATURE = 0.1

# ==================== Données TMDB ====================
LANGUAGE_FILTER = "en"
MIN_FILMS_REQUIRED = 500

# ==================== Streamlit ====================
STREAMLIT_PAGE_TITLE = "RAG Films TMDB"
STREAMLIT_LAYOUT = "wide"

# ==================== Validation ====================
REQUIRED_FILES = [
    FAISS_INDEX_PATH,
    CHUNKS_META_PATH,
    CONFIG_PATH,
]

REQUIRED_ENV_VARS = [
    "GROQ_API_KEY",
]


def get_config_dict() -> dict:
    """Retourner la configuration en dict pour sauvegarde JSON."""
    return {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "language_filter": LANGUAGE_FILTER,
        "llm_model": LLM_MODEL,
        "llm_temperature": LLM_TEMPERATURE,
    }


def validate_paths() -> list:
    """Vérifier que tous les chemins requis existent."""
    missing = [p for p in REQUIRED_FILES if not p.exists()]
    return missing


def validate_env() -> list[str]:
    """Vérifier que toutes les variables d'environnement requises sont définies."""
    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    return missing
