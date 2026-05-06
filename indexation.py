"""
Pipeline d'indexation pour RAG Films TMDB.
Charge le dataset TMDB, crée des chunks, génère des embeddings, et construit un index FAISS.
"""

import json
import pickle
from typing import Any

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import (
    DATASET_PATH, FAISS_INDEX_PATH, CHUNKS_META_PATH, CONFIG_PATH,
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
    LANGUAGE_FILTER, MIN_FILMS_REQUIRED,
    get_config_dict, validate_env
)


def load_tmdb_data() -> pd.DataFrame:
    """Charger et prétraiter le dataset TMDB."""
    df = pd.read_csv(DATASET_PATH)
    df = df[df["original_language"] == LANGUAGE_FILTER].copy()
    df = df.dropna(subset=["overview"])

    if len(df) < MIN_FILMS_REQUIRED:
        print(f"[WARNING] Seulement {len(df)} films (min {MIN_FILMS_REQUIRED} recommandé)")

    print(f"[OK] {len(df)} films chargés du dataset TMDB")
    return df


def format_film_text(row: pd.Series) -> str:
    """Formater les données du film en texte lisible."""
    try:
        genres = json.loads(row["genres"])
        genre_names = ", ".join([g["name"] for g in genres])
    except (json.JSONDecodeError, KeyError, TypeError):
        genre_names = "Unknown"

    return f"""Titre: {row['title']}
Genres: {genre_names}
Note: {row['vote_average']}/10
Synopsis: {row['overview']}"""


def chunker(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Diviser le texte en chunks chevauchants."""
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) + 2 <= chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def prepare_chunks(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convertir les films en chunks avec métadonnées."""
    chunks_with_meta = []

    for idx, row in df.iterrows():
        film_text = format_film_text(row)
        film_chunks = chunker(film_text)

        for chunk_idx, chunk in enumerate(film_chunks):
            chunks_with_meta.append({
                "content": chunk,
                "film_id": row["id"],
                "title": row["title"],
                "rating": row["vote_average"],
                "chunk_idx": chunk_idx
            })

    print(f"[OK] {len(chunks_with_meta)} chunks créés à partir de {len(df)} films")
    return chunks_with_meta


def embed_chunks(chunks_with_meta: list[dict[str, Any]]) -> tuple[np.ndarray, SentenceTransformer]:
    """Générer les embeddings pour tous les chunks."""
    print("[...] Chargement du modèle d'embedding...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [chunk["content"] for chunk in chunks_with_meta]
    print(f"[...] Embedding de {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)

    embeddings = embeddings.astype(np.float32)
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)

    print(f"[OK] Shape des embeddings: {embeddings.shape}")
    return embeddings, model


def create_faiss_index(embeddings: np.ndarray) -> Any:
    """Créer l'index FAISS avec similarité cosinus."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"[OK] Index FAISS créé avec {index.ntotal} vecteurs")
    return index


def save_index(index: Any, chunks_with_meta: list[dict[str, Any]]) -> None:
    """Sauvegarder l'index FAISS, métadonnées, et config."""
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(CHUNKS_META_PATH, "wb") as f:
        pickle.dump(chunks_with_meta, f)

    with open(CONFIG_PATH, "w") as f:
        json.dump(get_config_dict(), f, indent=2)

    print(f"[OK] Index sauvegardé dans {FAISS_INDEX_PATH}")
    print(f"[OK] Métadonnées sauvegardées dans {CHUNKS_META_PATH}")
    print(f"[OK] Config sauvegardée dans {CONFIG_PATH}")


def main() -> None:
    """Pipeline d'indexation complet."""
    print("=" * 60)
    print("Pipeline d'indexation RAG - Films TMDB 5000")
    print("=" * 60)

    missing_env = validate_env()
    if missing_env:
        print(f"[ERROR] Variables d'environnement manquantes: {missing_env}")
        return

    df = load_tmdb_data()
    chunks_with_meta = prepare_chunks(df)
    embeddings, model = embed_chunks(chunks_with_meta)
    index = create_faiss_index(embeddings)
    save_index(index, chunks_with_meta)

    print("\n" + "=" * 60)
    print("Indexation terminée avec succès !")
    print("=" * 60)
    print(f"Films: {len(df)} | Chunks: {len(chunks_with_meta)} | Dimension: {embeddings.shape[1]}")


if __name__ == "__main__":
    main()
