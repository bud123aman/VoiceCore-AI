"""
Q2 — Knowledge Base Ingestion (Rewritten)

Key fixes over v1:
1. No chunking — all records are <100 words, chunking only adds noise
2. Enriched embedding text: title + category + content concatenated
3. BM25 tokenization: lowercase + hyphen normalization + punctuation stripping
4. Consistent preprocessing applied to both index and query at search time

Usage:
    python ingest.py
"""

import os
import pickle
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from kb_data import KB_RECORDS

load_dotenv(Path(__file__).parent.parent / ".env")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "health_insurance_kb")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_DIM = 384
BM25_INDEX_PATH = Path(__file__).parent / "bm25_index.pkl"


def normalize_text(text: str) -> str:
    """
    Normalize text for BM25 tokenization.
    - Lowercase
    - Replace hyphens with space (pre-existing → pre existing)
    - Strip punctuation except apostrophes
    - Collapse whitespace
    Applied identically to both documents and queries.
    """
    text = text.lower()
    text = text.replace("-", " ")  
    text = re.sub(r"[^\w\s']", " ", text) 
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Normalize then split into tokens."""
    return normalize_text(text).split()


def build_embedding_text(record: dict) -> str:
    """
    Build enriched text for embedding.
    Includes title + category + content so vector search uses full context.
    Without this, embedding only sees content and misses the title's signal.
    """
    return f"{record['title']}. Category: {record['category']}. {record['content']}"


def build_points(records: list[dict], model: SentenceTransformer) -> list[PointStruct]:
    """
    One record → one point. No chunking.
    All records are 50–85 words — well within embedding model context window.
    Chunking 50-word documents into 180-word chunks is a no-op that wastes
    index space and confuses BM25 IDF weighting.
    """
    points = []
    for record in records:
        embed_text = build_embedding_text(record)
        embedding = model.encode(embed_text, normalize_embeddings=True).tolist()

        payload = {
            "record_id": record["record_id"],
            "title": record["title"],
            "content": record["content"],
            "embed_text": embed_text,  
            "category": record["category"],
            "source": record["source"],
            "version": record["version"],
            "pii": record["pii"],
        }

        point_id = len(points)
        points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

    return points


def build_bm25_index(points: list[PointStruct]) -> BM25Okapi:
    """
    Build BM25 index using normalized embed_text (title + category + content).
    Using normalized tokenization prevents hyphen/punctuation mismatches.
    """
    corpus = [tokenize(p.payload["embed_text"]) for p in points]
    return BM25Okapi(corpus)


def main():
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Building {len(KB_RECORDS)} records (no chunking)...")
    points = build_points(KB_RECORDS, model)
    print(f"  → {len(points)} points (1 per record)")

    print(f"\nConnecting to Qdrant at {QDRANT_URL}")
    client = (
        QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        if QDRANT_API_KEY
        else QdrantClient(url=QDRANT_URL)
    )

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        print(f"  Deleting existing collection: {COLLECTION}")
        client.delete_collection(COLLECTION)

    print(f"  Creating collection: {COLLECTION}")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )

    client.create_payload_index(COLLECTION, "category", PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION, "pii", PayloadSchemaType.BOOL)
    client.create_payload_index(COLLECTION, "record_id", PayloadSchemaType.KEYWORD)

    print(f"  Upserting {len(points)} points...")
    batch_size = 50
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=COLLECTION, points=points[i : i + batch_size])
    print(f"  ✓ Qdrant upsert complete")

    print(f"\nBuilding BM25 index...")
    bm25 = build_bm25_index(points)
    bm25_data = {
        "index": bm25,
        "point_ids": [p.id for p in points],
        "payloads": [p.payload for p in points],
    }
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_data, f)
    print(f"  ✓ BM25 index saved — {len(points)} documents")

    count = client.count(collection_name=COLLECTION).count
    print(f"\n✓ Ingestion complete. {count} vectors in '{COLLECTION}'")

    test_query = "What is the waiting period for pre-existing diseases?"
    test_vec = model.encode(
        f"Query: {test_query}", normalize_embeddings=True
    ).tolist()
    results = client.search(COLLECTION, query_vector=test_vec, limit=3)
    print(f"\nSmoke test — query: '{test_query}'")
    for r in results:
        print(f"  [{r.score:.3f}] {r.payload['record_id']} — {r.payload['title']}")


if __name__ == "__main__":
    main()
