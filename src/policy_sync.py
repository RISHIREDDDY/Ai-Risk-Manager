"""
policy_sync.py — Dynamic Ingestion & Pinecone Vector Upsert Pipeline

Reads policy sources from policy_sources.json,
generates vector embeddings via Pinecone's native multilingual-e5-large inference,
and indexes them into the Pinecone Vector Database.
"""

import os
import sys
import json
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "chargeback-policies")
EMBEDDING_MODEL = "multilingual-e5-large"
EMBEDDING_DIM = 1024

def sync_policies():
    print("=" * 60)
    print("[INIT] Initializing Dynamic Policy Ingestion to Pinecone...")
    print("=" * 60)

    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not set in .env")

    # 1. Initialize Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # 2. Re-create index if dimension mismatch (dim=1024 for multilingual-e5-large)
    existing_indexes = {idx.name: idx.dimension for idx in pc.list_indexes()}
    print(f"[PINECONE] Connected. Existing Indexes: {existing_indexes}")

    if PINECONE_INDEX_NAME in existing_indexes:
        if existing_indexes[PINECONE_INDEX_NAME] != EMBEDDING_DIM:
            print(f"[PINECONE] Re-creating index '{PINECONE_INDEX_NAME}' with correct dimension {EMBEDDING_DIM}...")
            pc.delete_index(PINECONE_INDEX_NAME)
            time.sleep(3)
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            time.sleep(5)
    else:
        print(f"[PINECONE] Creating Serverless Index '{PINECONE_INDEX_NAME}' (dim={EMBEDDING_DIM}, metric=cosine)...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(5)

    index = pc.Index(PINECONE_INDEX_NAME)

    # 3. Load policy sources
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sources_path = os.path.join(root_dir, "data", "policy_sources.json")
    if not os.path.exists(sources_path):
        sources_path = os.path.join(os.path.dirname(__file__), "policy_sources.json")
    with open(sources_path, "r", encoding="utf-8") as f:
        policy_sources = json.load(f)

    print(f"[DOCS] Found {len(policy_sources)} authoritative policy documents to index.")

    # 4. Generate batch embeddings via Pinecone inference
    texts = [
        f"passage: {p['name']} ({p['issuer']}) - Category: {p['category']}\n{p['text']}"
        for p in policy_sources
    ]
    print(f"[EMBED] Generating embeddings for {len(texts)} policies with model '{EMBEDDING_MODEL}'...")
    embeddings = pc.inference.embed(
        model=EMBEDDING_MODEL,
        inputs=texts,
        parameters={"input_type": "passage", "truncate": "END"}
    )

    vectors_to_upsert = []
    for i, policy in enumerate(policy_sources):
        vectors_to_upsert.append({
            "id": policy["id"],
            "values": embeddings[i].values,
            "metadata": {
                "name": policy["name"],
                "issuer": policy["issuer"],
                "category": policy["category"],
                "source_url": policy["source_url"],
                "reason_codes": ",".join(policy["reason_codes"]),
                "text": policy["text"]
            }
        })

    # 5. Upsert to Pinecone
    print(f"\n[UPSERT] Upserting {len(vectors_to_upsert)} vector records to Pinecone...")
    upsert_res = index.upsert(vectors=vectors_to_upsert)
    print(f"[SUCCESS] Upsert complete! Upserted count: {upsert_res.upserted_count}")

    # 6. Verify index stats
    time.sleep(2)
    stats = index.describe_index_stats()
    print(f"[STATS] Current Index Stats: Total Vector Count = {stats.total_vector_count}")
    print("=" * 60)
    print("[DONE] Policy Knowledge Base successfully synced with Pinecone!")
    print("=" * 60)
    return stats

if __name__ == "__main__":
    sync_policies()
