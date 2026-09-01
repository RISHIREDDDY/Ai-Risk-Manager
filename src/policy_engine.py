"""
policy_engine.py — Dynamic Retrieval Layer for Chargeback Policies via Pinecone RAG

Queries the Pinecone vector index for official card network rules (Visa/Mastercard),
regulatory directives (RBI 2FA), and merchant store terms matching the dispute reason.
"""

import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "chargeback-policies")
EMBEDDING_MODEL = "multilingual-e5-large"

_pinecone_client = None
_pinecone_index = None
_policy_cache = {}

def get_pinecone_index():
    """Lazily initializes Pinecone index."""
    global _pinecone_client, _pinecone_index
    if _pinecone_index is None and PINECONE_API_KEY:
        try:
            from pinecone import Pinecone
            _pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
            _pinecone_index = _pinecone_client.Index(PINECONE_INDEX_NAME)
        except Exception as e:
            print(f"[WARN] Failed to connect to Pinecone index: {e}")
            _pinecone_index = None
    return _pinecone_client, _pinecone_index

def retrieve_policy_context(dispute_reason: str, payment_method: str = "Credit Card", top_k: int = 2) -> list[dict]:
    """
    Retrieves the top-k most relevant official policy clauses and live URLs
    from Pinecone Vector DB based on dispute reason and payment rail.
    """
    cache_key = (dispute_reason, payment_method, top_k)
    if cache_key in _policy_cache:
        return _policy_cache[cache_key]

    pc, index = get_pinecone_index()
    query_text = f"query: Dispute reason: {dispute_reason}, Payment method: {payment_method}, Compelling evidence defense guidelines"

    # Try Pinecone semantic search
    if pc and index:
        try:
            embed_res = pc.inference.embed(
                model=EMBEDDING_MODEL,
                inputs=[query_text],
                parameters={"input_type": "query", "truncate": "END"}
            )
            raw_values = embed_res[0].values
            query_vector: list[float] = list(raw_values() if callable(raw_values) else raw_values)
            
            search_results = index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )
            
            policies = []
            for match in search_results.matches:
                metadata = match.metadata or {}
                policies.append({
                    "id": match.id,
                    "score": round(match.score, 4),
                    "name": metadata.get("name", "Card Network Rule"),
                    "issuer": metadata.get("issuer", "Card Network"),
                    "category": metadata.get("category", "CARD_NETWORK_RULE"),
                    "source_url": metadata.get("source_url", "https://usa.visa.com/"),
                    "text": metadata.get("text", "")
                })
            
            if policies:
                _policy_cache[cache_key] = policies
                return policies
        except Exception as e:
            print(f"[WARN] Pinecone query failed ({e}), using local grounded registry...")

    # Fallback to local grounded registry if Pinecone query is unavailable
    fallback_res = _local_fallback_retrieval(dispute_reason)
    _policy_cache[cache_key] = fallback_res
    return fallback_res

def _local_fallback_retrieval(dispute_reason: str) -> list[dict]:
    """Fallback in-memory policy retrieval if Pinecone is offline."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sources_path = os.path.join(root_dir, "data", "policy_sources.json")
    if not os.path.exists(sources_path):
        sources_path = os.path.join(os.path.dirname(__file__), "policy_sources.json")
    if os.path.exists(sources_path):
        with open(sources_path, "r", encoding="utf-8") as f:
            sources = json.load(f)
        
        reason_lower = (dispute_reason or "").lower()
        matched = []
        for s in sources:
            codes = [c.lower() for c in s.get("reason_codes", [])]
            if any(c in reason_lower for c in codes) or any(reason_lower in c for c in codes):
                matched.append({
                    "id": s["id"],
                    "score": 0.95,
                    "name": s["name"],
                    "issuer": s["issuer"],
                    "category": s["category"],
                    "source_url": s["source_url"],
                    "text": s["text"]
                })
        if matched:
            return matched[:2]
        return sources[:2]
    return []

if __name__ == "__main__":
    print("Testing Pinecone Policy Retrieval...")
    results = retrieve_policy_context("Item Not Received", "Credit Card", top_k=2)
    print(f"Retrieved {len(results)} policies:")
    for r in results:
        print(f"\n[{r['issuer']}] {r['name']} (Score: {r['score']})")
        print(f"URL: {r['source_url']}")
        print(f"Clause: {r['text'][:120]}...")
