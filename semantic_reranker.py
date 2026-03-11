import numpy as np
import faiss

from sentence_transformers import SentenceTransformer, CrossEncoder


# -----------------------------
# Load Models
# -----------------------------

# Embedding model for semantic retrieval
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Cross-encoder reranker
reranker_model = CrossEncoder("BAAI/bge-reranker-base")


# -----------------------------
# Embed Text
# -----------------------------

def embed_texts(texts):

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings


# -----------------------------
# Vector Search (FAISS)
# -----------------------------

def vector_search(query, candidate_chunks, top_k=20):

    texts = [c["chunk"] for c in candidate_chunks]

    chunk_embeddings = embed_texts(texts)

    dim = chunk_embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)

    index.add(chunk_embeddings)

    query_embedding = embed_texts([query])

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        results.append({
            "chunk": texts[idx],
            "score": float(score),
            "original_index": candidate_chunks[idx]["index"]
        })

    return results


# -----------------------------
# Cross Encoder Reranker
# -----------------------------

def rerank(query, retrieved_chunks, top_n=5):

    pairs = [(query, r["chunk"]) for r in retrieved_chunks]

    scores = reranker_model.predict(pairs)

    reranked = []

    for r, s in zip(retrieved_chunks, scores):

        r["rerank_score"] = float(s)
        reranked.append(r)

    reranked = sorted(
        reranked,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked[:top_n]


# -----------------------------
# Full Retrieval Pipeline
# -----------------------------

def semantic_retrieve_and_rerank(query, bm25_candidates):

    # Step 1: semantic vector search
    semantic_candidates = vector_search(
        query,
        bm25_candidates,
        top_k=20
    )

    # Step 2: rerank with cross encoder
    final_results = rerank(
        query,
        semantic_candidates,
        top_n=5
    )

    return final_results