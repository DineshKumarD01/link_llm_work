from extractor import extract_from_urls
from preprocessing import preprocess_text
from chunking import build_chunks
from bm25_retriever import bm25_retrieve
from semantic_reranker import semantic_retrieve_and_rerank


if __name__ == "__main__":

    urls = [
        "https://jalammar.github.io/illustrated-transformer/",
        "https://arxiv.org/abs/1706.03762",
        "https://www.datacamp.com/tutorial/how-transformers-work",
        "https://arxiv.org/pdf/1706.03762.pdf",
        "https://calibre-ebook.com/downloads/demos/demo.docx"
    ]

    # STEP 1 — Extract documents
    results = extract_from_urls(urls)

    # STEP 2 — Collect all chunks
    all_chunks = []

    for r in results:

        clean_text = preprocess_text(r["content"])

        chunks = build_chunks(clean_text)

        for chunk in chunks:
            all_chunks.append(chunk)

    print("\nTotal chunks:", len(all_chunks))

    # STEP 3 — Query
    query = "How do transformers improve machine translation?"

    # STEP 4 — BM25 retrieval
    candidates = bm25_retrieve(query, all_chunks)

    print("\nBM25 returned:", len(candidates), "chunks")

    # STEP 5 — Semantic retrieval + reranking
    final_chunks = semantic_retrieve_and_rerank(query, candidates)

    print("\nFINAL CHUNKS FOR LLM")

    for i, c in enumerate(final_chunks):

        print("\n====================")
        print("Rank:", i+1)
        print("Rerank Score:", c["rerank_score"])
        print(c["chunk"][:500])