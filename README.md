# link_llm_work

User Question + URLs
        │
        ▼
Async Web Scraping
        │
        ▼
Text Cleaning
        │
        ▼
Semantic Chunking
        │
        ▼
Embedding Generation
        │
        ▼
Temporary Vector Index
        │
        ▼
Vector Retrieval (Top 30)
        │
        ▼
Cross Encoder Reranking
        │
        ▼
Top 5–10 Chunks
        │
        ▼
LLM Prompt
        │
        ▼
Final Answer + Reasoning



Stage 1 → Keyword retrieval (BM25)
Stage 2 → Dense retrieval (vector embeddings)
Stage 3 → Neural reranker
Stage 4 → LLM reasoning

https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF