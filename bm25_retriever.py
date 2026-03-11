from rank_bm25 import BM25Okapi
import numpy as np


def tokenize(text):
    return text.lower().split()


def build_bm25_index(chunks):

    tokenized_chunks = [tokenize(chunk) for chunk in chunks]

    bm25 = BM25Okapi(tokenized_chunks)

    return bm25, tokenized_chunks


def compute_candidate_size(total_chunks):

    candidates = int(total_chunks * 0.05)

    candidates = max(50, candidates)

    candidates = min(300, candidates)

    return candidates


def bm25_retrieve(query, chunks):

    bm25, tokenized_chunks = build_bm25_index(chunks)

    tokenized_query = tokenize(query)

    scores = bm25.get_scores(tokenized_query)

    total_chunks = len(chunks)

    top_k = compute_candidate_size(total_chunks)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in top_indices:

        results.append({
            "chunk": chunks[idx],
            "score": scores[idx],
            "index": idx
        })

    return results



# EXPLANATION OF BM25 RETRIEVAL CODE
# ----------------------------------

# This function retrieves the most relevant chunks for a query using the BM25 algorithm.

# FUNCTION:
# bm25_retrieve(query, chunks)

# INPUT
# -----
# query  : user question
# chunks : list of text chunks extracted from documents

# OUTPUT
# ------
# Top candidate chunks ranked by BM25 score


# STEP 1 — BUILD BM25 INDEX
# -------------------------
# bm25, tokenized_chunks = build_bm25_index(chunks)

# What happens:

# Each chunk is tokenized into words.

# Example chunks:

# Chunk0:
# "transformers improve machine translation"

# Chunk1:
# "self attention mechanism in transformers"

# Chunk2:
# "reinforcement learning algorithms"

# Tokenized form:

# [
#  ['transformers','improve','machine','translation'],
#  ['self','attention','mechanism','transformers'],
#  ['reinforcement','learning','algorithms']
# ]

# The BM25 index stores:
# - term frequency of words in each chunk
# - document length
# - inverse document frequency


# STEP 2 — TOKENIZE QUERY
# -----------------------
# tokenized_query = tokenize(query)

# Example query:

# "How do transformers improve translation?"

# Tokenized:

# ['transformers','improve','translation']


# STEP 3 — COMPUTE BM25 SCORES
# ----------------------------
# scores = bm25.get_scores(tokenized_query)

# BM25 computes a relevance score for each chunk.

# BM25 formula:

# score(D,Q) =
# Σ IDF(q_i) * ((f(q_i,D)*(k1+1)) /
#              (f(q_i,D)+k1*(1-b + b*(|D|/avgdl))))

# Where:

# q_i   = query term
# f(q_i,D) = frequency of query word in chunk
# |D|   = length of chunk
# avgdl = average document length
# k1,b  = tuning parameters

# Typical values:
# k1 = 1.5
# b  = 0.75


# EXAMPLE SCORING
# ---------------

# Query:
# "transformers improve translation"

# Chunks:

# Chunk0: transformers improve machine translation
# Chunk1: self attention mechanism transformers
# Chunk2: reinforcement learning algorithms

# Term frequencies:

# word           Chunk0   Chunk1   Chunk2
# ---------------------------------------
# transformers     1        1        0
# improve          1        0        0
# translation      1        0        0

# BM25 scores (example):

# Chunk0 → 3.2
# Chunk1 → 1.1
# Chunk2 → 0.0


# STEP 4 — COUNT TOTAL CHUNKS
# ---------------------------
# total_chunks = len(chunks)

# Example:

# 3 chunks


# STEP 5 — DECIDE HOW MANY CANDIDATES
# -----------------------------------
# top_k = compute_candidate_size(total_chunks)

# Dynamic rule used:

# top_k = min(300, max(50, total_chunks * 0.05))

# Example results:

# Total Chunks     BM25 Candidates
# --------------------------------
# 200              50
# 1000             50
# 8000             300


# STEP 6 — SORT BY SCORE
# ----------------------
# top_indices = np.argsort(scores)[::-1][:top_k]

# Example:

# scores = [3.2, 1.1, 0.0]

# np.argsort(scores)

# returns indices sorted ascending:

# [2,1,0]

# Reverse order:

# [0,1,2]

# Top results:

# [0,1]


# STEP 7 — BUILD RESULT OBJECTS
# ------------------------------
# results.append({
#     "chunk": chunks[idx],
#     "score": scores[idx],
#     "index": idx
# })

# Example result:

# [
#  {
#    "chunk": "transformers improve machine translation",
#    "score": 3.2,
#    "index": 0
#  },
#  {
#    "chunk": "self attention mechanism transformers",
#    "score": 1.1,
#    "index": 1
#  }
# ]


# STEP 8 — RETURN RESULTS
# -----------------------
# return results

# The function returns the top BM25 candidate chunks.


# WHAT HAPPENS AFTER BM25
# -----------------------
# BM25 is only the first stage of retrieval.

# Typical pipeline:

# 8000 chunks
# ↓
# BM25 retrieval → 200 candidates
# ↓
# Vector embedding search → 20 chunks
# ↓
# Neural reranker → 5 chunks
# ↓
# LLM context


# WHY BM25 IS USED FIRST
# ----------------------
# Vector search complexity:

# O(N × embedding_dimension)

# BM25 complexity:

# O(N × query_terms)

# Since embedding dimensions are large (768–3072), vector search is more expensive.

# Therefore BM25 acts as a fast filtering stage.


# SUMMARY
# -------
# BM25 performs lexical retrieval using keyword statistics.
# It ranks chunks using TF-IDF style scoring.

# The retrieved candidates are then passed to:

# 1. Vector similarity search
# 2. Neural reranker
# 3. LLM for final answer generation


################################# BM25
# DETAILED EXPLANATION OF THE BM25 FORMULA
# ----------------------------------------

# BM25 scoring formula:

# score(D,Q) =
# Σ IDF(q_i) * ((f(q_i,D)*(k1+1)) /
#              (f(q_i,D)+k1*(1-b + b*(|D|/avgdl))))

# This formula calculates how relevant a document (or chunk) D is to a query Q.


# 1. WHAT THE FORMULA IS DOING
# ----------------------------

# The score is calculated for each query word q_i.

# Then all scores are summed.

# So if the query has words:

# Query = ["transformers", "improve", "translation"]

# BM25 calculates:

# score(D,Q) =
# score(transformers) +
# score(improve) +
# score(translation)


# 2. IDF(q_i) — INVERSE DOCUMENT FREQUENCY
# ----------------------------------------

# IDF measures how rare a word is across the entire corpus.

# Formula conceptually:

# IDF(q) = log((N - n(q) + 0.5) / (n(q) + 0.5))

# Where:

# N = total number of documents
# n(q) = number of documents containing the word

# Example:

# Corpus:

# Doc1: transformers improve translation
# Doc2: transformers use attention
# Doc3: machine learning models

# Word: "transformers"

# n(q) = 2
# N = 3

# If a word appears in many documents → IDF becomes small.

# If a word appears in few documents → IDF becomes large.

# Meaning:

# Rare words carry more importance.


# 3. f(q_i, D) — TERM FREQUENCY
# ------------------------------

# f(q_i, D) is the number of times the query word appears in the document.

# Example document:

# Doc:
# "transformers transformers improve translation"

# Word frequencies:

# transformers = 2
# improve = 1
# translation = 1

# Higher frequency means the document likely discusses that topic.


# 4. WHY (k1 + 1) IS USED
# -----------------------

# The term:

# (f(q_i,D)*(k1+1))

# controls how much importance term frequency has.

# However, BM25 does not allow frequency to grow indefinitely.

# Example problem without saturation:

# Word appearing 20 times would dominate the score.

# But after some point, repeating the word doesn't add new meaning.

# So BM25 limits growth using the denominator.


# 5. THE DENOMINATOR PART
# -----------------------

# Denominator:

# f(q_i,D) + k1*(1-b + b*(|D|/avgdl))

# This controls:

# 1) term frequency saturation
# 2) document length normalization


# 6. DOCUMENT LENGTH NORMALIZATION
# --------------------------------

# |D| = length of the document (or chunk)

# avgdl = average document length across all documents

# Example:

# Chunk lengths:

# Chunk1 = 100 words
# Chunk2 = 500 words
# Chunk3 = 50 words

# Average length:

# avgdl = (100 + 500 + 50) / 3 = 216


# The ratio:

# |D| / avgdl

# Example:

# Chunk1 → 100/216 = 0.46
# Chunk2 → 500/216 = 2.31
# Chunk3 → 50/216 = 0.23


# Meaning:

# Longer documents are penalized slightly because they naturally contain more words.


# 7. PARAMETER b
# --------------

# b controls how strongly document length affects scoring.

# Range:

# 0 ≤ b ≤ 1

# Typical value:

# b = 0.75

# Meaning:

# 75% weight given to document length normalization.

# Special cases:

# b = 0   → ignore document length
# b = 1   → full normalization


# 8. PARAMETER k1
# ---------------

# k1 controls how term frequency grows.

# Typical value:

# k1 = 1.5

# Effect:

# Small k1 → term frequency saturates quickly
# Large k1 → term frequency grows more


# Example:

# Word frequency increases:

# transformers = 1 → score increases
# transformers = 2 → increases slightly
# transformers = 10 → almost no further gain


# 9. FULL NUMERICAL EXAMPLE
# -------------------------

# Query:
# "transformers improve translation"

# Document:

# "transformers improve machine translation"

# Word counts:

# transformers = 1
# improve = 1
# translation = 1

# Assume:

# k1 = 1.5
# b = 0.75
# |D| = 4
# avgdl = 5

# Compute denominator component:

# 1 - b + b*(|D|/avgdl)

# = 1 - 0.75 + 0.75*(4/5)

# = 0.25 + 0.75*(0.8)

# = 0.25 + 0.6

# = 0.85


# Now compute BM25 term part:

# (f*(k1+1)) / (f + k1*(0.85))

# f = 1

# (1*(1.5+1)) / (1 + 1.5*(0.85))

# = 2.5 / (1 + 1.275)

# = 2.5 / 2.275

# ≈ 1.098


# Multiply by IDF.

# Assume:

# IDF(transformers) = 1.2

# Final score contribution:

# 1.2 * 1.098 = 1.317


# This is done for each query term and summed.


# 10. FINAL INTERPRETATION
# ------------------------

# BM25 score increases when:

# - query words appear in the document
# - words are rare across documents
# - word frequency increases moderately
# - document length is reasonable

# BM25 score decreases when:

# - document is extremely long
# - query terms are missing
# - words are very common across corpus