import tiktoken
from structure import detect_paragraphs, detect_sentences

tokenizer = tiktoken.get_encoding("cl100k_base")


def token_count(text):
    return len(tokenizer.encode(text))


def sliding_window_chunks(text, chunk_size=500, overlap=100):

    tokens = tokenizer.encode(text)

    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(tokens), step):

        end = start + chunk_size

        chunk_tokens = tokens[start:end]

        if not chunk_tokens:
            break

        chunk_text = tokenizer.decode(chunk_tokens)

        chunks.append(chunk_text)

    return chunks


def build_chunks(text, chunk_size=500, overlap=100):

    paragraphs = detect_paragraphs(text)

    semantic_chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        sentences = detect_sentences(paragraph)

        for sentence in sentences:

            candidate = current_chunk + " " + sentence

            if token_count(candidate) <= chunk_size:

                current_chunk = candidate

            else:

                if current_chunk:
                    semantic_chunks.append(current_chunk.strip())

                current_chunk = sentence

    if current_chunk:
        semantic_chunks.append(current_chunk.strip())

    final_chunks = []

    for chunk in semantic_chunks:

        if token_count(chunk) > chunk_size:

            final_chunks.extend(
                sliding_window_chunks(chunk, chunk_size, overlap)
            )

        else:

            final_chunks.append(chunk)

    return final_chunks