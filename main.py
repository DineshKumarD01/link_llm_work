from extractor import extract_from_urls
from preprocessing import preprocess_text
from chunking import build_chunks


if __name__ == "__main__":

    urls = [
        "https://jalammar.github.io/illustrated-transformer/",
        "https://arxiv.org/abs/1706.03762",
        "https://www.datacamp.com/tutorial/how-transformers-work",
        "https://arxiv.org/pdf/1706.03762.pdf",
        "https://calibre-ebook.com/downloads/demos/demo.docx"
    ]

    results = extract_from_urls(urls)

    for r in results:

        clean_text = preprocess_text(r["content"])

        chunks = build_chunks(clean_text)

        print("\n====================")
        print("URL:", r["url"])
        print("Chunks:", len(chunks))

        if chunks:
            print(chunks[0][:500])