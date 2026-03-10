import re

def detect_paragraphs(text):

    paragraphs = text.split("\n\n")

    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    return paragraphs


def detect_sentences(paragraph):

    sentences = re.split(r'(?<=[.!?])\s+', paragraph)

    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences