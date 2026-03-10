import re
import unicodedata


NAVIGATION_TERMS = [
    "home",
    "login",
    "sign up",
    "subscribe",
    "menu",
    "share",
    "download",
    "cookie policy",
    "privacy policy",
    "advertisement",
    "related articles",
]


def normalize_unicode(text):

    text = unicodedata.normalize("NFKC", text)

    replacements = {
        "–": "-",
        "—": "-",
        "“": '"',
        "”": '"',
        "’": "'",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def normalize_spacing(text):

    text = text.replace("\t", " ")

    text = re.sub(r" +", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def is_navigation_line(line):

    line_lower = line.lower()

    for term in NAVIGATION_TERMS:
        if term in line_lower:
            return True

    return False


def preprocess_text(raw_text):

    if not raw_text:
        return ""

    text = raw_text.strip()

    text = normalize_unicode(text)

    text = normalize_spacing(text)

    # detect paragraphs
    paragraphs = text.split("\n\n")

    clean_paragraphs = []

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        lines = paragraph.split("\n")

        lines = [l.strip() for l in lines if l.strip()]

        # remove navigation lines
        lines = [l for l in lines if not is_navigation_line(l)]

        paragraph_clean = " ".join(lines)

        if paragraph_clean:
            clean_paragraphs.append(paragraph_clean)

    clean_text = "\n\n".join(clean_paragraphs)

    return clean_text