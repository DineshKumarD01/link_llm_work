import requests
import trafilatura
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright
from pypdf import PdfReader
from docx import Document
import tempfile
import os
from preprocessing import preprocess_text

# -------------------------
# Detect file type
# -------------------------
def detect_file_type(url):

    parsed = urlparse(url)
    path = parsed.path.lower()

    if path.endswith(".pdf"):
        return "pdf"

    if path.endswith(".docx"):
        return "docx"

    return "html"


# -------------------------
# HTML extraction
# -------------------------
def extract_html_requests(url):

    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, headers=headers, timeout=15)

        html = response.text

        text = trafilatura.extract(html)

        if text and "Enable JavaScript" not in text:
            return text

        # fallback using reader proxy
        reader_url = "https://r.jina.ai/" + url
        print("using jina reader proxy:", reader_url)

        response = requests.get(reader_url, timeout=15)

        return response.text

    except:
        return None


# -------------------------
# HTML extraction (JS sites)
# -------------------------
def extract_html_playwright(url):

    with sync_playwright() as p:

        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(url, timeout=60000)

        page.wait_for_timeout(3000)

        html = page.content()

        browser.close()

    text = trafilatura.extract(html)

    return text


# -------------------------
# PDF extraction
# -------------------------
def extract_pdf(url):

    try:

        response = requests.get(url, timeout=20)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(response.content)
            temp_path = f.name

        reader = PdfReader(temp_path)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        os.remove(temp_path)

        return text

    except Exception as e:
        print("PDF extraction error:", e)
        return None


# -------------------------
# DOCX extraction
# -------------------------
def extract_docx(url):

    try:

        response = requests.get(url, timeout=20)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as f:
            f.write(response.content)
            temp_path = f.name

        doc = Document(temp_path)

        text = ""

        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"

        os.remove(temp_path)

        return text

    except Exception as e:
        print("DOCX extraction error:", e)
        return None


# -------------------------
# Main extraction function
# -------------------------
def extract_content(url):

    file_type = detect_file_type(url)

    print("Detected type:", file_type)

    if file_type == "pdf":
        return extract_pdf(url)

    if file_type == "docx":
        return extract_docx(url)

    # HTML
    text = extract_html_requests(url)

    if text:
        return text

    # fallback for JS websites
    print("Using browser rendering...")
    return extract_html_playwright(url)


# -------------------------
# Extract multiple URLs
# -------------------------
def extract_from_urls(urls):

    results = []

    for url in urls:

        print("\nProcessing:", url)

        content = extract_content(url)

        results.append({
            "url": url,
            "content": content
        })

    return results