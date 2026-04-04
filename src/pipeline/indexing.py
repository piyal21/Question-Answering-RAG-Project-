"""Build the FAISS index from the source PDF.

Run from project root:
    python -m src.pipeline.indexing
"""

from langchain_community.vectorstores import FAISS
from src.config import get_embeddings, PDF_PATH, INDEX_DIR
from src.pipeline.extraction import extract_raw_pdf_text
from src.pipeline.cleaning import clean_text_with_openai
from src.pipeline.chunking import split_into_semantic_chunks


def build_index(pdf_path=PDF_PATH, index_dir=INDEX_DIR):
    print(f"Extracting text from {pdf_path}...")
    raw_text = extract_raw_pdf_text(pdf_path)

    print("Cleaning text with OpenAI...")
    cleaned_text = clean_text_with_openai(raw_text)

    print("Splitting into semantic chunks...")
    chunks = split_into_semantic_chunks(cleaned_text)

    print(f"Embedding {len(chunks)} chunks...")
    embedding = get_embeddings()
    store = FAISS.from_texts(chunks, embedding)
    store.save_local(index_dir)

    print(f"Index saved to {index_dir} ({len(chunks)} chunks)")


if __name__ == "__main__":
    build_index()
