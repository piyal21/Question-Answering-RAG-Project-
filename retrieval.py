import os
from langchain_community.vectorstores import FAISS
from config import get_embeddings, INDEX_DIR, TOP_K

_vector_store = None


def _load_store():
    """Lazy-load the FAISS index (loaded once, cached for subsequent queries)."""
    global _vector_store
    if _vector_store is None:
        index_path = os.path.join(INDEX_DIR, "index.faiss")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {INDEX_DIR}. Run: python indexing.py"
            )
        _vector_store = FAISS.load_local(
            INDEX_DIR,
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return _vector_store


def retrieve_similar_chunks(user_query, top_k=TOP_K):
    """Retrieve the top-k most similar chunks for a given query."""
    store = _load_store()
    results = store.similarity_search(user_query, k=top_k)
    return [doc.page_content for doc in results]
