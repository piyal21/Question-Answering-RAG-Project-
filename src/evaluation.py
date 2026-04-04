"""Evaluate RAG groundedness using cosine similarity between answer and retrieved context.

Run from project root:
    python -m src.evaluation
"""

from sklearn.metrics.pairwise import cosine_similarity
from src.config import get_embeddings
from src.rag.retrieval import retrieve_similar_chunks
from src.rag.generation import generate_answer


def evaluate_groundedness(answer, retrieved_chunks):
    """Compute cosine similarity between the answer and the concatenated retrieved chunks."""
    all_text = " ".join(retrieved_chunks)
    embeddings = get_embeddings()
    answer_embedding = embeddings.embed_query(answer)
    context_embedding = embeddings.embed_query(all_text)
    score = cosine_similarity([answer_embedding], [context_embedding])[0][0]
    return round(score, 3)


if __name__ == "__main__":
    memory = []
    user_query = "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"

    answer = generate_answer(user_query, memory)
    top_chunks = retrieve_similar_chunks(user_query)
    score = evaluate_groundedness(answer, top_chunks)

    print(f"Query: {user_query}")
    print(f"Answer: {answer}")
    print(f"Groundedness Score: {score}")
