"""Full pipeline demo.

Prerequisites:
    1. pip install -r requirements.txt
    2. Copy .env.example to .env and add your OpenAI API key
    3. python -m src.pipeline.indexing  (build the FAISS index first)
"""

from src.rag.generation import generate_answer


def main():
    memory = []

    print("=== Bangla Literature Q&A Demo ===\n")

    query = "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"
    answer = generate_answer(query, memory)
    print(f"User: {query}")
    print(f"Bot:  {answer}\n")

    # Interactive mode
    print("Ask more questions (type 'exit' to quit):\n")
    while True:
        query = input("User: ").strip()
        if not query or query.lower() == "exit":
            break
        answer = generate_answer(query, memory)
        print(f"Bot:  {answer}\n")


if __name__ == "__main__":
    main()
