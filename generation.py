from config import client, GENERATION_MODEL, SHORT_TERM_MEMORY_MAX
from retrieval import retrieve_similar_chunks


def update_short_term_memory(memory, user_question, bot_answer, max_len=SHORT_TERM_MEMORY_MAX):
    """Append a Q&A pair to the memory list, keeping only the last max_len entries."""
    if len(memory) >= max_len:
        memory.pop(0)
    memory.append(f"User: {user_question}\nBot: {bot_answer}")


def generate_answer(query, short_term_memory, top_k=3):
    """Retrieve relevant context and generate an answer using OpenAI."""
    relevant_texts = retrieve_similar_chunks(query, top_k=top_k)
    context = "\n\n".join(relevant_texts)
    short_memory_context = (
        "\n".join(short_term_memory) if short_term_memory else "No prior chat history."
    )

    prompt = f"""
You are a knowledgeable assistant for Bangla literature Q&A.
Answer the user's question based on the context from a textbook.
Use both retrieved document and chat history if needed.
Do not answer if you don't know the exact answer.

Chat History:
{short_memory_context}

Context:
{context}

User Question:
{query}

Answer very briefly, in one short phrase or word, directly and factually.
"""

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": "You're a helpful and precise Bangla literature expert."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    answer = response.choices[0].message.content.strip()
    update_short_term_memory(short_term_memory, query, answer)
    return answer
