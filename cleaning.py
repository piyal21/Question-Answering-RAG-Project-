from config import client, CLEANING_MODEL


def clean_text_with_openai(raw_text):
    """Clean raw extracted Bangla text using OpenAI, preserving semantic meaning."""
    prompt = f"""
The following is raw extracted text from a Bangla literature PDF.
Please clean it up into readable paragraphs, fix broken sentences, and keep the semantic meaning intact.

Raw text:
{raw_text}

Cleaned text:
"""
    response = client.chat.completions.create(
        model=CLEANING_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()
