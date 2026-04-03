from config import client, CHUNKING_MODEL, DESIRED_CHUNK_COUNT, CHUNK_WINDOW_SIZE


def _split_into_windows(text, max_size=CHUNK_WINDOW_SIZE):
    """Split text into windows at paragraph boundaries, each under max_size chars."""
    paragraphs = text.split("\n\n")
    windows = []
    current_window = ""

    for para in paragraphs:
        if len(current_window) + len(para) > max_size and current_window:
            windows.append(current_window.strip())
            current_window = para
        else:
            current_window += "\n\n" + para

    if current_window.strip():
        windows.append(current_window.strip())

    return windows


def _chunk_single_window(text, chunks_per_window):
    """Use OpenAI to split a text window into semantic chunks."""
    prompt = f"""
You are a helpful assistant. The following is a cleaned Bangla literature text.

Your task is to split the text into around {chunks_per_window} semantically meaningful chunks.
Each chunk should represent a complete idea, paragraph, or topic without breaking the meaning.
Separate each chunk using this delimiter:
====CHUNK====

Text:
{text}

Semantic Chunks:
"""
    response = client.chat.completions.create(
        model=CHUNKING_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    chunks_text = response.choices[0].message.content.strip()
    return [chunk.strip() for chunk in chunks_text.split("====CHUNK====") if chunk.strip()]


def split_into_semantic_chunks(cleaned_text, desired_chunk_count=DESIRED_CHUNK_COUNT):
    """Split the full cleaned text into semantic chunks by processing it in windows.

    This fixes the original truncation bug where only the first 3500 chars were processed.
    """
    windows = _split_into_windows(cleaned_text)
    chunks_per_window = max(1, desired_chunk_count // len(windows))

    all_chunks = []
    for window in windows:
        chunks = _chunk_single_window(window, chunks_per_window)
        all_chunks.extend(chunks)

    print(f"Created {len(all_chunks)} chunks from {len(windows)} windows")
    return all_chunks
