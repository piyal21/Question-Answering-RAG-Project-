# Bangla Literature Q&A — RAG Pipeline

A Retrieval-Augmented Generation (RAG) system that answers questions about HSC Bangla 1st Paper using OpenAI GPT and FAISS vector search.

## RAG Pipeline

```
PDF → Extract (PyMuPDF) → Clean (GPT-4o-mini) → Chunk (GPT-4o-mini) → Embed (HuggingFace) → FAISS Index
                                                                                                    ↓
User Query → Embed → Similarity Search (FAISS) → Top-K Chunks → Generate Answer (GPT-3.5-turbo) → Response
```

## Setup

1. Clone the repository
2. Create a virtual environment and activate it
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file from the template:
   ```bash
   cp .env.example .env
   ```
5. Add your OpenAI API key to `.env`

## Usage

### Step 1: Build the FAISS Index (run once)
```bash
python indexing.py
```
This extracts text from the PDF, cleans it, chunks it semantically, embeds with HuggingFace, and saves the FAISS index.

### Step 2: Run the Q&A System

**Demo script:**
```bash
python main.py
```

**Streamlit UI:**
```bash
streamlit run app.py
```

**Flask API:**
```bash
python api.py
```
Then send a POST request:
```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"}'
```

### Evaluate Groundedness
```bash
python evaluation.py
```

## Project Structure

| File | Description |
|------|-------------|
| `config.py` | Central configuration (env, API client, constants) |
| `extraction.py` | PDF text extraction using PyMuPDF |
| `cleaning.py` | Text cleaning via OpenAI (preserves Bangla semantics) |
| `chunking.py` | Semantic chunking via OpenAI (processes full document) |
| `indexing.py` | CLI script to build the FAISS index |
| `retrieval.py` | FAISS similarity search with lazy-loaded index |
| `generation.py` | Answer generation with short-term memory |
| `evaluation.py` | Groundedness scoring via cosine similarity |
| `main.py` | Interactive demo script |
| `api.py` | Flask REST API (stateless) |
| `app.py` | Streamlit chat UI |

## Tools Used

| Component | Tool |
|-----------|------|
| Language Model | OpenAI GPT-3.5-turbo (generation), GPT-4o-mini (cleaning/chunking) |
| Embeddings | HuggingFace `paraphrase-multilingual-MiniLM-L12-v2` (free, local) |
| Vector Store | FAISS |
| Document Parsing | PyMuPDF |
| API Framework | Flask |
| UI Framework | Streamlit |

## Sample Q&A

- **Query:** বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?
- **Answer:** ১৫ বছর

## API Documentation

**POST** `/ask`

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | The question to answer |

**Response:**
```json
{
  "question": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
  "answer": "১৫ বছর"
}
```
