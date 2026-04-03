import os
from dotenv import load_dotenv
import openai

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not found. Copy .env.example to .env and add your key.")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Paths
PDF_PATH = os.path.join(os.path.dirname(__file__), "data", "HSC26_Bangla_1st_Paper.pdf")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "index", "faiss_index")

# Models
CLEANING_MODEL = "gpt-4o-mini"
CHUNKING_MODEL = "gpt-4o-mini"
GENERATION_MODEL = "gpt-3.5-turbo"

# Embedding (free, local, multilingual — good for Bangla)
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# RAG settings
DESIRED_CHUNK_COUNT = 20
CHUNK_WINDOW_SIZE = 3000
TOP_K = 3
SHORT_TERM_MEMORY_MAX = 5


def get_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
